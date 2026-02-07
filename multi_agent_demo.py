import torch
from split_inference.cnn_model import SimpleCNN, DeepCNN, MiniResNet
from integrated_system.inference_engine import MultiTaskRunner
from integrated_system.resource_manager import ResourceManager
from rl_pdnn.agent import DQNAgent
from rl_pdnn.multi_agent_env import MultiAgentIoTEnv
import numpy as np
import os
from PIL import Image
# removed torchvision dependency
import matplotlib.pyplot as plt

def run_multi_agent_demo():
    print("=== Multi-Agent RL-PDNN Demo (SimpleCNN, DeepCNN, MiniResNet) ===")
    
    NUM_AGENTS = 3
    NUM_DEVICES = 5
    # Aligned with rl_pdnn/utils.py model names
    MODEL_TYPES = ["simplecnn", "deepcnn", "miniresnet"]
    
    # 1. Setup Environment
    env = MultiAgentIoTEnv(num_agents=NUM_AGENTS, num_devices=NUM_DEVICES, model_types=MODEL_TYPES)
    
    # 2. Load Agents
    agents = []
    print("\n[Stage 1] Loading RL Agents...")
    for i in range(NUM_AGENTS):
        agent = DQNAgent(state_dim=env.single_state_dim, action_dim=NUM_DEVICES)
        path = f"rl_pdnn/models/agent_{i}.pth"
        if os.path.exists(path):
            agent.load(path)
            print(f"  Agent {i} loaded from {path}")
        else:
            print(f"  Warning: Agent {i} model not found. Using random policy.")
        agents.append(agent)
        
    # 3. Generate Allocation Maps
    print("\n[Stage 2] Scheduling Inference Tasks via RL...")
    states = env.reset()
    
    # Storage for decision maps
    allocation_maps = [[] for _ in range(NUM_AGENTS)]
    dones = [False] * NUM_AGENTS
    
    step_count = 0
    while not all(dones):
        actions = []
        for i in range(NUM_AGENTS):
            if dones[i]:
                actions.append(0)
            else:
                action = agents[i].act(states[i])
                actions.append(action)
                allocation_maps[i].append(action)
        
        next_states, rewards, now_dones, _ = env.step(actions)
        states = next_states
        dones = now_dones
        step_count += 1
        
    for i in range(NUM_AGENTS):
        print(f"  Agent {i} ({MODEL_TYPES[i]}) Map: {allocation_maps[i]}")
        
    # 4. Load Models (Instantiate Real Architectures from split_inference)
    print("\n[Stage 3] Preparing PyTorch Models with Trained Weights...")
    models = [SimpleCNN(), DeepCNN(), MiniResNet()]
    weight_paths = [
        "split_inference/mnist_simplecnn.pth",
        "split_inference/mnist_deepcnn.pth",
        "split_inference/mnist_miniresnet.pth"
    ]
    
    # Mapping old state_dict keys to new structure
    mappings = [
        # SimpleCNN
        {
            "conv1.weight": "layers.0.0.weight", "conv1.bias": "layers.0.0.bias",
            "conv2.weight": "layers.1.0.weight", "conv2.bias": "layers.1.0.bias",
            "fc1.weight": "layers.3.0.weight", "fc1.bias": "layers.3.0.bias",
            "fc2.weight": "layers.4.weight", "fc2.bias": "layers.4.bias"
        },
        # DeepCNN
        {
            "conv1.weight": "layers.0.0.weight", "conv1.bias": "layers.0.0.bias",
            "conv2.weight": "layers.0.2.weight", "conv2.bias": "layers.0.2.bias",
            "conv3.weight": "layers.1.0.weight", "conv3.bias": "layers.1.0.bias",
            "conv4.weight": "layers.1.2.weight", "conv4.bias": "layers.1.2.bias",
            "fc1.weight": "layers.3.0.weight", "fc1.bias": "layers.3.0.bias",
            "fc2.weight": "layers.4.weight", "fc2.bias": "layers.4.bias"
        },
        # MiniResNet
        {
            "conv1.weight": "layers.0.0.weight", "conv1.bias": "layers.0.0.bias",
            "conv2.weight": "layers.0.2.weight", "conv2.bias": "layers.0.2.bias",
            "conv3.weight": "layers.1.conv.weight", "conv3.bias": "layers.1.conv.bias",
            "fc1.weight": "layers.3.0.weight", "fc1.bias": "layers.3.0.bias",
            "fc2.weight": "layers.4.weight", "fc2.bias": "layers.4.bias"
        }
    ]
    
    for i, path in enumerate(weight_paths):
        if os.path.exists(path):
            try:
                state_dict = torch.load(path, map_location=torch.device('cpu'))
                new_state_dict = {}
                mapping = mappings[i]
                for old_key, val in state_dict.items():
                    if old_key in mapping:
                        new_state_dict[mapping[old_key]] = val
                    else:
                        new_state_dict[old_key] = val
                
                models[i].load_state_dict(new_state_dict, strict=False)
                print(f"  Weights for {MODEL_TYPES[i]} loaded and remapped from {path}")
            except Exception as e:
                print(f"  Warning: Could not load weights for {MODEL_TYPES[i]}: {e}")
        else:
            print(f"  Warning: Weight file {path} not found.")

    # 5. Execute Multi-Task Simulation
    print("\n[Stage 4] Running Distributed Multi-Task Inference...")
    
    # --- Input Preparation ---
    # Testing with the processed image from the results folder as requested
    fixed_image_path = "results/processed_test_input.png"
    
    if os.path.exists(fixed_image_path):
        print(f"\n[Info] Loading real test image: {fixed_image_path}")
        # Load image, convert to grayscale, resize to 28x28
        image = Image.open(fixed_image_path).convert('L').resize((28, 28))
        
        # Convert to numpy and normalize (to match transforms.ToTensor() and transforms.Normalize)
        img_np = np.array(image).astype(np.float32) / 255.0  # [0, 1] range
        img_np = (img_np - 0.1307) / 0.3081  # MNIST mean/std
        
        # Convert to torch tensor [Batch, Channel, H, W]
        shared_dummy_input = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    else:
        print("\n[Info] Real test image not found. Using random dummy input.")
        torch.manual_seed(42)
        shared_dummy_input = torch.randn(1, 1, 28, 28)
    
    # Save the processed image to results for verification
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Reshape and normalize for saving (reverse normalization visual-only)
    img_np = shared_dummy_input.squeeze().detach().numpy()
    plt.imsave("results/processed_test_input.png", img_np, cmap='gray')
    print(f"  Processed image saved to results/processed_test_input.png")
    
    for i, m_type in enumerate(MODEL_TYPES):
        print(f"\n--- Task {i} ({m_type}) ---")
        map_ = allocation_maps[i]
        model = models[i]
        print(f"  Allocation: {map_}")
        
        if len(map_) != len(model.layers):
            print(f"  [Warning] Map length ({len(map_)}) != Model layers ({len(model.layers)}).")
        
        print(f"  Executing {m_type} Structure (Split Inference)")
        runner = MultiTaskRunner(model)
        
        try:
            output = runner.run(shared_dummy_input, map_)
            pred = torch.argmax(output).item()
            print(f"  Result: Predicted Class {pred}")
        except Exception as e:
            print(f"  Execution Failed for {m_type}: {e}")

    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    run_multi_agent_demo()
