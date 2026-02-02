import torch
from split_inference.cnn_model import SimpleCNN, DeepCNN, MiniResNet
from integrated_system.inference_engine import MultiTaskRunner
from integrated_system.resource_manager import ResourceManager
from rl_pdnn.agent import DQNAgent
from rl_pdnn.multi_agent_env import MultiAgentIoTEnv
import numpy as np
import os

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
    print("\n[Stage 3] Preparing PyTorch Models for Split Inference...")
    # Instantiate the 3 models as updated in split_inference/cnn_model.py
    models = [SimpleCNN(), DeepCNN(), MiniResNet()]
    
    # 5. Execute Multi-Task Simulation
    print("\n[Stage 4] Running Distributed Multi-Task Inference...")
    
    for i, m_type in enumerate(MODEL_TYPES):
        print(f"\n--- Task {i} ({m_type}) ---")
        map_ = allocation_maps[i]
        model = models[i]
        print(f"  Allocation: {map_}")
        
        # Check if map length matches model depth
        if len(map_) != len(model.layers):
            print(f"  [Warning] Map length ({len(map_)}) != Model layers ({len(model.layers)}).")
            # In a robust system, we would handle padding/truncation here.
        
        # All models now follow the same execution pattern via MultiTaskRunner
        print(f"  Executing {m_type} Structure (Split Inference)")
        runner = MultiTaskRunner(model)
        
        # Standard input for MNIST-style tasks
        dummy_input = torch.randn(1, 1, 28, 28)

        try:
            output = runner.run(dummy_input, map_)
            pred = torch.argmax(output).item()
            print(f"  Result (Random Weights): Predicted Class {pred}")
        except Exception as e:
            print(f"  Execution Failed for {m_type}: {e}")

    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    run_multi_agent_demo()

if __name__ == "__main__":
    run_multi_agent_demo()
