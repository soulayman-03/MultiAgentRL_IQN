import numpy as np
import matplotlib.pyplot as plt
from .multi_agent_env import MultiAgentIoTEnv
from .agent import DQNAgent
from split_inference.cnn_model import SimpleCNN, DeepCNN, MiniResNet
import torch
import os

def evaluate():
    NUM_EPISODES = 100
    NUM_AGENTS = 3
    NUM_DEVICES = 5
    MODEL_DIR = "rl_pdnn/models"
    # Corresponds to SimpleCNN, DeepCNN, MiniResNet
    MODEL_TYPES = ["simplecnn", "deepcnn", "miniresnet"]
    RESULTS_DIR = "results"
    
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    print("Starting MARL Evaluation...")
    
    env = MultiAgentIoTEnv(num_agents=NUM_AGENTS, num_devices=NUM_DEVICES, model_types=MODEL_TYPES)
    
    agents = []
    agents = []
    
    # Instantiate the actual models
    cv_models = [SimpleCNN(), DeepCNN(), MiniResNet()]
    model_paths = [
        "split_inference/mnist_simplecnn.pth",
        "split_inference/mnist_deepcnn.pth",
        "split_inference/mnist_miniresnet.pth"
    ]
    
    for i in range(NUM_AGENTS):
        agent = DQNAgent(state_dim=env.single_state_dim, action_dim=NUM_DEVICES)
        
        # Assign the specific model
        if i < len(cv_models):
            cv_model = cv_models[i]
            # Load weights if available
            if os.path.exists(model_paths[i]):
                try:
                    cv_model.load_state_dict(torch.load(model_paths[i]))
                    print(f"Agent {i}: Loaded Inference Model {model_paths[i]}")
                except Exception as e:
                    print(f"Agent {i}: Failed to load Inference weights ({e})")
            
            agent.assign_inference_model(cv_model)
        
        model_path = f"{MODEL_DIR}/agent_{i}.pth"
        if os.path.exists(model_path):
            agent.load(model_path)
            print(f"Loaded Agent {i} ({MODEL_TYPES[i]})")
        else:
            print(f"WARNING: Model {i} not found. Random weights.")
        agent.epsilon = 0.0
        agents.append(agent)
    
    agent_scores = [[] for _ in range(NUM_AGENTS)]
    agent_latencies = [[] for _ in range(NUM_AGENTS)]
    
    # For Visualization of one episode flow
    sample_episode_data = None
    
    print(f"Running {NUM_EPISODES} test episodes...")
    
    for e in range(NUM_EPISODES):
        states = env.reset()
        dones = [False] * NUM_AGENTS
        episode_rewards = [0.0] * NUM_AGENTS
        
        # Trace for sample episode (the last one)
        episode_trace = { "agents": [ [] for _ in range(NUM_AGENTS) ] }
        
        while not all(dones):
            actions = []
            for i in range(NUM_AGENTS):
                if dones[i]:
                    actions.append(0)
                else:
                    action = agents[i].act(states[i])
                    actions.append(action)
                    # Log step: (layer_idx, device_id)
                    current_layer_idx = env.agents_progress[i]
                    episode_trace["agents"][i].append( (current_layer_idx, int(action)) )
            
            next_states, rewards, now_dones, _ = env.step(actions)
            
            for i in range(NUM_AGENTS):
                if not dones[i]:
                    episode_rewards[i] += rewards[i]
            
            states = next_states
            dones = now_dones
            
        for i in range(NUM_AGENTS):
            agent_scores[i].append(episode_rewards[i])
            agent_latencies[i].append(-episode_rewards[i])
            
        if e == NUM_EPISODES - 1:
            sample_episode_data = episode_trace

    # --- PLOTTING ---
    print("Generating Plots...")
    
    # 1. Latency Distribution Boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(agent_latencies, labels=MODEL_TYPES)
    plt.title("Latency Distribution per Agent (100 Test Episodes)")
    plt.ylabel("Latency (ms)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{RESULTS_DIR}/latency_boxplot.png")
    plt.close()
    
    # 2. Training Convergence (If history exists)
    history_path = f"{MODEL_DIR}/train_history.npy"
    if os.path.exists(history_path):
        history = np.load(history_path) # (Episodes, Agents)
        plt.figure(figsize=(12, 6))
        
        # Moving average
        window = 20
        for i in range(NUM_AGENTS):
            raw = history[:, i]
            smoothed = np.convolve(raw, np.ones(window)/window, mode='valid')
            plt.plot(smoothed, label=f"Agent {i} ({MODEL_TYPES[i]})")
            
        plt.title("Training Convergence (Moving Average)")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{RESULTS_DIR}/training_curve.png")
        plt.close()
        print(" - Training curve plotted.")
    
    # 3. Sample Episode Flow (Allocation Map)
    plt.figure(figsize=(12, 6))
    colors = ['r', 'g', 'b']
    markers = ['o', 's', '^']
    
    for i in range(NUM_AGENTS):
        trace = sample_episode_data["agents"][i]
        layers = [t[0] for t in trace]
        devices = [t[1] for t in trace]
        plt.plot(layers, devices, label=f"{MODEL_TYPES[i]}", color=colors[i], marker=markers[i], linestyle='-')
    
    plt.yticks(range(NUM_DEVICES), [f"Device {d}" for d in range(NUM_DEVICES)])
    plt.xlabel("Layer Index")
    plt.title("Resource Allocation Strategy (Sample Episode)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/execution_flow.png")
    plt.close()
    
    print(f"Plots saved to {RESULTS_DIR}/")

    # Generate Text Report (Same as before)
    global_avg = np.mean([np.mean(s) for s in agent_scores])
    with open(f"{RESULTS_DIR}/PERFORMANCE_REPORT.md", "w") as f:
        f.write("# RL-PDNN Visual Report\n")
        f.write("## Visualizations\n")
        f.write("![Latency Boxplot](latency_boxplot.png)\n")
        if os.path.exists(history_path):
            f.write("![Training Curve](training_curve.png)\n")
        f.write("![Execution Flow](execution_flow.png)\n")
        f.write(f"\n**Global Score**: {global_avg:.2f}\n")
    
    print("Evaluation Complete.")

if __name__ == "__main__":
    evaluate()
