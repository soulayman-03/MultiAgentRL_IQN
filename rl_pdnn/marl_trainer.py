import torch
import numpy as np
from rl_pdnn.agent import DQNAgent
from rl_pdnn.multi_agent_env import MultiAgentIoTEnv
from split_inference.cnn_model import SimpleCNN, DeepCNN, MiniResNet
import os

def train_marl():
    NUM_AGENTS = 3
    NUM_DEVICES = 5
    # Corresponds to SimpleCNN, DeepCNN, MiniResNet
    MODEL_TYPES = ["simplecnn", "deepcnn", "miniresnet"]
    EPISODES = 500
    
    env = MultiAgentIoTEnv(num_agents=NUM_AGENTS, num_devices=NUM_DEVICES, model_types=MODEL_TYPES)
    
    agents = []
    
    # Instantiate the actual models
    cv_models = [SimpleCNN(), DeepCNN(), MiniResNet()]
    
    # Try to load weights if they exist
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
                    print(f"Agent {i}: Loaded {model_paths[i]}")
                except Exception as e:
                    print(f"Agent {i}: Failed to load weights ({e})")
            else:
                 print(f"Agent {i}: No weights found at {model_paths[i]}, using random init.")
            
            agent.assign_inference_model(cv_model)
            
        agents.append(agent)
    
    print(f"Starting MARL Training for {NUM_AGENTS} Agents...")
    
    # History for plotting
    # shape: (EPISODES, NUM_AGENTS)
    history_rewards = np.zeros((EPISODES, NUM_AGENTS))
    
    for e in range(EPISODES):
        states = env.reset()
        episode_rewards = [0] * NUM_AGENTS
        dones = [False] * NUM_AGENTS
        
        while not all(dones):
            actions = []
            for i in range(NUM_AGENTS):
                if dones[i]:
                    actions.append(0) 
                else:
                    action = agents[i].act(states[i])
                    actions.append(action)
            
            next_states, rewards, now_dones, _ = env.step(actions)
            
            for i in range(NUM_AGENTS):
                if not dones[i]:
                    agents[i].remember(states[i], actions[i], rewards[i], next_states[i], now_dones[i])
                    agents[i].replay()
                    episode_rewards[i] += rewards[i]
            
            states = next_states
            dones = now_dones
            
        # Log rewards
        for i in range(NUM_AGENTS):
            history_rewards[e, i] = episode_rewards[i]
            
        if e % 10 == 0:
            avg_rew = sum(episode_rewards) / NUM_AGENTS
            print(f"Episode {e}/{EPISODES} | Avg Reward: {avg_rew:.2f} | Epsilon: {agents[0].epsilon:.2f}")

    if not os.path.exists("rl_pdnn/models"):
        os.makedirs("rl_pdnn/models", exist_ok=True)
        
    for i in range(NUM_AGENTS):
        agents[i].save(f"rl_pdnn/models/agent_{i}.pth")
    
    # Save history
    np.save("rl_pdnn/models/train_history.npy", history_rewards)
    print("MARL Training Complete. Models and History saved.")

if __name__ == "__main__":
    train_marl()
