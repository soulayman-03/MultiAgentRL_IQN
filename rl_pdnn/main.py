import numpy as np
from rl_pdnn.env import IoTEnv
from rl_pdnn.agent import DQNAgent
import matplotlib.pyplot as plt

def main():
    # Simulation Parameters
    NUM_EPISODES = 500
    NUM_DEVICES = 5
    NUM_LAYERS = 10
    
    # Initialize Environment and Agent
    env = IoTEnv(num_devices=NUM_DEVICES, num_layers=NUM_LAYERS)
    state_dim = env.state_dim
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    
    scores = []
    
    print(f"Starting Training: {NUM_EPISODES} episodes")
    print(f"Devices: {NUM_DEVICES}, Layers: {NUM_LAYERS}")
    
    for e in range(NUM_EPISODES):
        state = env.reset()
        done = False
        score = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            agent.replay()
            
        scores.append(score)
        
        # Update Target Network periodically
        if e % 10 == 0:
            agent.update_target_network()
            print(f"Episode {e}/{NUM_EPISODES}, Score: {score:.2f}, Epsilon: {agent.epsilon:.2f}")

    print("Training Completed.")
    
    # Save the trained model
    agent.save("rl_pdnn/model.pth")
    print("Model saved to rl_pdnn/model.pth")

if __name__ == "__main__":
    main()
