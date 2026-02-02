import gym
from gym import spaces
import numpy as np
from typing import List
from .utils import generate_dummy_dnn_model, DNNLayer
from integrated_system.resource_manager import ResourceManager

class MultiAgentIoTEnv(gym.Env):
    """
    Multi-Agent Environment where K agents (tasks) compete for resources.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agents=3, num_devices=5, model_types=None):
        super(MultiAgentIoTEnv, self).__init__()
        
        self.num_agents = num_agents
        self.num_devices = num_devices
        # model_types: List of strings e.g. ["lenet", "resnet18", "mobilenet"]
        self.model_types = model_types if model_types else ["lenet"] * num_agents
        
        self.resource_manager = ResourceManager(num_devices)
        
        # Action Space: K agents, each chooses a device (0..D-1)
        self.action_space = spaces.Discrete(num_devices) # Per agent
        
        # State Space Per Agent: Same as Single Agent
        # [LayerIdx, DevP_1, ... DevP_D]
        self.single_state_dim = 1 + (4 * num_devices)
        self.observation_space = spaces.Box(low=0, high=1000, 
                                            shape=(self.single_state_dim,), 
                                            dtype=np.float32)
        
        self.agents_tasks: List[List[DNNLayer]] = []
        self.agents_progress: List[int] = [] # Current layer index for each agent
        self.agents_prev_device: List[int] = [] 
        self.agents_done: List[bool] = []
        
        self.reset()
        
    def reset(self):
        """Resets the environment and the shared resource manager."""
        self.resource_manager.reset(self.num_devices)
        
        self.agents_tasks = []
        self.agents_progress = []
        self.agents_prev_device = []
        self.agents_done = []
        
        from .utils import generate_specific_model # Imported here to avoid circular
        
        for i in range(self.num_agents):
            m_type = self.model_types[i] if i < len(self.model_types) else "lenet"
            self.agents_tasks.append(generate_specific_model(m_type))
            self.agents_progress.append(0)
            self.agents_prev_device.append(-1)
            self.agents_done.append(False)
            
        return self._get_all_observations()

    def _get_all_observations(self):
        obs_list = []
        for i in range(self.num_agents):
            if self.agents_done[i]:
                # Return zero state if done
                obs_list.append(np.zeros(self.single_state_dim))
                continue
                
            # Construct state for Agent i
            # 1. Progress (Normalized by THAT agent's total layers)
            total_layers = len(self.agents_tasks[i])
            current_obs = [float(self.agents_progress[i]) / total_layers]
            
            # 2. Shared Device States (from Resource Manager)
            for d_id in range(self.num_devices):
                current_obs.extend(self.resource_manager.get_state_for_device(d_id))
            
            obs_list.append(np.array(current_obs, dtype=np.float32))
        return obs_list

    def step(self, actions: List[int]):
        """
        Executes one step for ALL agents.
        Args:
            actions: List of ints, one for each agent.
        """
        rewards = [0.0] * self.num_agents
        dones = [False] * self.num_agents
        infos = [{} for _ in range(self.num_agents)]
        
        # We assume agents act 'simultaneously' in this discrete step.
        # Conflict resolution: First come first serve based on index (simplified)
        # Or random shuffle to be fair.
        agent_indices = list(range(self.num_agents))
        # np.random.shuffle(agent_indices) # Shuffle for fairness if training
        
        for idx in agent_indices:
            if self.agents_done[idx]:
                dones[idx] = True
                continue
                
            action = actions[idx]
            selected_device_id = int(action)
            current_layer = self.agents_tasks[idx][self.agents_progress[idx]]
            
            # 1. Try Allocate
            success = self.resource_manager.try_allocate(selected_device_id, current_layer)
            
            if not success:
                # Penalty for failure (Constraint violation or Resource Full)
                rewards[idx] = -50.0 # Heavy penalty
                # We do NOT advance the layer. The agent retries next step? 
                # Or we fail the episode?
                # Option A: Fail episode for this agent
                # dones[idx] = True 
                # self.agents_done[idx] = True
                
                # Option B: Retry penalty (Latency increases)
                # Let's just penalize and move on to next layer to keep logic simple for now
                # In real scenario, must retry or drop. We'll simulate 'drop' or 'cloud fallback'
                rewards[idx] -= 50 # Add cloud latency
                self._advance_agent(idx, selected_device_id)

            else:
                # 2. Calculate Latency (Cost)
                dev = self.resource_manager.devices[selected_device_id]
                
                # Compute
                comp_latency = current_layer.computation_demand / dev.cpu_speed
                
                # Transmit
                trans_latency = 0
                prev_dev = self.agents_prev_device[idx]
                
                if prev_dev != -1 and prev_dev != selected_device_id:
                     # Get output size of PREVIOUS layer
                     if self.agents_progress[idx] > 0:
                         prev_data = self.agents_tasks[idx][self.agents_progress[idx]-1].output_data_size
                     else:
                         prev_data = 5.0 # Input
                     trans_latency = prev_data / dev.bandwidth
                elif prev_dev == -1:
                    trans_latency = 5.0 / dev.bandwidth
                    
                total_latency = comp_latency + trans_latency
                rewards[idx] = -total_latency
                
                self._advance_agent(idx, selected_device_id)

        next_obs = self._get_all_observations()
        
        # Global Done?
        all_done = all(self.agents_done)
        
        return next_obs, rewards, self.agents_done, infos # Return individual dones list

    def _advance_agent(self, idx, device_id):
        self.agents_prev_device[idx] = device_id
        self.agents_progress[idx] += 1
        if self.agents_progress[idx] >= len(self.agents_tasks[idx]):
            self.agents_done[idx] = True
