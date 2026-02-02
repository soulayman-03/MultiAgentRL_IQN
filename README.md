# Privacy-Aware Multi-Agent Distributed Deep Neural Networks (RL-PDNN)

> **New:** Now featuring **Multi-Agent Reinforcement Learning (MARL)** for concurrent task scheduling!

This repository hosts a complete framework for distributing **Deep Learning** inference tasks across a network of resource-constrained **IoT devices**. It uses a team of **Deep Reinforcement Learning (DRL)** agents to optimize resource allocation, prevent bottlenecks, and enforce data privacy.

---

## 📚 Table of Contents
1.  [Project Structure](#-project-structure)
2.  [The Multi-Agent Revolution](#-the-multi-agent-revolution)
3.  [Key Components](#-key-components)
4.  [How to Run](#-how-to-run)

---

## 📂 Project Structure

```text
RL/
├── multi_agent_demo.py         # 🚀 START HERE: The main Multi-Agent Demo
├── .gitignore                  # Git configuration
│
├── rl_pdnn/                    # THE BRAIN (Reinforcement Learning)
│   ├── marl_trainer.py         # 🆕 Multi-Agent Training Script
│   ├── multi_agent_env.py      # 🆕 Multi-Agent Simulation Environment
│   ├── agent.py                # Deep Q-Network (DQN) Agent
│   ├── utils.py                # Device & Layer definitions
│   └── models/                 # Saved Agent Models
│
├── integrated_system/          # THE SYSTEM (Shared Resources)
│   ├── resource_manager.py     # 🆕 Shared Resource State Manager (The "Referee")
│   └── inference_engine.py     # Execution Engine (Runner)
│
├── split_inference/            # THE WORKLOAD (Deep Learning)
│   ├── cnn_model.py            # Neural Network Architectures (LeNet, etc.)
│   └── train_cnn.py            # Training script for the vision models
│
└── README.md                   # This file
```

---

## 🤖 The Multi-Agent Revolution

In previous versions, a single agent managed one task. But real IoT systems are busy!
**RL-PDNN v2.0** introduces:

1.  **Concurrent Execution**: Multiple inference requests happen at once.
2.  **Resource Competition**: Agents must effectively share limited device memory and bandwidth.
3.  **Global Resource Manager**: A central system component that ensures physical constraints are respecting (preventing memory overflows).

### How it works
*   **Agent A** wants to run SimpleCNN for a Security Camera.
*   **Agent B** wants to run DeepCNN for a Smart Speaker.
*   **Agent C** wants to run MiniResNet for an industrial sensor.
*   They communicate with the **Resource Manager** to reserve compute slots on the edge devices.
*   If Agent A hogs the powerful server, Agent B learns to offload to other available nodes.

---

## 🔑 Key Components

### 1. `integrated_system/resource_manager.py`
The singleton class that tracks the global state of the network. It prevents two agents from crashing a device by overfilling its RAM.

### 2. `rl_pdnn/multi_agent_env.py`
The gym-like environment that steps multiple agents simultaneously (`step(actions_list)`).

### 3. `rl_pdnn/marl_trainer.py`
The training loop that improves all agents in parallel, teaching them to handle diverse workloads.

---

## 🚀 How to Run

1.  **Install Dependencies**:
    ```bash
    pip install gym torch numpy matplotlib torchvision
    ```

2.  **Train the Multi-Agent System**:
    ```bash
    python -m rl_pdnn.marl_trainer
    ```

3.  **Train the Vision Model** (for the realistic demo):
    ```bash
    python -m split_inference.train_cnn
    ```

4.  **Run the Full Demo**:
    ```bash
    python multi_agent_demo.py
    ```
