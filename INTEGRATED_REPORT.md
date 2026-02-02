# Integrated Multi-Agent RL-PDNN System Report

## 1. System Overview
This project has evolved into a **Multi-Agent Reinforcement Learning (MARL)** system capable of orchestrating distributed inference for multiple concurrent Deep Learning tasks on resource-constrained IoT devices.

**New Architecture Components:**
*   **Multi-Agent Environment**: A shared gym environment where multiple agents (LeNet Agent, ResNet Agent, etc.) interact simultaneously.
*   **Resource Manager**: A centralized module ensuring fair resource allocation (Memory, Bandwidth) and preventing collisions.
*   **Heterogeneous Workloads**: Support for different types of Neural Networks running in parallel.

## 2. Dynamic Performance Objectives
The system is designed to optimize global objectives rather than just individual ones:

*   **Global Throughput**: Maximizing the number of successful inferences per second across the entire network.
*   **Fairness**: Ensuring that smaller tasks (like LeNet) aren't starved by larger tasks (like ResNet).
*   **Conflict Resolution**: The training process teaches agents to "yield" or choose alternative paths when preferred devices are congested.

## 3. Multi-Agent Demo Capabilities
The `multi_agent_demo.py` script demonstrates the complex interaction between agents.

**Scenario:**
1.  **Agent A (LeNet)** requests inference.
2.  **Agent B (ResNet)** requests inference simultaneously.
3.  **Resource Manager** updates device states in real-time.

**Simulation Output Example:**
```text
[Step 1]
- Agent A allocates Layer 1 to Device 0 (Edge).
- Agent B allocates Layer 1 to Device 4 (Cloud) -> avoiding conflict on Edge.
[Resource Manager] Device 0 Memory: 80% Used.
```

## 4. Conclusion
The transition to a Multi-Agent System represents a significant leap towards realistic **Edge Intelligence**. The system effectively demonstrates **Decentralized Decision Making** under **Centralized Constraint Management**.

Generated automatically.
