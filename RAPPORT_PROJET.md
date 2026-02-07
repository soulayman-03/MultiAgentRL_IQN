# Rapport de Projet : Multi-Agent RL-PDNN pour l'IoT

## Résumé
Ce projet porte sur le développement d'un système de **Multi-Agent Reinforcement Learning (MARL)** optimisé pour l'inférence distribuée de réseaux de neurones profonds (DNN) sur des appareils IoT à ressources limitées. L'objectif est de coordonner plusieurs agents effectuant des tâches de reconnaissance d'images (MNIST) en parallèle, tout en respectant des contraintes de mémoire, de bande passante et de confidentialité. Un **Gestionnaire de Ressources** centralisé assure l'arbitrage pour éviter les collisions et optimiser la latence globale du système.

## Abstract
This project implements a **Multi-Agent Reinforcement Learning (MARL)** framework designed for distributed Deep Neural Network (DNN) inference in resource-constrained IoT environments. The system coordinates multiple agents performing concurrent image recognition tasks (MNIST) while adhering to strict memory, bandwidth, and privacy constraints. A centralized **Resource Manager** acts as a referee to prevent hardware overflows and optimize global system latency, ensuring efficient load balancing across edge and cloud devices.

---

## 1. Introduction
Avec l'explosion de l'Internet des Objets (IoT), le besoin d'intelligence en périphérie (Edge Intelligence) est devenu crucial. Cependant, les modèles de Deep Learning modernes sont souvent trop lourds pour être exécutés sur un seul petit appareil. L'inférence distribuée, ou "Split Computing", permet de diviser un modèle en plusieurs morceaux répartis sur différents appareils. 

Le défi majeur réside dans la gestion dynamique de ces ressources lorsque plusieurs tâches s'exécutent simultanément. Ce projet répond à ce besoin en utilisant l'apprentissage par renforcement multi-agents pour apprendre des politiques d'allocation intelligentes et adaptatives.

---

## 2. Cadre Général du Projet
Le système est composé de trois piliers principaux :

### A. Environnement Multi-Agents (MARL)
Contrairement aux approches classiques avec un seul agent, notre système simule un environnement complexe où chaque requête d'inférence est gérée par un agent autonome. Ces agents apprennent à :
- Choisir les meilleurs appareils pour chaque couche du modèle.
- Éviter les appareils saturés.
- Minimiser le temps de réponse total.

### B. Gestionnaire de Ressources (Resource Manager)
Le "Resource Manager" est le garant de l'intégrité physique du réseau. Il maintient un état global de la mémoire et de la bande passante disponible sur chaque nœud (Edge, Fog, Cloud). Il valide ou refuse les allocations demandées par les agents, renvoyant des pénalités en cas de dépassement de capacité.

### C. Modèles Hétérogènes et Split Inference
Le projet supporte différents types d'architectures CNN :
- **SimpleCNN** (LeNet-like) pour la rapidité.
- **DeepCNN** pour la précision.
- **MiniResNet** pour tester des connexions résiduelles complexes.
Chaque modèle est découpé dynamiquement en fonction des décisions des agents.

---

## 3. État de l'art : Reconnaissance des Chiffres Manuscrits (MNIST) et RL-PDNN
La reconnaissance de chiffres manuscrits, basée sur le dataset MNIST, est un standard pour valider les architectures de vision. Les réseaux de neurones convolutionnels (CNN) comme LeNet-5 ont historiquement dominé ce domaine avec des précisions dépassant 99%.

### L'apport de RL-PDNN
L'article **"RL-PDNN: Reinforcement Learning for Privacy-Aware Distributed Neural Networks"** introduit une dimension critique : la **confidentialité (Privacy)**. 
- **Privacy-Awareness** : Au lieu de simplement optimiser la vitesse, RL-PDNN veille à ce qu'aucune entité unique dans le réseau distribué ne possède une vue complète du modèle ou des données d'entrée, prévenant ainsi les attaques de reconstruction de données.
- **Optimisation par RL** : L'utilisation de Deep Q-Learning permet de s'adapter aux changements de bande passante et à la disponibilité fluctuante des serveurs Edge de manière bien plus efficace que les heuristiques statiques.
- **Application au MNIST** : Dans notre projet, nous utilisons ces principes pour garantir que même si un nœud malveillant intercepte un fragment de calcul du chiffre "5", il ne peut pas en déduire l'image originale grâce au partitionnement stratégique.

---

## 4. Conclusion
Le passage à une architecture Multi-Agents représente une avancée vers des systèmes IoT industriels réalistes. En combinant la puissance de la distribution des calculs, l'intelligence du MARL et les garanties de confidentialité de l'approche PDNN, nous avons créé une plateforme robuste capable d'optimiser l'inférence collaborative à grande échelle.
