# Comprendre notre Algorithme : Multi-Agent RL-PDNN

Voici une explication claire de l'évolution de notre système vers une architecture **Multi-Agents**. Nous sommes passés d'un seul agent gérant une tâche à une "équipe" d'agents gérant plusieurs tâches simultanées.

---

## 1. Le Nouveau Problème : La Concurrence (Embouteillages)
Dans la version précédente, un seul modèle d'IA s'exécutait. C'est facile.
Mais dans la réalité (ville intelligente, usine 4.0), **plusieurs caméras** et capteurs demandent des calculs en même temps.
*   **Conflit** : Si la Caméra A utilise toute la mémoire de l'Appareil 1, la Caméra B ne peut plus l'utiliser.
*   **Risque** : Embouteillages, plantages, latence explosée.

## 2. La Solution : Système Multi-Agents (MARL)
Nous avons créé un environnement où **plusieurs Agents** (un par demande d'inférence) évoluent ensemble.

*   **Agent 1** (s'occupe de la tâche SimpleCNN)
*   **Agent 2** (s'occupe de la tâche DeepCNN)
*   **Agent 3** (s'occupe de la tâche MiniResNet)

### Comment ils cohabitent ? (Le "Resource Manager")
Nous avons introduit un **Resource Manager (Gestionnaire de Ressources)**. C'est le "Policier" du système.
1.  **Vision Globale** : Il connaît l'état de chaque appareil (Mémoire restante, Bande passante utilisée) en temps réel.
2.  **Arbitrage** : Quand l'Agent 1 prend 100 Mo sur l'Appareil A, le Resource Manager met à jour la capacité disponible. Si l'Agent 2 essaie ensuite de prendre 500 Mo sur ce même appareil alors qu'il ne reste que 200 Mo, l'action est **interdite** (ou pénalisée).

---

## 3. Le Cerveau : Chaque Agent apprend sa stratégie
Chaque agent est un réseau de neurones (DQN) indépendant (ou partageant un "cerveau" commun mais agissant individuellement).

### A. L'Observation (Ce qu'ils voient)
Chaque agent regarde :
1.  **Son état personnel** : "Je suis à la couche 3 de mon modèle LeNet".
2.  **L'état du système** : "L'appareil 2 est libre, l'appareil 4 est saturé".

### B. L'Action (La décision)
Chaque agent décide indépendamment :
> *"Je place ma prochaine couche sur l'Appareil 2."*

### C. La Récompense (La Note)
C'est ici que ça devient collaboratif/compétitif :
*   Si un agent choisit un appareil saturé : **Grosse punition (-100)**.
*   Si un agent réussit à terminer sa tâche rapidement sans gêner les autres : **Récompense positive**.

---

## 4. L'Entraînement Multi-Agents
Au lieu d'entraîner un seul robot, on en entraîne 3 (ou plus) en parallèle dans le même gymnase.
*   Ils apprennent à **éviter les conflits**.
*   Ils apprennent à **se répartir la charge** (Load Balancing naturel).
*   L'un peut apprendre à utiliser les appareils puissants, tandis que l'autre se contente des petits appareils pour ne pas gêner.

## 5. Résumé de la Transition

| Concept | Avant (Single Agent) | Maintenant (Multi-Agent) |
| :--- | :--- | :--- |
| **Tâche** | Une seule inférence à la fois | Plusieurs inférences en parallèle |
| **Environnement** | Statique (sauf action de l'agent) | Dynamique (les autres agents modifient l'état) |
| **Contraintes** | "Est-ce que l'appareil a de la place ?" | "Est-ce qu'il aura *encore* de la place après le passage de l'Agent 2 ?" |
| **Objectif** | Optimiser MA latence | Optimiser la latence GLOBALE de tous les utilisateurs |

C'est un système beaucoup plus robuste et réaliste pour l'IoT Distribué.
