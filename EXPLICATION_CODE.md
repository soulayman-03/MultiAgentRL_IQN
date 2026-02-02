# Explication Ultra-Détaillée du Code (Multi-Agent Version)

Ce document décortique les nouveaux fichiers clés introduits pour supporter le Multi-Agent RL.

---

## 1. Le Chef d'Orchestre : `integrated_system/resource_manager.py`

C'est la pièce maîtresse qui permet aux agents de ne pas se marcher dessus.

### `class ResourceManager`
*   **Rôle** : Garder une trace **unique et partagée** de l'état de chaque appareil. C'est un singleton (il n'y en a qu'un seul pour tout le monde).
*   **`allocate(device_id, memory_demand)`** :
    *   Quand un agent veut utiliser un appareil, il appelle cette fonction.
    *   Le Manager vérifie `if device.current_memory + demand <= device.capacity`.
    *   Si oui, il réserve la place et renvoie `True`. Sinon, `False` (et l'agent est puni).
*   **`release(device_id, memory_demand)`** :
    *   Quand une tâche est finie sur un appareil, on libère la mémoire pour les autres.

---

## 2. Le Monde : `rl_pdnn/multi_agent_env.py`

C'est le nouveau gymnase. Il remplace `env.py` pour le cas multi-agents.

### `class MultiAgentIoTEnv`
*   Au lieu d'avoir un seul état, il gère **une liste d'agents**.
*   **`step(actions)`** :
    *   Cette fonction prend une **liste d'actions** (ex: `[Agent A -> Device 1, Agent B -> Device 2]`).
    *   Elle applique les actions **dans l'ordre** (ou simultanément).
    *   Elle interroge le `ResourceManager` pour valider chaque action.
    *   Elle retourne une liste de récompenses : `[Reward_Agent_A, Reward_Agent_B]`.

---

## 3. Le Cerveau : `rl_pdnn/models` et `marl_trainer.py`

### `marl_trainer.py` (L'Entraîneur)
C'est le script qui remplace `main.py` pour l'apprentissage.
1.  Il crée **N agents** (par exemple 3 agents DQN).
2.  Il lance une boucle d'épisodes.
3.  À chaque pas de temps, il demande à **tous les agents** de choisir une action.
4.  Il collecte les expériences de tout le monde et entraîne les réseaux de neurones.

---

## 4. Les Modèles Hétérogènes : `split_inference/cnn_model.py`

Nous utilisons maintenant trois types de modèles CNN pour représenter différents niveaux de complexité et besoins en ressources :

1.  **SimpleCNN** : Un petit modèle rapide inspiré de LeNet, idéal pour les appareils à faibles ressources.
2.  **DeepCNN** : Un modèle plus profond avec plus de couches de convolution pour une meilleure généralisation, demandant plus de calculs.
3.  **MiniResNet** : Une architecture moderne incluant des connexions résiduelles (*skip connections*) pour une performance accrue au prix d'une complexité supérieure.

Cela permet de tester si notre système est capable de gérer **des tâches et des architectures différentes** en même temps sur le même réseau IoT.

---

## 5. Le Moteur d'Inférence : `integrated_system/inference_engine.py` (Mis à jour)

Il a été modifié pour être **thread-safe** ou supporter l'exécution séquentielle de plusieurs modèles.
L'idée est de pouvoir dire :
> "Exécute le modèle A avec le plan X, ET exécute le modèle B avec le plan Y."

Le moteur calcule les latences cumulées : si deux agents utilisent le même lien Wifi en même temps, la simulation (dans une version avancée) prend en compte le ralentissement dû au partage de la bande passante.
