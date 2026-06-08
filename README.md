# Astro Dynamics — Atterrisseur lunaire Eagle

Projet d'apprentissage par renforcement réalisé dans le cadre du parcours OpenClassrooms.  
L'objectif final est d'entraîner un agent capable d'atterrir sur la Lune avec l'environnement **LunarLander-v3** de Gymnasium.

## Structure du projet

```
astro-dynamics/
├── exercice_1_blocs_construction.ipynb   # Découvrir le cycle observation → action → récompense
├── exercice_2_qtable.ipynb               # Entraîner un agent avec une Q-table
├── exercice_3_dqn.ipynb                  # Remplacer la Q-table par un réseau de neurones (DQN)
├── mission_atterrisseur.ipynb            # Mission finale : piloter l'atterrisseur Eagle
└── tests/                                # Tests unitaires
```

## Installation

Ce projet utilise [uv](https://docs.astral.sh/uv/) pour gérer l'environnement et les dépendances.

```bash
# Installer uv si ce n'est pas déjà fait
curl -LsSf https://astral.sh/uv/install.sh | sh

# Installer les dépendances
uv sync
```

## Utilisation

```bash
# Vérifier le style du code
uv run ruff check .

# Formater le code
uv run ruff format .

# Lancer les tests
uv run pytest
```

## Dépendances principales

| Librairie | Rôle |
|---|---|
| `gymnasium` | Environnements de simulation (CartPole, LunarLander...) |
| `stable-baselines3` | Algorithmes RL pré-implémentés (DQN, PPO...) |
| `matplotlib` | Visualisation des résultats |

## Progression

- [x] Exercice 1 — Blocs de construction de l'apprentissage par renforcement
- [ ] Exercice 2 — Agent avec Q-table
- [ ] Exercice 3 — Agent avec DQN
- [ ] Mission — Atterrisseur lunaire Eagle
