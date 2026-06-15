"""Enregistre une vidéo (.mp4) d'un atterrissage réussi du pilote Eagle-1 (DQN).

Conforme à la consigne : une vidéo de 20 à 30 secondes montrant une performance
réussie (récompense >= 200) du pilote automatique, en mode déterministe.

Le script rejoue des parties jusqu'à en trouver une réussie, puis l'exporte en
MP4 à un fps calé pour obtenir une durée dans la cible 20-30 s.

Lancement : uv run python record_video.py
"""

import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3 import DQN

MODEL_PATH = "models/dqn_lunarlander_best.zip"
VIDEO_PATH = "logs/videos/eagle1-atterrissage-reussi.mp4"
SUCCESS_THRESHOLD = 200.0
TARGET_DURATION_S = 25.0  # durée visée, dans la fenêtre 20-30 s


def play_episode(model: DQN, seed: int) -> tuple[list, float, bool]:
    """Joue une partie déterministe. Renvoie (frames, récompense, atterrissage propre).

    Un atterrissage est jugé « propre » s'il rapporte >= 200 points, se termine
    naturellement, avec les deux pattes au sol et l'engin bien centré : c'est ce
    qui garantit qu'il ne dérive pas hors de la zone des drapeaux.
    """
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames, total_reward, terminated, truncated = [], 0.0, False, False
    while not (terminated or truncated):
        frames.append(env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += float(reward)
    env.close()
    x, leg_left, leg_right = obs[0], obs[6], obs[7]
    clean = (
        total_reward >= SUCCESS_THRESHOLD
        and terminated
        and leg_left == 1
        and leg_right == 1
        and abs(x) < 0.1
    )
    return frames, total_reward, clean


model = DQN.load(MODEL_PATH)

# Chercher des atterrissages propres, puis garder le plus long : davantage de
# frames = un fps plus élevé pour la durée visée, donc une vidéo plus fluide.
best_frames, best_reward, best_seed = [], 0.0, -1
for seed in range(50):
    frames, reward, clean = play_episode(model, seed)
    if clean and len(frames) > len(best_frames):
        best_frames, best_reward, best_seed = frames, reward, seed
frames, reward = best_frames, best_reward
print(f"Partie retenue — graine {best_seed} : {reward:.2f} pts, {len(frames)} étapes")

# Caler le fps pour viser ~25 s (donc rester dans la fenêtre 20-30 s).
fps = max(1, round(len(frames) / TARGET_DURATION_S))
duration = len(frames) / fps
imageio.mimsave(VIDEO_PATH, frames, fps=fps, macro_block_size=None)
print(f"Vidéo enregistrée : {VIDEO_PATH} ({duration:.1f} s à {fps} fps)")
