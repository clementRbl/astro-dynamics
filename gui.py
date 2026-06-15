"""GUI Streamlit — Eagle-1 : visualise une partie jouée par l'agent.

La GUI ne contient aucune logique RL : pour chaque pas, elle envoie l'état à
l'API et récupère l'action à jouer. L'API doit donc tourner en parallèle :
    uv run uvicorn api:app
Lancement de la GUI :
    uv run streamlit run gui.py
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import requests
import streamlit as st

API_URL = "http://localhost:8000/play"
ACTION_LABELS = ["Rien", "Gauche", "Principal", "Droit"]


def get_action(observation) -> int:
    response = requests.post(API_URL, json={"observation": observation.tolist()})
    response.raise_for_status()
    return response.json()["action"]


def run_episode(seed: int) -> tuple[list, list[int], float]:
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames, actions = [], []
    total_reward = 0.0
    done = False
    while not done:
        frames.append(env.render())
        action = get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        actions.append(action)
        total_reward += float(reward)
        done = terminated or truncated
    env.close()
    return frames, actions, total_reward


st.set_page_config(page_title="Eagle-1 GUI", layout="wide")
st.title("Eagle-1 — Visualisation d'un atterrissage")
st.caption("Les actions sont fournies par l'API (`/play`).")

col1, col2 = st.columns([2, 1])
with col1:
    seed = st.number_input(
        "Graine aléatoire", min_value=0, max_value=9999, value=42, step=1
    )
with col2:
    st.write("")
    st.write("")
    run = st.button("Lancer la partie", type="primary")

if run:
    try:
        with st.spinner("Simulation en cours…"):
            frames, actions, total_reward = run_episode(seed=int(seed))
    except requests.exceptions.ConnectionError:
        st.error(
            "Impossible de joindre l'API. Lancez-la d'abord : `uv run uvicorn api:app`"
        )
        st.stop()

    color = "green" if total_reward >= 200 else "orange" if total_reward >= 0 else "red"
    st.markdown(
        f"**Récompense totale :** :{color}[{total_reward:.2f}]  "
        f"| **Étapes :** {len(actions)}"
    )

    st.subheader("Lecture image par image")
    step_idx = st.slider("Étape", 0, len(frames) - 1, 0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(frames[step_idx])
    action_shown = actions[step_idx] if step_idx < len(actions) else 0
    ax.set_title(
        f"Étape {step_idx + 1}/{len(frames)} — Action : {ACTION_LABELS[action_shown]}"
    )
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)
