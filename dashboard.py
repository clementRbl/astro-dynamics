"""Dashboard Streamlit — Eagle-1 : suivi interactif des performances.

Affiche les courbes d'apprentissage, les métriques sur les parties jouées
(moyenne, écart-type) et les décisions de l'IA selon la circonstance.
Comme la GUI, le dashboard ne fait aucune prédiction : il interroge l'API.
    uv run uvicorn api:app          (backend)
    uv run streamlit run dashboard.py
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = "http://localhost:8000/play"
LOGS_DIR = Path("logs")
ACTION_LABELS = {0: "Rien", 1: "Gauche", 2: "Principal", 3: "Droit"}

# Découpage des observations en circonstances lisibles.
CIRCUMSTANCES = {
    "Position horizontale": (0, ["Gauche", "Centre", "Droite"], [-0.3, 0.3]),
    "Vitesse verticale": (3, ["Chute rapide", "Chute lente", "Montée"], [-0.5, -0.1]),
    "Inclinaison": (4, ["Penché gauche", "Droit", "Penché droit"], [-0.2, 0.2]),
}


def get_action(observation) -> int:
    response = requests.post(API_URL, json={"observation": observation.tolist()})
    response.raise_for_status()
    return response.json()["action"]


@st.cache_data(show_spinner=False)
def play_games(n_episodes: int) -> tuple[list[float], pd.DataFrame]:
    """Joue n parties via l'API et collecte récompenses + (état, action)."""
    env = gym.make("LunarLander-v3")
    rewards, records = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        total = 0.0
        done = False
        while not done:
            action = get_action(obs)
            row = {f"obs{i}": v for i, v in enumerate(obs)}
            row["action"] = action
            records.append(row)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
        rewards.append(total)
    env.close()
    return rewards, pd.DataFrame(records)


@st.cache_data
def load_training_curves() -> dict[str, pd.DataFrame]:
    curves = {}
    if not LOGS_DIR.exists():
        return curves
    for exp_dir in sorted(LOGS_DIR.iterdir()):
        npz = exp_dir / "evaluations.npz"
        if npz.exists():
            data = np.load(npz)
            curves[exp_dir.name] = pd.DataFrame(
                {
                    "timesteps": data["timesteps"],
                    "mean_reward": data["results"].mean(axis=1),
                }
            )
    return curves


st.set_page_config(page_title="Eagle-1 Dashboard", layout="wide")
st.title("Eagle-1 — Dashboard des performances")

# --- Filtres dynamiques ---
st.sidebar.header("Filtres")
n_episodes = st.sidebar.slider("Nombre de parties à jouer", 1, 20, 5)
circumstance = st.sidebar.selectbox("Circonstance à analyser", list(CIRCUMSTANCES))

# --- Courbes d'apprentissage (depuis les logs d'entraînement) ---
st.subheader("Courbes d'apprentissage")
curves = load_training_curves()
if curves:
    fig = go.Figure()
    for name, df in curves.items():
        fig.add_trace(go.Scatter(x=df["timesteps"], y=df["mean_reward"], name=name))
    fig.add_hline(
        y=200, line_dash="dash", line_color="green", annotation_text="Objectif 200 pts"
    )
    fig.update_layout(
        xaxis_title="Timesteps",
        yaxis_title="Récompense moyenne",
        height=400,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Aucun log d'entraînement trouvé dans `logs/`.")

# --- Parties jouées via l'API ---
st.subheader(f"Performances sur {n_episodes} parties jouées")
try:
    rewards, df_steps = play_games(n_episodes)
except requests.exceptions.ConnectionError:
    st.error(
        "Impossible de joindre l'API. Lancez-la d'abord : `uv run uvicorn api:app`"
    )
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Récompense moyenne", f"{np.mean(rewards):.2f}")
c2.metric("Écart-type", f"± {np.std(rewards):.2f}")
c3.metric("Taux de réussite (≥200)", f"{np.mean(np.array(rewards) >= 200):.0%}")

df_rewards = pd.DataFrame({"Partie": range(1, len(rewards) + 1), "Récompense": rewards})
fig_r = px.bar(df_rewards, x="Partie", y="Récompense", title="Récompense par partie")
fig_r.add_hline(y=200, line_dash="dash", line_color="green")
fig_r.update_layout(height=350)
st.plotly_chart(fig_r, use_container_width=True)

# --- Décisions de l'IA selon la circonstance ---
st.subheader(f"Décisions de l'IA selon : {circumstance}")
obs_idx, bin_labels, thresholds = CIRCUMSTANCES[circumstance]
edges = [-np.inf, *thresholds, np.inf]
df_steps["Situation"] = pd.cut(df_steps[f"obs{obs_idx}"], bins=edges, labels=bin_labels)
df_steps["Action"] = df_steps["action"].map(ACTION_LABELS)

dist = (
    df_steps.groupby(["Situation", "Action"], observed=True)
    .size()
    .reset_index(name="count")
)
total_per_sit = dist.groupby("Situation", observed=True)["count"].transform("sum")
dist["Proportion"] = dist["count"] / total_per_sit

fig_d = px.bar(
    dist,
    x="Situation",
    y="Proportion",
    color="Action",
    barmode="group",
    title="Répartition des actions selon la situation",
    color_discrete_map={
        "Rien": "#636EFA",
        "Gauche": "#EF553B",
        "Principal": "#00CC96",
        "Droit": "#AB63FA",
    },
)
fig_d.update_layout(height=400, yaxis_tickformat=".0%")
st.plotly_chart(fig_d, use_container_width=True)
st.caption(
    "Lecture : pour chaque situation, proportion de chaque action choisie par l'agent."
)
