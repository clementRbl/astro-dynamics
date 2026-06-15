"""API FastAPI — Eagle-1 : reçoit un état, renvoie l'action du modèle DQN.

Toute la logique RL (chargement du modèle, prédiction) est ici, côté backend.
Lancement : uv run uvicorn api:app --reload
"""

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from stable_baselines3 import DQN

MODEL_PATH = Path("models/dqn_lunarlander_best.zip")

app = FastAPI(title="Eagle-1 API", version="1.0")

_model: DQN | None = None


def get_model() -> DQN:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Modèle introuvable : {MODEL_PATH}. Lancez d'abord le notebook."
                ),
            )
        _model = DQN.load(MODEL_PATH)
    return _model


class State(BaseModel):
    observation: list[float]


class Action(BaseModel):
    action: int


@app.post("/play", response_model=Action)
def play(state: State) -> Action:
    if len(state.observation) != 8:
        raise HTTPException(
            status_code=422,
            detail=(
                f"L'observation doit contenir 8 valeurs, reçu {len(state.observation)}."
            ),
        )
    model = get_model()
    obs = np.array(state.observation, dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    return Action(action=int(action))
