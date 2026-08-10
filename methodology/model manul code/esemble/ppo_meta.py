"""PPO meta-learner: blend XGBoost + LSTM forecasts (standalone copy).

Cfg dict is passed from stacked_ensemble.py — edit total_timesteps there only.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False


def sb3_available() -> bool:
    return _SB3_AVAILABLE


if _SB3_AVAILABLE:

    class ForecastBlendEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, rows: list[dict[str, float]]):
            super().__init__()
            self._rows = rows
            self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32)
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        def _obs(self, row: dict[str, float]) -> np.ndarray:
            xgb = float(row["xgb_pred"])
            lstm = float(row.get("lstm_pred", xgb))
            if not np.isfinite(lstm):
                lstm = xgb
            return np.array([xgb, lstm, xgb - lstm, abs(xgb - lstm)], dtype=np.float32)

        def _combine(self, row: dict[str, float], w: float) -> float:
            xgb = float(row["xgb_pred"])
            lstm = row.get("lstm_pred")
            if lstm is None or not np.isfinite(lstm):
                return xgb
            w = float(np.clip(w, 0.0, 1.0))
            return w * xgb + (1.0 - w) * float(lstm)

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            rng = np.random.default_rng(seed)
            self._order = rng.permutation(len(self._rows)).tolist()
            self._pos = 0
            row = self._rows[self._order[self._pos]]
            return self._obs(row), {}

        def step(self, action):
            row = self._rows[self._order[self._pos]]
            pred = self._combine(row, float(action[0]))
            err = float(row["y"]) - pred
            reward = float(-err * err)
            terminated = self._pos >= len(self._order) - 1
            self._pos += 1
            if terminated:
                obs = self._obs(row)
            else:
                obs = self._obs(self._rows[self._order[self._pos]])
            return obs, reward, terminated, False, {"y_pred": pred, "y_true": float(row["y"])}


def _rows_from_frame(frame: pd.DataFrame, use_lstm: bool) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for rec in frame.to_dict("records"):
        rows.append({
            "y": float(rec["y"]),
            "xgb_pred": float(rec["xgb_pred"]),
            "lstm_pred": float(rec["lstm_pred"]) if use_lstm and "lstm_pred" in rec else np.nan,
        })
    return rows


def _combine_row(row: dict[str, float], w: float) -> float:
    xgb = float(row["xgb_pred"])
    lstm = row.get("lstm_pred")
    if lstm is None or not np.isfinite(lstm):
        return xgb
    w = float(np.clip(w, 0.0, 1.0))
    return w * xgb + (1.0 - w) * float(lstm)


def train_ppo_meta(meta_val: pd.DataFrame, use_lstm: bool, cfg: dict[str, Any]) -> Any | None:
    if not _SB3_AVAILABLE:
        return None
    ppo_cfg = cfg.get("ppo_meta", {})
    rows = _rows_from_frame(meta_val, use_lstm)
    if len(rows) < 5:
        return None

    total_ts = int(ppo_cfg.get("total_timesteps", 2000))
    print(f"  [ppo meta] training total_timesteps={total_ts} on {len(rows)} validation rows")

    env = DummyVecEnv([lambda: ForecastBlendEnv(rows)])
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        n_steps=int(ppo_cfg.get("n_steps", 64)),
        batch_size=int(ppo_cfg.get("batch_size", 32)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        verbose=0,
        seed=42,
        device="cpu",
    )
    model.learn(total_timesteps=total_ts)
    return model


def predict_ppo_meta(
    model: Any | None,
    meta_df: pd.DataFrame,
    use_lstm: bool,
) -> tuple[np.ndarray, dict[str, float] | None]:
    if model is None or not _SB3_AVAILABLE:
        preds = meta_df["xgb_pred"].values.astype(float)
        if use_lstm and "lstm_pred" in meta_df.columns:
            preds = 0.5 * meta_df["xgb_pred"].values + 0.5 * meta_df["lstm_pred"].values
        return preds, None

    weights: list[float] = []
    preds: list[float] = []
    for rec in meta_df.to_dict("records"):
        row = {
            "y": float(rec.get("y", 0.0)),
            "xgb_pred": float(rec["xgb_pred"]),
            "lstm_pred": float(rec.get("lstm_pred", np.nan)) if use_lstm else np.nan,
        }
        xgb = float(row["xgb_pred"])
        lstm = row.get("lstm_pred")
        if lstm is None or not np.isfinite(lstm):
            lstm = xgb
        obs = np.array([xgb, float(lstm), xgb - float(lstm), abs(xgb - float(lstm))], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        w = float(np.clip(action[0], 0.0, 1.0))
        weights.append(w)
        preds.append(_combine_row(row, w))

    return np.array(preds, dtype=float), {
        "mean_blend_weight_on_xgb": float(np.mean(weights)),
        "std_blend_weight_on_xgb": float(np.std(weights)),
    }


def fallback_blend(meta_df: pd.DataFrame, use_lstm: bool) -> np.ndarray:
    if use_lstm and "lstm_pred" in meta_df.columns:
        return (meta_df["xgb_pred"].values + meta_df["lstm_pred"].values) / 2.0
    return meta_df["xgb_pred"].values.astype(float)
