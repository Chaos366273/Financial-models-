"""Load config and resolve paths relative to package root."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict[str, Any]:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    overrides = {
        "master_panel": os.environ.get("DISSLAB_MASTER_PANEL"),
        "features_csv": os.environ.get("DISSLAB_FEATURES_CSV"),
        "fundamentals_csv": os.environ.get("DISSLAB_FUNDAMENTALS_CSV"),
        "risk_free_csv": os.environ.get("DISSLAB_RISK_FREE_CSV"),
        "panel_cache": os.environ.get("DISSLAB_PANEL_CACHE"),
    }
    for key in ("master_panel", "features_csv", "fundamentals_csv", "risk_free_csv", "panel_cache"):
        raw = overrides.get(key) or cfg["data"].get(key)
        if raw is None or raw == "":
            cfg["data"][key] = None
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        cfg["data"][key] = str(p)
    cfg["outputs"]["dir"] = str((ROOT / cfg["outputs"]["dir"]).resolve())
    cfg["outputs"]["bundle"] = str((ROOT / cfg["outputs"]["bundle"]).resolve())
    return cfg
