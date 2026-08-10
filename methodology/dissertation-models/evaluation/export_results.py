"""Export Lab-compatible JSON bundles."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ResultsStore:
    def __init__(self) -> None:
        self.h1: list[dict] = []
        self.h2: list[dict] = []
        self.h3a: list[dict] = []
        self.h3b: list[dict] = []
        self.shap: list[dict] = []
        self._run_id = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self._data_context: dict[str, Any] = {}

    @property
    def run_id(self) -> str:
        return self._run_id

    def set_data_context(self, context: dict[str, Any]) -> None:
        self._data_context = dict(context)

    def _base_fields(self) -> dict[str, Any]:
        return {
            "run_id": self._run_id,
            "source": "python",
            "data_context": dict(self._data_context),
        }

    def add_h1(
        self,
        model_id: str,
        metrics: dict,
        fold: str,
        feature_set: str = "traditional",
        calculation: dict | None = None,
    ) -> None:
        row = {
            "model_id": model_id,
            "fold": fold,
            "feature_set": feature_set,
            **self._base_fields(),
            **metrics,
        }
        if calculation:
            row["calculation"] = calculation
        self.h1.append(row)

    def add_h2(
        self,
        model_id: str,
        without: dict,
        with_alt: dict,
        fold: str,
        calculation: dict | None = None,
    ) -> None:
        delta = None
        if without.get("rmse") is not None and with_alt.get("rmse") is not None:
            delta = with_alt["rmse"] - without["rmse"]
        row = {
            "model_id": model_id,
            "fold": fold,
            "without_alt_data": without,
            "with_alt_data": with_alt,
            "delta_rmse": delta,
            **self._base_fields(),
        }
        if calculation:
            row["calculation"] = calculation
        self.h2.append(row)

    def add_h3a(
        self,
        strategy_id: str,
        metrics: dict,
        period: str,
        calculation: dict | None = None,
    ) -> None:
        row = {
            "strategy_id": strategy_id,
            "period": period,
            **self._base_fields(),
            **metrics,
        }
        if calculation:
            row["calculation"] = calculation
        self.h3a.append(row)

    def add_h3b(
        self,
        model_id: str,
        metrics: dict,
        period: str,
        calculation: dict | None = None,
    ) -> None:
        row = {
            "model_id": model_id,
            "period": period,
            **self._base_fields(),
            **metrics,
        }
        if calculation:
            row["calculation"] = calculation
        self.h3b.append(row)

    def add_shap(self, model_id: str, features: list[dict], feature_set: str = "traditional") -> None:
        self.shap.append({
            "model_id": model_id,
            "feature_set": feature_set,
            "features": features,
            **self._base_fields(),
        })

    def model_status_updates(self) -> dict[str, str]:
        ids = set()
        for r in self.h1:
            ids.add(r["model_id"])
        for r in self.h3a:
            ids.add(r["strategy_id"])
        for r in self.h3b:
            ids.add(r["model_id"])
        for r in self.h2:
            ids.add(r["model_id"])
        return {mid: "pilot_done" for mid in ids}

    def to_bundle(self) -> dict[str, Any]:
        return {
            "schemaVersion": 2,
            "exportedAt": self._run_id,
            "data_context": self._data_context,
            "artifactsRoot": "outputs/artifacts",
            "results": {
                "h1": self.h1,
                "h2": self.h2,
                "h3a": self.h3a,
                "h3b": self.h3b,
                "shap": self.shap,
            },
            "model_status_updates": self.model_status_updates(),
            "summary": {
                "h1": len(self.h1),
                "h2": len(self.h2),
                "h3a": len(self.h3a),
                "h3b": len(self.h3b),
                "shap": len(self.shap),
                "trackable_models": len(self.model_status_updates()),
            },
        }

    @staticmethod
    def _result_key(bucket: str, item: dict) -> tuple:
        if bucket == "h3a":
            return (item.get("strategy_id"),)
        if bucket == "shap":
            return (item.get("model_id"), item.get("feature_set"))
        return (item.get("model_id"),)

    def _merge_existing(self, path: Path, bundle: dict[str, Any]) -> dict[str, Any]:
        if not path.is_file():
            return bundle
        try:
            with open(path, encoding="utf-8") as f:
                previous = json.load(f)
        except (OSError, json.JSONDecodeError):
            return bundle
        old_results = previous.get("results")
        if not isinstance(old_results, dict):
            return bundle
        for bucket in ("h1", "h2", "h3a", "h3b", "shap"):
            old_items = old_results.get(bucket, [])
            new_items = bundle["results"][bucket]
            if not isinstance(old_items, list):
                continue
            new_keys = {self._result_key(bucket, item) for item in new_items}
            bundle["results"][bucket] = [
                item for item in old_items
                if self._result_key(bucket, item) not in new_keys
            ] + new_items
        statuses = previous.get("model_status_updates", {})
        if isinstance(statuses, dict):
            bundle["model_status_updates"] = {**statuses, **bundle["model_status_updates"]}
        bundle["summary"] = {
            **{bucket: len(bundle["results"][bucket]) for bucket in ("h1", "h2", "h3a", "h3b", "shap")},
            "trackable_models": len(bundle["model_status_updates"]),
        }
        return bundle

    def save(self, path: str | Path, merge_existing: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = self.to_bundle()
        if merge_existing:
            bundle = self._merge_existing(path, bundle)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)
