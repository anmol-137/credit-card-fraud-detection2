"""Model loading, input validation and scoring. No web-framework code lives here, so it is easy to test."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_model.joblib"
CARD_PATH = BASE_DIR / "models" / "model_card.json"

MAX_BATCH_ROWS = 5000
N_SIGNALS = 5          # how many "unusual signals" to return per transaction
SIGNAL_MIN_Z = 2.0     # a feature must be at least this many robust std-devs from the legit median


class InputError(ValueError):
    """The caller sent data the model cannot score (maps to HTTP 422)."""


class FraudService:
    def __init__(self, model_path: Path = MODEL_PATH, card_path: Path = CARD_PATH) -> None:
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.features: list[str] = list(bundle["features"])
        self.threshold: float = float(bundle["threshold"])
        self._median = pd.Series(bundle["ref_median"])[self.features]
        self._scale = pd.Series(bundle["ref_scale"])[self.features]
        importance = pd.Series(bundle["importance"])
        self._signal_features = list(importance.sort_values(ascending=False).head(10).index)
        self.card: dict[str, Any] = json.loads(Path(card_path).read_text())

    # ---- helpers ---------------------------------------------------------------------------------------
    def risk_level(self, score: float) -> str:
        if score >= self.threshold:
            return "high"
        if score >= self.threshold / 2:
            return "medium"
        return "low"

    def _signals(self, row: pd.Series) -> list[dict[str, float | str]]:
        """Features that sit far from the typical legitimate transaction.

        This describes the input, it does not explain the model's decision."""
        z = ((row[self._signal_features] - self._median[self._signal_features])
             / self._scale[self._signal_features])
        z = z[z.abs() >= SIGNAL_MIN_Z].sort_values(key=lambda s: -s.abs()).head(N_SIGNALS)
        return [{"feature": f, "value": round(float(row[f]), 4),
                 "typical": round(float(self._median[f]), 4), "z": round(float(v), 1)}
                for f, v in z.items()]

    def _validate_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise InputError(f"Missing features: {', '.join(missing)}")
        X = df[self.features].apply(pd.to_numeric, errors="coerce")
        bad = X.isna().any(axis=1) | X.isin([math.inf, -math.inf]).any(axis=1)
        if bad.any():
            first = int(bad.to_numpy().nonzero()[0][0]) + 1
            raise InputError(f"{int(bad.sum())} row(s) have empty or non-numeric values (first problem: row {first}).")
        if (X["Amount"] < 0).any():
            raise InputError("Amount cannot be negative.")
        if not X["Hour"].between(0, 23).all():
            raise InputError("Hour must be between 0 and 23.")
        return X.astype(float)

    # ---- public API ------------------------------------------------------------------------------------
    def predict_one(self, features: dict[str, float]) -> dict[str, Any]:
        X = self._validate_frame(pd.DataFrame([features]))
        score = float(self.model.predict_proba(X)[0, 1])
        return {
            "fraud_score": round(score, 4),
            "is_fraud": score >= self.threshold,
            "risk_level": self.risk_level(score),
            "threshold": round(self.threshold, 4),
            "signals": self._signals(X.iloc[0]),
        }

    def predict_batch(self, df: pd.DataFrame) -> dict[str, Any]:
        if df.empty:
            raise InputError("The file has no rows.")
        if len(df) > MAX_BATCH_ROWS:
            raise InputError(f"Too many rows ({len(df):,}). The limit is {MAX_BATCH_ROWS:,} per upload.")
        df = df.copy()
        if "Hour" not in df.columns and "Time" in df.columns:      # raw Kaggle files have Time, not Hour
            df["Hour"] = ((pd.to_numeric(df["Time"], errors="coerce") // 3600) % 24)
        X = self._validate_frame(df)
        scores = self.model.predict_proba(X)[:, 1]
        flagged = scores >= self.threshold
        rows = []
        for i, (s, f) in enumerate(zip(scores, flagged), start=1):
            row = {"row": i, "fraud_score": round(float(s), 4), "is_fraud": bool(f),
                   "risk_level": self.risk_level(float(s)), "amount": round(float(X["Amount"].iloc[i - 1]), 2)}
            if "Class" in df.columns:
                row["actual"] = int(df["Class"].iloc[i - 1])
            rows.append(row)
        summary: dict[str, Any] = {"rows": len(rows), "flagged": int(flagged.sum()),
                                   "flagged_pct": round(float(flagged.mean() * 100), 2)}
        if "Class" in df.columns:
            actual = pd.to_numeric(df["Class"], errors="coerce").fillna(0).astype(int).to_numpy()
            summary["actual_fraud"] = int(actual.sum())
            summary["caught"] = int(((actual == 1) & flagged).sum())
            summary["false_alarms"] = int(((actual == 0) & flagged).sum())
        return {"summary": summary, "threshold": round(self.threshold, 4), "results": rows}
