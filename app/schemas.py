from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="One transaction: V1..V28 (PCA components), Amount (>= 0) and Hour (0-23).",
    )


class Signal(BaseModel):
    feature: str
    value: float
    typical: float = Field(..., description="Median of legitimate transactions in the training data.")
    z: float = Field(..., description="Robust z-score versus legitimate transactions.")


class PredictionResponse(BaseModel):
    fraud_score: float = Field(..., ge=0, le=1, description="Model score. Not a calibrated probability.")
    is_fraud: bool
    risk_level: Literal["low", "medium", "high"]
    threshold: float
    signals: list[Signal]
