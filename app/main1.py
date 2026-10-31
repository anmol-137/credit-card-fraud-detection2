"""FastAPI application: JSON API under /api, static front end at /."""
from __future__ import annotations

import io
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .schemas import PredictionResponse, PredictRequest
from .service import FraudService, InputError

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = FraudService()      # load the model once at startup
    yield


app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="1.0.0",
    description="Scores card transactions with a Random Forest trained on the ULB credit-card dataset.",
    lifespan=lifespan,
)


@app.get("/api/health", tags=["meta"])
def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "service")}


@app.get("/api/model", tags=["meta"])
def model_info():
    """Model card: metrics, threshold, dataset facts and the feature list."""
    return app.state.service.card


@app.get("/api/samples", tags=["meta"])
def samples():
    """Real, unseen test-set transactions used by the demo buttons."""
    return json.loads((STATIC_DIR / "data" / "samples.json").read_text())


@app.post("/api/predict", response_model=PredictionResponse, tags=["predict"])
def predict(req: PredictRequest):
    try:
        return app.state.service.predict_one(req.features)
    except InputError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/api/predict/batch", tags=["predict"])
def predict_batch(file: UploadFile = File(...)):
    """Upload a CSV with V1..V28, Amount and Hour (or Time). Up to 5,000 rows / 5 MB."""
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 5 MB.")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=422, detail="Could not read the file as CSV.")
    try:
        return app.state.service.predict_batch(df)
    except InputError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
