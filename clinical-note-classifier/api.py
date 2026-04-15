"""FastAPI wrapper around the clinical note classifier.

Usage:
    uvicorn api:app --reload
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"note": "65 year old male with chest pain", "top_k": 3}'
"""

import os
import sys

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import load_model, predict

app = FastAPI(
    title="Clinical Note Classifier",
    description="Predict ICD-9 diagnosis categories from clinical notes using Bio_ClinicalBERT.",
)


# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)
model, encoder, tokenizer = load_model(device)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for the /predict endpoint."""
    note: str = Field(..., description="Clinical note text to classify.")
    top_k: int = Field(default=3, ge=1, le=19, description="Number of top predictions to return.")


class Prediction(BaseModel):
    """A single predicted category with its confidence score."""
    category: str
    confidence: float


class PredictResponse(BaseModel):
    """Response body for the /predict endpoint."""
    predictions: list[Prediction]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest) -> PredictResponse:
    """Predict ICD-9 chapter categories for a clinical note.

    Returns the top-k predictions sorted by confidence (descending).
    """
    results = predict(req.note, model, encoder, tokenizer, device, req.top_k)
    return PredictResponse(
        predictions=[
            Prediction(category=cat, confidence=round(conf, 4))
            for cat, conf in results
        ]
    )
