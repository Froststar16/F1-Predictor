"""
FastAPI backend. Run from the project root with:
    uvicorn backend.main:app --reload

Endpoints:
    GET /circuits          -> list of available circuit ids/names
    GET /predict/{circuit} -> top-3 picks + probabilities for every driver
"""
import os
import sys
import glob

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.predict import predict_for_circuit
from features.build_features import load_circuit_profile
from backend.schemas import RacePrediction, DriverPrediction

CIRCUITS_DIR = os.path.join(os.path.dirname(__file__), "..", "circuits")

app = FastAPI(title="F1 Race Predictor")

# Wide-open CORS so the simple frontend (opened as a local file or
# served separately) can call this API. Tighten this if you deploy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/circuits")
def list_circuits():
    files = glob.glob(os.path.join(CIRCUITS_DIR, "*.yaml"))
    circuits = []
    for f in files:
        circuit_id = os.path.splitext(os.path.basename(f))[0]
        if circuit_id.startswith("_"):
            continue
        profile = load_circuit_profile(circuit_id)
        circuits.append({"id": circuit_id, "name": profile["name"]})
    return sorted(circuits, key=lambda c: c["name"])


@app.get("/predict/{circuit_id}", response_model=RacePrediction)
def predict(circuit_id: str):
    try:
        circuit_name, result_df = predict_for_circuit(circuit_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Unknown circuit '{circuit_id}'")

    predictions = [DriverPrediction(**row) for row in result_df.to_dict(orient="records")]
    return RacePrediction(circuit_id=circuit_id, circuit_name=circuit_name, predictions=predictions)
