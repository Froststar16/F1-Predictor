"""
Glue code the backend calls: given a circuit id, loads that circuit's
profile + the current driver lineup, attaches circuit features, and
runs the simulation. This is the only file that needs to know how the
backend talks to the model.
"""
import os
import sys
import joblib
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from features.build_features import load_circuit_profile, TYRE_DEG_MAP
from model.simulate import simulate_race

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved", "model.pkl")
LINEUP_PATH = os.path.join(os.path.dirname(__file__), "..", "sample_data", "current_lineup.csv")


def predict_for_circuit(circuit_id, n_sims=10000):
    profile = load_circuit_profile(circuit_id)  # raises FileNotFoundError if unknown
    lineup = pd.read_csv(LINEUP_PATH)

    # NOTE: grid_position/recent_form here come from sample_data/current_lineup.csv,
    # a placeholder. In real use, refresh this file (or fetch live) with actual
    # qualifying results before each race weekend.
    lineup["overtaking_difficulty"] = profile["overtaking_difficulty"]
    lineup["tyre_deg_severity"] = TYRE_DEG_MAP[profile["tyre_deg_severity"]]
    lineup["safety_car_prob"] = profile["safety_car_prob"]
    lineup["rain_prob"] = profile["rain_prob"]
    lineup["pole_to_win_rate"] = profile["pole_to_win_rate"]

    bundle = joblib.load(MODEL_PATH)
    result = simulate_race(lineup, bundle, n_sims=n_sims)
    return profile["name"], result
