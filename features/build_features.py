"""
Turns raw historical race rows + circuit profiles into a feature
matrix the model can train on. This is the one place where "raw
results" and "circuit config" get combined — keep all feature logic
here so train.py and predict.py never duplicate it, and changing a
circuit's yaml automatically flows through everywhere.
"""
import os
import pandas as pd
import yaml

CIRCUITS_DIR = os.path.join(os.path.dirname(__file__), "..", "circuits")

TYRE_DEG_MAP = {"low": 0, "medium": 1, "high": 2}

FEATURE_COLUMNS = [
    "grid_position", "recent_form", "team_recent_form",
    "overtaking_difficulty", "tyre_deg_severity",
    "safety_car_prob", "rain_prob", "pole_to_win_rate",
]


def load_circuit_profile(circuit_id):
    path = os.path.join(CIRCUITS_DIR, f"{circuit_id}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def compute_driver_form(history_df, window=5):
    """Rolling average finish position per driver over their last N races
    BEFORE the current one (shift() prevents leaking the result we're
    trying to predict into its own feature)."""
    history_df = history_df.sort_values("race_id")
    history_df["recent_form"] = (
        history_df.groupby("driver_id")["finish_position"]
        .transform(lambda s: s.shift().rolling(window, min_periods=1).mean())
    )
    history_df["recent_form"] = history_df["recent_form"].fillna(10)  # new driver -> assume mid-grid
    return history_df


def compute_team_pace(history_df, window=5):
    history_df["team_recent_form"] = (
        history_df.groupby("team_id")["finish_position"]
        .transform(lambda s: s.shift().rolling(window, min_periods=1).mean())
    )
    history_df["team_recent_form"] = history_df["team_recent_form"].fillna(10)
    return history_df


def attach_circuit_features(history_df):
    profiles = {}
    for circuit_id in history_df["circuit_id"].unique():
        try:
            profiles[circuit_id] = load_circuit_profile(circuit_id)
        except FileNotFoundError:
            raise ValueError(
                f"No circuit profile found for '{circuit_id}'. "
                f"Add circuits/{circuit_id}.yaml (copy circuits/_template.yaml)."
            )

    rows = []
    for _, row in history_df.iterrows():
        profile = profiles[row["circuit_id"]]
        rows.append({
            "overtaking_difficulty": profile["overtaking_difficulty"],
            "tyre_deg_severity": TYRE_DEG_MAP[profile["tyre_deg_severity"]],
            "safety_car_prob": profile["safety_car_prob"],
            "rain_prob": profile["rain_prob"],
            "pole_to_win_rate": profile["pole_to_win_rate"],
        })
    circuit_feats = pd.DataFrame(rows, index=history_df.index)
    return pd.concat([history_df, circuit_feats], axis=1)


def build_feature_matrix(history_df):
    df = compute_driver_form(history_df.copy())
    df = compute_team_pace(df)
    df = attach_circuit_features(df)
    X = df[FEATURE_COLUMNS]
    y = df["finish_position"]
    meta = df[["race_id", "circuit_id", "driver_id", "team_id"]]
    return X, y, meta
