"""
Monte Carlo simulation that converts the model's point predictions
(expected finishing position) into full probability distributions:
P(win), P(podium), P(points), etc.

Why simulate instead of training a separate classifier per outcome?
One simulation run gives every probability you want at once, and the
relative ordering between drivers stays internally consistent — no
contradictions between the "podium probability" and "win probability"
numbers, because they come from the same simulated races.
"""
import numpy as np
import pandas as pd


def simulate_race(driver_features_df, model_bundle, n_sims=10000, seed=None):
    """
    driver_features_df: one row per driver for the upcoming race, with
        columns matching model_bundle['feature_columns'] plus a
        'driver_id' column used for labeling output.
    """
    rng = np.random.default_rng(seed)
    model = model_bundle["model"]
    residuals = model_bundle["residuals"]
    feature_cols = model_bundle["feature_columns"]

    base_pred = model.predict(driver_features_df[feature_cols])
    n_drivers = len(driver_features_df)

    # Bootstrap noise from the model's real validation residuals rather
    # than assuming a Gaussian shape — race outcomes aren't symmetric
    # (a strategy gamble or DNF skews things much more than luck helps).
    noise = rng.choice(residuals, size=(n_sims, n_drivers), replace=True)
    simulated_scores = base_pred[np.newaxis, :] + noise  # shape (n_sims, n_drivers)

    # Rank within each simulated race (lower score = better finish)
    sim_ranks = simulated_scores.argsort(axis=1).argsort(axis=1) + 1  # 1-indexed

    results = []
    for i, driver_id in enumerate(driver_features_df["driver_id"]):
        positions = sim_ranks[:, i]
        results.append({
            "driver_id": driver_id,
            "team_id": driver_features_df["team_id"].iloc[i],
            "predicted_position": round(float(base_pred[i]), 2),
            "win_probability": float(np.mean(positions == 1)),
            "podium_probability": float(np.mean(positions <= 3)),
            "points_probability": float(np.mean(positions <= 10)),
        })

    out = pd.DataFrame(results).sort_values("podium_probability", ascending=False).reset_index(drop=True)
    out["predicted_podium"] = out.index < 3
    return out
