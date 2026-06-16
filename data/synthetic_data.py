"""
Generates a synthetic historical race dataset purely so the pipeline
(features -> training -> simulation) can be tested end-to-end without
network access. Replace this with real data pulled via
data/fetchers/fastf1_fetcher.py once you're running this locally with
internet access.
"""
import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

DRIVERS = [
    ("VER", "Red Bull", 0.95), ("PER", "Red Bull", 0.80),
    ("LEC", "Ferrari", 0.88), ("SAI", "Ferrari", 0.85),
    ("HAM", "Mercedes", 0.86), ("RUS", "Mercedes", 0.83),
    ("NOR", "McLaren", 0.87), ("PIA", "McLaren", 0.81),
    ("ALO", "Aston Martin", 0.78), ("STR", "Aston Martin", 0.65),
    ("GAS", "Alpine", 0.68), ("OCO", "Alpine", 0.67),
    ("ALB", "Williams", 0.66), ("SAR", "Williams", 0.58),
    ("TSU", "RB", 0.64), ("RIC", "RB", 0.66),
    ("BOT", "Sauber", 0.60), ("ZHO", "Sauber", 0.55),
    ("MAG", "Haas", 0.61), ("HUL", "Haas", 0.63),
]

CIRCUITS = ["monaco", "monza", "spa"]

OVERTAKING_DIFFICULTY = {"monaco": 9.5, "monza": 2.5, "spa": 3.5}


def generate_race(circuit_id, race_id):
    rows = []
    skill_noise = RNG.normal(0, 0.05, size=len(DRIVERS))
    grid = RNG.permutation(len(DRIVERS))

    diff = OVERTAKING_DIFFICULTY[circuit_id]
    grid_weight = diff / 10
    skill_weight = 1 - grid_weight

    for i, (driver, team, skill) in enumerate(DRIVERS):
        true_skill = skill + skill_noise[i]
        grid_pos = grid[i] + 1
        score = skill_weight * (1 - true_skill) + grid_weight * (grid_pos / len(DRIVERS))
        score += RNG.normal(0, 0.08)  # race-day randomness
        rows.append({
            "race_id": race_id, "circuit_id": circuit_id,
            "driver_id": driver, "team_id": team,
            "grid_position": grid_pos, "_score": score,
        })

    df = pd.DataFrame(rows)
    df["finish_position"] = df["_score"].rank(method="first").astype(int)
    return df.drop(columns="_score")


def generate_history(n_races_per_circuit=15):
    races = []
    race_id = 0
    for circuit in CIRCUITS:
        for _ in range(n_races_per_circuit):
            races.append(generate_race(circuit, race_id))
            race_id += 1
    return pd.concat(races, ignore_index=True)


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "processed", "race_history.csv")
    history = generate_history()
    history.to_csv(out_path, index=False)
    print(f"Wrote {len(history)} rows to {out_path}")
