"""
Pulls real historical race data using the FastF1 library.

This needs outbound internet access to F1's timing-data backend, so
run it from your own machine, not inside a network-restricted sandbox.
Output is cached as CSV in data/processed/, in the same shape
synthetic_data.py produces, so everything downstream (features,
training) works unchanged either way.

Install first:  pip install fastf1

Usage:
    python data/fetchers/fastf1_fetcher.py --start 2021 --end 2024
"""
import argparse
import os
import fastf1
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "processed", "race_history.csv")

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def fetch_season(year):
    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as e:
        print(f"Could not get schedule for{year}, skipping: {e}")
        return pd.DataFrame()
    
    rows = []
    for _, event in schedule.iterrows():
        if str(event.get("EventFormat", "")).lower() == "testing":
            continue
        try:
            session = fastf1.get_session(year, event["RoundNumber"], "R")
            session.load(laps=False, telemetry=False, weather=False, messages=False)
        except Exception as e:
            print(f"Skipping {event.get('EventName')} {year}: {e}")
            continue



        # circuit_id must match the filename of a circuits/*.yaml profile.
        # Adjust this mapping as needed so event names line up with your
        # circuit config filenames (e.g. "Italian Grand Prix" -> "monza").
        circuit_id = event["EventName"].lower().replace(" grand prix", "").replace(" ", "_")

        for _, res in session.results.iterrows():
            rows.append({
                "race_id": f"{year}_{event['RoundNumber']}",
                "circuit_id": circuit_id,
                "driver_id": res["Abbreviation"],
                "team_id": res["TeamName"],
                "grid_position": res["GridPosition"],
                "finish_position": res["Position"],
            })
    return pd.DataFrame(rows)


def main(start_year, end_year):
    all_rows = [fetch_season(y) for y in range(start_year, end_year + 1)]
    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(full)} rows to {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2021)
    parser.add_argument("--end", type=int, default=2024)
    args = parser.parse_args()
    main(args.start, args.end)
