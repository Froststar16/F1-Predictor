import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CIRCUITS_DIR = os.path.join(BASE_DIR, "circuits")
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "model", "saved")
SAMPLE_DATA_DIR = os.path.join(BASE_DIR, "sample_data")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

N_SIMULATIONS = 10000
