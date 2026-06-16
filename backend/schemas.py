from typing import List
from pydantic import BaseModel


class DriverPrediction(BaseModel):
    driver_id: str
    team_id: str
    predicted_position: float
    win_probability: float
    podium_probability: float
    points_probability: float
    predicted_podium: bool


class RacePrediction(BaseModel):
    circuit_id: str
    circuit_name: str
    predictions: List[DriverPrediction]
