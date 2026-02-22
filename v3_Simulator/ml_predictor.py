"""Lightweight online predictor for vehicle arrivals.

Implements an exponential moving average (EMA) predictor per direction.
This is intentionally simple: it provides a compact ML-like component
that can be used immediately without heavy dependencies or offline training.
"""
from typing import List

class EmaPredictor:
    def __init__(self, directions: int = 4, alpha: float = 0.3):
        self.alpha = float(alpha)
        self.directions = directions
        # initialize EMA states to small non-zero to avoid zero-division surprises
        self.ema = [1.0 for _ in range(directions)]

    def update(self, observed_counts: List[int]):
        """Update EMA with newly observed queue-lengths per direction.

        observed_counts should be a list of length `directions`.
        """
        for i in range(min(self.directions, len(observed_counts))):
            self.ema[i] = self.alpha * observed_counts[i] + (1 - self.alpha) * self.ema[i]

    def predict(self, horizon: int = 1) -> List[float]:
        """Predict arrivals for the next `horizon` periods.

        For EMA, a simple heuristic is to return the EMA scaled by horizon.
        """
        return [v * horizon for v in self.ema]

    def get_state(self):
        return list(self.ema)
