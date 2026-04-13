"""Bayesian Knowledge Tracing (BKT) model skeleton."""

from __future__ import annotations

from config import settings


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a float value to an inclusive range."""
    return max(low, min(high, value))


class BKTModel:
    """Simple BKT-style probability updater."""

    def __init__(self, initial_probability: float = 0.0, alpha: float = settings.BKT_ALPHA) -> None:
        self.current_probability = _clamp(initial_probability)
        self.alpha = alpha

    def update(self, score: float) -> float:
        """Update the mastery probability using a learning-rate smoothing formula."""
        bounded_score = _clamp(score)
        bounded_probability = _clamp(self.current_probability)
        self.current_probability = bounded_probability + self.alpha * (
            bounded_score - bounded_probability
        )
        self.current_probability = _clamp(self.current_probability)
        return self.current_probability
