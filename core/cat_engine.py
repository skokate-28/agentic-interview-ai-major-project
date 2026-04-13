"""Computerized Adaptive Testing (CAT) difficulty controller skeleton."""

from __future__ import annotations

class CATEngine:
    """Maintain and update interview difficulty from evaluation score."""

    def __init__(self, initial_difficulty: int = 3) -> None:
        self.difficulty = max(1, min(5, initial_difficulty))

    def update_difficulty(self, score: float) -> int:
        """Update difficulty using score thresholds and return the new level."""
        if score > 0.75:
            self.difficulty = min(5, self.difficulty + 1)
        elif score < 0.4:
            self.difficulty = max(1, self.difficulty - 1)

        return self.difficulty

    def get_difficulty(self) -> int:
        """Return the current CAT-controlled difficulty level."""
        return self.difficulty
