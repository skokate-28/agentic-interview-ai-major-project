"""Skill graph data structure for tracking per-skill interview state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a float value to an inclusive range."""
    return max(low, min(high, value))


@dataclass
class SkillGraph:
    """Represents adaptive state for a single skill during interviews."""

    skill_name: str
    proficiency: float = 0.0
    confidence: float = 0.0
    history: List[Dict[str, float]] = field(default_factory=list)
    weak_areas: Dict[str, Dict[str, float | int]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Backward-compatible alias for skill name."""
        return self.skill_name

    def update(
        self,
        score: float,
        probability: float,
        weak_areas: list[str],
        question_count: int,
        max_questions: int,
    ) -> None:
        """Update skill proficiency, confidence, and weak-area metadata."""
        normalized_score = _clamp(score)
        normalized_probability = _clamp(probability)
        previous_proficiency = self.proficiency
        delta = abs(normalized_probability - previous_proficiency)

        self.history.append(
            {
                "score": round(normalized_score, 4),
                "probability": round(normalized_probability, 4),
            }
        )

        self.proficiency = normalized_probability
        normalized_progress = min(1.0, max(0.0, question_count / max(1, max_questions)))
        confidence = 0.5 * normalized_progress + 0.5 * (1 - delta)
        self.confidence = _clamp(confidence)

        for concept in weak_areas:
            self.add_weak_area(concept)

    def get_latest_probability(self) -> float:
        """Return the most recent skill mastery probability."""
        if not self.history:
            return self.proficiency
        return float(self.history[-1]["probability"])

    def update_proficiency(self, score: float) -> float:
        """Backward-compatible helper for legacy callers."""
        self.update(
            score=score,
            probability=score,
            weak_areas=[],
            question_count=len(self.history) + 1,
            max_questions=max(1, len(self.history) + 1),
        )
        return self.proficiency

    def add_weak_area(self, concept: str) -> None:
        """Track weak concept frequency and difficulty metadata."""
        cleaned = concept.strip()
        if not cleaned:
            return

        existing = self.weak_areas.get(cleaned)
        if existing is None:
            self.weak_areas[cleaned] = {
                "frequency": 1,
                "difficulty": 0.3,
            }
            return

        frequency = int(existing.get("frequency", 0)) + 1
        difficulty = float(existing.get("difficulty", 0.3))
        self.weak_areas[cleaned] = {
            "frequency": frequency,
            "difficulty": _clamp(min(1.0, difficulty + 0.1)),
        }

    def get_summary(self) -> Dict[str, object]:
        """Return a serializable summary of skill state."""
        return {
            "skill": self.skill_name,
            "proficiency": round(self.proficiency, 4),
            "confidence": round(self.confidence, 4),
            "history": self.history,
            "weak_areas": self.weak_areas,
        }
