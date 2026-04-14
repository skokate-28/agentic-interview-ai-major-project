"""Skill graph data structure for tracking per-skill interview state."""

from __future__ import annotations

import re
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
        question_difficulty: int = 1,
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
            self.add_weak_area(concept, question_difficulty)

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

    def add_weak_area(self, concept: str, question_difficulty: int = 1) -> None:
        """Track weak concept frequency and difficulty metadata."""
        cleaned = self._normalize_weak_area_key(concept)
        if not cleaned:
            return

        normalized_difficulty = _clamp(question_difficulty / 5.0)

        existing = self.weak_areas.get(cleaned)
        if existing is None:
            self.weak_areas[cleaned] = {
                "frequency": 1,
                "difficulty": round(normalized_difficulty, 4),
            }
        else:
            frequency = int(existing.get("frequency", 0)) + 1
            existing_difficulty = float(existing.get("difficulty", 0.3))
            self.weak_areas[cleaned] = {
                "frequency": frequency,
                "difficulty": round(max(existing_difficulty, normalized_difficulty), 4),
            }

        self._prune_weak_areas(max_weak_areas=3)

    @staticmethod
    def _normalize_weak_area_key(concept: str) -> str:
        """Normalize near-duplicate weak-area concepts into a stable key."""
        text = concept.strip().lower()
        if not text:
            return ""

        if "mutable" in text and "object" in text:
            return "mutable object handling"

        tokens = re.findall(r"\b[a-z]+\b", text)
        synonym_map = {
            "objects": "object",
            "handling": "handle",
            "handles": "handle",
            "handled": "handle",
            "cases": "case",
            "edges": "edge",
        }
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "with",
            "for",
            "of",
            "in",
            "to",
            "complex",
        }

        normalized_tokens: list[str] = []
        for token in tokens:
            mapped = synonym_map.get(token, token)
            if mapped in stop_words:
                continue
            if mapped not in normalized_tokens:
                normalized_tokens.append(mapped)

        if not normalized_tokens:
            return text

        return " ".join(normalized_tokens[:4])

    def _prune_weak_areas(self, max_weak_areas: int) -> None:
        """Keep only top weak areas ranked by difficulty then frequency."""
        if len(self.weak_areas) <= max_weak_areas:
            return

        ranked = sorted(
            self.weak_areas.items(),
            key=lambda item: (
                -float(item[1].get("difficulty", 0.0)),
                -int(item[1].get("frequency", 0)),
                item[0],
            ),
        )
        self.weak_areas = dict(ranked[:max_weak_areas])

    def get_summary(self) -> Dict[str, object]:
        """Return a serializable summary of skill state."""
        return {
            "skill": self.skill_name,
            "proficiency": round(self.proficiency, 4),
            "confidence": round(self.confidence, 4),
            "history": self.history,
            "weak_areas": self.weak_areas,
        }
