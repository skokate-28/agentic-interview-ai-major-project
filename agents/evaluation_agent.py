"""Answer evaluation agent for technical responses."""

from __future__ import annotations

from typing import Dict

from llm.groq_client import GroqClient
from llm.prompt_templates import build_evaluation_prompt
from utils.helpers import clamp, safe_json_loads


class EvaluationAgent:
    """Evaluate technical answers and return normalized scoring metadata."""

    WEIGHTS = {
        "correctness": 0.25,
        "concept_coverage": 0.20,
        "depth": 0.15,
        "clarity": 0.10,
        "confidence": 0.10,
        "overall_reasoning": 0.10,
        "error_penalty": -0.10,
    }

    def __init__(self, llm_client: GroqClient | None = None) -> None:
        self.llm_client = llm_client or GroqClient()

    def evaluate_answer(self, question: str, answer: str, skill: str) -> Dict[str, object]:
        """Evaluate a technical answer using 7 parameters and return structured output."""
        prompt = build_evaluation_prompt(question=question, answer=answer, skill=skill)
        response = self.llm_client.generate_response(prompt)
        parsed = safe_json_loads(response)
        normalized = self._normalize_payload(parsed if isinstance(parsed, dict) else {})

        if not isinstance(parsed, dict) or "final_score" not in parsed:
            normalized["final_score"] = self._compute_final_score(normalized)

        normalized["final_score"] = clamp(float(normalized["final_score"]))
        return normalized

    def _normalize_payload(self, payload: Dict[str, object]) -> Dict[str, object]:
        """Normalize model payload to expected schema and numeric bounds."""
        normalized = {
            "correctness": self._to_score(payload.get("correctness")),
            "concept_coverage": self._to_score(payload.get("concept_coverage")),
            "depth": self._to_score(payload.get("depth")),
            "clarity": self._to_score(payload.get("clarity")),
            "confidence": self._to_score(payload.get("confidence")),
            "error_penalty": self._to_score(payload.get("error_penalty")),
            "overall_reasoning": self._to_score(payload.get("overall_reasoning")),
            "weak_areas": self._normalize_weak_areas(payload.get("weak_areas")),
            "summary": str(payload.get("summary", "Evaluation generated successfully.")).strip(),
            "final_score": self._to_score(payload.get("final_score")),
        }

        if not normalized["summary"]:
            normalized["summary"] = "Evaluation generated successfully."

        return normalized

    @staticmethod
    def _to_score(value: object, default: float = 0.5) -> float:
        """Convert a value into a bounded [0,1] score."""
        try:
            return clamp(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_weak_areas(value: object) -> list[str]:
        """Normalize weak area field to a clean string list."""
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _compute_final_score(self, payload: Dict[str, object]) -> float:
        """Compute weighted final score when not provided by the model."""
        weighted_sum = 0.0
        for key, weight in self.WEIGHTS.items():
            score = self._to_score(payload.get(key), default=0.5)
            weighted_sum += weight * score

        return clamp(weighted_sum)
