"""Answer evaluation agent for technical responses."""

from __future__ import annotations

import re
from typing import Dict

from llm.groq_client import GroqClient
from utils.helpers import clamp, safe_json_loads


class EvaluationAgent:
    """Evaluate technical answers using a structured rubric-based scoring signal."""

    METRIC_WEIGHTS = {
        "accuracy": 0.6,
        "specificity": 0.2,
        "completeness": 0.2,
    }

    def __init__(self, llm_client: GroqClient | None = None) -> None:
        self.llm_client = llm_client or GroqClient()

    def evaluate_answer(self, question: str, answer: str, skill: str) -> Dict[str, object]:
        """Evaluate answer and return compatibility fields consumed by downstream BKT flow."""
        structured = self.evaluate_answer_llm(question=question, answer=answer)
        score = round(self._to_score(structured.get("score"), default=0.0), 4)
        metrics = structured.get("metrics", {})
        reasoning = str(structured.get("reasoning", "Evaluation generated successfully.")).strip()
        weak_areas = self._derive_weak_areas(metrics if isinstance(metrics, dict) else {}, skill)

        return {
            "final_score": score,
            "score": score,
            "metrics": metrics if isinstance(metrics, dict) else {},
            "reasoning": reasoning,
            "summary": reasoning,
            "weak_areas": weak_areas,
            # Backward-compatible aliases for any existing UI/debug integrations.
            "accuracy": self._to_score((metrics or {}).get("accuracy"), default=0.0),
            "specificity": self._to_score((metrics or {}).get("specificity"), default=0.0),
            "completeness": self._to_score((metrics or {}).get("completeness"), default=0.0),
        }

    def evaluate_answer_llm(self, question: str, answer: str) -> Dict[str, object]:
        """Structured LLM evaluation entrypoint with strict rubric and deterministic score recompute."""
        cleaned_question = str(question or "").strip()
        cleaned_answer = str(answer or "").strip()

        prompt = (
            "You are an expert technical interviewer.\n\n"
            "Your job is to score the answer based on true understanding, not surface fluency.\n\n"
            "========================\n"
            "CALIBRATION TARGETS\n\n"
            "- Weak answer: usually 0.30 to 0.50\n"
            "- Average answer: usually 0.60 to 0.75\n"
            "- Strong answer: usually 0.80 to 0.90\n"
            "- Exceptional answer: 0.90+ only with deep explanation and clear internal mechanism/example\n\n"
            "A basic correct answer with limited detail should NOT receive very high scores.\n\n"
            "========================\n"
            "SCORING METRICS (0 to 1)\n\n"
            "Evaluate each metric independently:\n\n"
            "1. Accuracy\n"
            "How factually correct is the answer? Are the claims true?\n"
            "How strong is the user's understanding of the topic?\n"
            "Penalize only factual errors and low understanding.\n\n"
            "2. Specificity\n"
            "Reward answers that are not vague and directly address what the question asks.\n"
            "If the answer is generic, evasive, or not exact to the question, reduce Specificity.\n\n"
            "3. Completeness\n"
            "Does the answer include the key points needed for the interviewer to be confident\n"
            "that the candidate truly knows the topic?\n"
            "If required key points are missing, reduce Completeness.\n\n"
            "========================\n"
            "SCORING DISCIPLINE\n\n"
            "- Do not inflate scores for brief correctness alone.\n"
            "- Reserve 0.85+ for answers with strong factual correctness, strong specificity, and strong completeness.\n"
            "- High scores require clear technical detail plus examples or internal mechanism explanation.\n\n"
            "========================\n"
            "OUTPUT FORMAT (STRICT)\n\n"
            "Return ONLY a JSON object:\n\n"
            f"Question:\n{cleaned_question}\n\n"
            f"Answer:\n{cleaned_answer}\n\n"
            "{\n"
            '"accuracy": float,\n'
            '"specificity": float,\n'
            '"completeness": float,\n'
            '"final_score": float,\n'
            '"reasoning": "brief explanation"\n'
            "}"
        )

        try:
            response = self.llm_client.generate_response(prompt)
            parsed = safe_json_loads(response)
            if not isinstance(parsed, dict):
                raise ValueError("LLM did not return a JSON object")

            metrics = self._normalize_metrics(parsed)
            final_score = self._recalculate_final_score(metrics)
            reasoning = str(parsed.get("reasoning", "")).strip() or "Structured rubric evaluation completed."

            self._print_eval_debug(metrics, final_score)
            return {
                "score": round(final_score, 4),
                "metrics": metrics,
                "reasoning": reasoning,
            }
        except Exception:
            return self._heuristic_structured_evaluation(cleaned_question, cleaned_answer)

    @staticmethod
    def _to_score(value: object, default: float = 0.5) -> float:
        """Convert a value into a bounded [0,1] score."""
        try:
            return clamp(float(value))
        except (TypeError, ValueError):
            return default

    def _normalize_metrics(self, payload: Dict[str, object]) -> Dict[str, float]:
        """Normalize required rubric metrics from payload."""
        return {
            "accuracy": self._to_score(payload.get("accuracy"), default=0.5),
            "specificity": self._to_score(payload.get("specificity"), default=0.5),
            "completeness": self._to_score(payload.get("completeness"), default=0.5),
        }

    def _enforce_accuracy_floor(
        self,
        metrics: Dict[str, float],
        payload: Dict[str, object],
    ) -> Dict[str, float]:
        """Keep accuracy reasonable when no explicit factual issue is indicated."""
        updated = dict(metrics)
        reasoning = str(payload.get("reasoning", "")).strip().lower()

        explicit_error_markers = [
            "factual error",
            "misconception",
            "contradict",
            "incorrect",
            "factually wrong",
            "inaccurate",
        ]
        explicit_no_error_markers = [
            "no factual error",
            "no factual errors",
            "no misconception",
            "no misconceptions",
            "no contradiction",
            "no contradictions",
            "factually correct",
            "conceptually correct",
            "no incorrect statements",
        ]

        has_error = any(marker in reasoning for marker in explicit_error_markers)
        has_no_error = any(marker in reasoning for marker in explicit_no_error_markers)

        relevance = self._to_score(updated.get("relevance"), default=0.0)
        if relevance >= 0.75 and not has_error:
            updated["accuracy"] = max(self._to_score(updated.get("accuracy"), default=0.0), 0.75)
        elif has_no_error:
            updated["accuracy"] = max(self._to_score(updated.get("accuracy"), default=0.0), 0.75)

        return updated

    def _recalculate_final_score(self, metrics: Dict[str, float]) -> float:
        """Recompute weighted score in code; never trust LLM final_score blindly."""
        w = self.METRIC_WEIGHTS
        score = (
            (w["accuracy"] * self._to_score(metrics.get("accuracy"), default=0.0))
            + (w["specificity"] * self._to_score(metrics.get("specificity"), default=0.0))
            + (w["completeness"] * self._to_score(metrics.get("completeness"), default=0.0))
        )
        return clamp(score)

    def _heuristic_structured_evaluation(self, question: str, answer: str) -> Dict[str, object]:
        """Fallback evaluator using keyword overlap and answer detail heuristics."""
        q_tokens = self._tokenize(question)
        a_tokens = self._tokenize(answer)
        overlap = len(q_tokens & a_tokens)
        q_count = max(1, len(q_tokens))
        overlap_ratio = overlap / q_count

        answer_len = len(a_tokens)
        has_reasoning_words = bool(
            re.search(r"\b(because|therefore|trade-?off|approach|reason|why|design|constraint)\b", answer.lower())
        )
        has_specific_terms = bool(
            re.search(r"\b(api|index|latency|cache|transaction|normalization|ac\w+|complexity|memory)\b", answer.lower())
        )

        # Fallback remains oral-friendly and allows excellent answers to reach top range.
        accuracy = clamp(
            (0.60 * overlap_ratio)
            + (0.25 if has_reasoning_words else 0.0)
            + (0.20 if has_specific_terms else 0.0)
        )
        specificity = clamp(0.30 + (0.45 * overlap_ratio) + (0.25 if has_specific_terms else 0.0))
        completeness = clamp((0.65 * overlap_ratio) + (0.35 if answer_len >= 18 else 0.05))

        metrics = {
            "accuracy": accuracy,
            "specificity": specificity,
            "completeness": completeness,
        }
        final_score = self._recalculate_final_score(metrics)

        reasoning = (
            "Fallback heuristic evaluation used due to LLM/JSON failure; "
            "scores are based on overlap, reasoning indicators, and specificity terms."
        )

        self._print_eval_debug(metrics, final_score)
        return {
            "score": round(final_score, 4),
            "metrics": metrics,
            "reasoning": reasoning,
        }

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize text into lowercase alpha-numeric terms."""
        return set(re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower()))

    def _derive_weak_areas(self, metrics: Dict[str, object], skill: str) -> list[str]:
        """Map low metric dimensions to weak-area labels used by downstream skill graph logic."""
        weak_areas: list[str] = []
        if self._to_score(metrics.get("accuracy"), default=1.0) < 0.6:
            weak_areas.append(f"{skill} conceptual accuracy")
        if self._to_score(metrics.get("specificity"), default=1.0) < 0.6:
            weak_areas.append(f"{skill} technical specificity")
        if self._to_score(metrics.get("completeness"), default=1.0) < 0.6:
            weak_areas.append(f"{skill} answer completeness")
        return weak_areas

    @staticmethod
    def _print_eval_debug(metrics: Dict[str, float], final_score: float) -> None:
        """Print mandatory structured debug logs for score tracing."""
        print("[LLM EVAL]")
        print(f"Accuracy: {metrics.get('accuracy', 0.0):.4f}")
        print(f"Specificity: {metrics.get('specificity', 0.0):.4f}")
        print(f"Completeness: {metrics.get('completeness', 0.0):.4f}")
        print(f"Final Score: {float(final_score):.4f}")
