"""HR round agent with fixed interview questions and scoring support."""

from __future__ import annotations

import re
from typing import Dict, List

from llm.groq_client import GroqClient
from llm.prompt_templates import build_hr_evaluation_prompt
from utils.helpers import clamp, safe_json_loads


class HRAgent:
    """Conduct a fixed HR round and score soft skills from LLM + NLP signals."""

    WEAK_WORDS = ["maybe", "i think", "probably", "not sure", "guess"]

    def __init__(self, llm_client: GroqClient | None = None) -> None:
        self.llm_client = llm_client or GroqClient()
        self.questions = [
            "Tell me about yourself.",
            "Describe a challenging situation and how you handled it.",
            "What are your strengths and weaknesses?",
            "Tell me about a time you showed leadership.",
            "Why should we hire you?",
        ]

    def get_questions(self) -> list[str]:
        """Return the fixed HR round question list."""
        return self.questions

    def run_hr_round(self) -> Dict[str, object]:
        """Run the full non-adaptive HR round and return structured results."""
        per_question_results: List[Dict[str, object]] = []
        all_responses: List[str] = []

        for index, question in enumerate(self.questions, start=1):
            print(f"\nHR Question {index}: {question}")
            answer = input("Your Answer: ").strip()
            all_responses.append(answer)
            evaluation = self.evaluate_response(question=question, answer=answer)

            per_question_results.append(
                {
                    "question": question,
                    "answer": answer,
                    "scores": {
                        "communication": evaluation["communication"],
                        "confidence": evaluation["confidence"],
                        "leadership": evaluation["leadership"],
                        "problem_solving": evaluation["problem_solving"],
                    },
                    "confidence_penalty": evaluation["confidence_penalty"],
                    "summary": evaluation["summary"],
                }
            )

        return {
            "questions_asked": len(self.questions),
            "per_question": per_question_results,
            "average_scores": self._compute_averages(per_question_results),
            "responses": all_responses,
        }

    def evaluate_response(self, question: str, answer: str) -> Dict[str, object]:
        """Evaluate one HR answer and apply NLP-based confidence penalty."""
        prompt = build_hr_evaluation_prompt(question=question, answer=answer)
        response = self.llm_client.generate_response(prompt)
        parsed = safe_json_loads(response)

        communication = self._to_score(parsed.get("communication") if isinstance(parsed, dict) else None)
        llm_confidence = self._to_score(parsed.get("confidence") if isinstance(parsed, dict) else None)
        leadership = self._to_score(parsed.get("leadership") if isinstance(parsed, dict) else None)
        problem_solving = self._to_score(parsed.get("problem_solving") if isinstance(parsed, dict) else None)
        summary = (
            str(parsed.get("summary", "HR evaluation completed.")).strip()
            if isinstance(parsed, dict)
            else "HR evaluation completed."
        )

        confidence_penalty = self._confidence_penalty(answer)
        final_confidence = max(0.0, llm_confidence - confidence_penalty)

        return {
            "communication": communication,
            "confidence": clamp(final_confidence),
            "leadership": leadership,
            "problem_solving": problem_solving,
            "confidence_penalty": round(confidence_penalty, 4),
            "summary": summary,
        }

    @staticmethod
    def _to_score(value: object, default: float = 0.5) -> float:
        """Convert any numeric value into a bounded soft-skill score."""
        try:
            return clamp(float(value))
        except (TypeError, ValueError):
            return default

    def _confidence_penalty(self, answer: str) -> float:
        """Compute weak-language penalty from answer text."""
        words = re.findall(r"\b\w+\b", answer.lower())
        total_words = max(1, len(words))
        lower_answer = answer.lower()

        weak_count = 0
        for weak_word in self.WEAK_WORDS:
            weak_count += lower_answer.count(weak_word)

        return weak_count / total_words

    @staticmethod
    def _compute_averages(per_question_results: List[Dict[str, object]]) -> Dict[str, float]:
        """Compute average soft-skill scores across all HR answers."""
        if not per_question_results:
            return {
                "communication": 0.0,
                "confidence": 0.0,
                "leadership": 0.0,
                "problem_solving": 0.0,
            }

        totals = {
            "communication": 0.0,
            "confidence": 0.0,
            "leadership": 0.0,
            "problem_solving": 0.0,
        }

        for row in per_question_results:
            scores = row.get("scores", {})
            if not isinstance(scores, dict):
                continue
            for key in totals:
                totals[key] += float(scores.get(key, 0.0))

        count = max(1, len(per_question_results))
        return {key: round(value / count, 4) for key, value in totals.items()}
