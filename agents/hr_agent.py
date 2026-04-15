"""HR round agent with fixed interview questions and scoring support."""

from __future__ import annotations

from typing import Dict, List

from llm.groq_client import GroqClient
from llm.prompt_templates import build_hr_evaluation_prompt
from utils.helpers import clamp, safe_json_loads


class HRAgent:
    """Conduct a fixed HR round and score role-based soft skills."""

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
                        "leadership": evaluation["leadership"],
                        "problem_solving": evaluation["problem_solving"],
                        "adaptability": evaluation["adaptability"],
                        "teamwork": evaluation["teamwork"],
                    },
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
        """Evaluate one HR answer using role-focused soft-skill dimensions."""
        prompt = build_hr_evaluation_prompt(question=question, answer=answer)
        response = self.llm_client.generate_response(prompt)
        parsed = safe_json_loads(response)

        leadership = self._to_score(parsed.get("leadership") if isinstance(parsed, dict) else None)
        problem_solving = self._to_score(parsed.get("problem_solving") if isinstance(parsed, dict) else None)
        adaptability = self._to_score(parsed.get("adaptability") if isinstance(parsed, dict) else None)
        teamwork = self._to_score(parsed.get("teamwork") if isinstance(parsed, dict) else None)
        summary = (
            str(parsed.get("summary", "HR evaluation completed.")).strip()
            if isinstance(parsed, dict)
            else "HR evaluation completed."
        )

        return {
            "leadership": leadership,
            "problem_solving": problem_solving,
            "adaptability": adaptability,
            "teamwork": teamwork,
            "summary": summary,
        }

    @staticmethod
    def _to_score(value: object, default: float = 0.5) -> float:
        """Convert any numeric value into a bounded soft-skill score."""
        try:
            return clamp(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _compute_averages(per_question_results: List[Dict[str, object]]) -> Dict[str, float]:
        """Compute average soft-skill scores across all HR answers."""
        if not per_question_results:
            return {
                "leadership": 0.0,
                "problem_solving": 0.0,
                "adaptability": 0.0,
                "teamwork": 0.0,
            }

        totals = {
            "leadership": 0.0,
            "problem_solving": 0.0,
            "adaptability": 0.0,
            "teamwork": 0.0,
        }

        for row in per_question_results:
            scores = row.get("scores", {})
            if not isinstance(scores, dict):
                continue
            for key in totals:
                totals[key] += float(scores.get(key, 0.0))

        count = max(1, len(per_question_results))
        return {key: round(value / count, 4) for key, value in totals.items()}
