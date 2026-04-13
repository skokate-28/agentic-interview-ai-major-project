"""Technical question generation agent."""

from __future__ import annotations

from llm.groq_client import GroqClient
from llm.prompt_templates import build_question_prompt


class QuestionAgent:
    """Generate technical interview questions for a skill and difficulty."""

    def __init__(self, llm_client: GroqClient | None = None) -> None:
        self.llm_client = llm_client or GroqClient()

    def generate_question(
        self,
        skill: str,
        previous_question: str,
        evaluation_summary: str,
        difficulty: int,
        weak_areas: list[str],
    ) -> str:
        """Generate the next question using the provided CAT-controlled difficulty."""
        normalized_difficulty = max(1, min(5, difficulty))
        prompt = build_question_prompt(
            skill=skill,
            previous_question=previous_question,
            evaluation_summary=evaluation_summary,
            difficulty=normalized_difficulty,
            weak_areas=weak_areas,
        )
        response = self.llm_client.generate_response(prompt)
        if response:
            return response.strip().splitlines()[0].strip()

        return self._fallback_question(skill, normalized_difficulty, weak_areas)

    @staticmethod
    def _fallback_question(skill: str, difficulty: int, weak_areas: list[str]) -> str:
        """Provide deterministic one-concept fallback questions by difficulty."""
        target = weak_areas[0] if weak_areas else f"a core concept in {skill}"

        templates = {
            1: f"Define {target} in the context of {skill}.",
            2: f"Explain how {target} works in {skill} with a simple example.",
            3: f"Compare {target} with a closely related concept in {skill}.",
            4: f"In a practical scenario, how would you apply {target} in {skill}?",
            5: f"Describe an edge case involving {target} in {skill} and how to solve it.",
        }
        return templates[max(1, min(5, difficulty))]
