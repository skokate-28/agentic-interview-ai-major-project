"""Technical question generation agent."""

from __future__ import annotations

from typing import Any

from llm.groq_client import GroqClient


def clean_and_validate_question(q: str) -> str | None:
    """Return a single clean question or None when model output is invalid."""
    if not q:
        return None

    q = q.strip()

    # Reject meta-text prefixes that break interview UI flow.
    banned_phrases = [
        "here is",
        "here's",
        "rewritten",
        "oral interview",
        "example",
        "question:",
        "output:",
        "format:",
    ]

    lower_q = q.lower()
    for phrase in banned_phrases:
        if lower_q.startswith(phrase):
            return None

    # Must be a valid question.
    if "?" not in q:
        return None

    # Keep exactly one complete question.
    q = q.split("?")[0].strip() + "?"
    return q


class QuestionAgent:
    """Generate strict skill-scoped technical interview questions."""

    def __init__(self, llm_client: GroqClient | None = None) -> None:
        self.llm_client = llm_client or GroqClient()
        self.last_source_type = "llm"

    def generate_bkt_question(
        self,
        skill: str,
        level: str,
        question_type: str,
        modality: str,
        previous_questions: list[str],
        weakness_hint: str | None = None,
    ) -> str:
        """Generate one technical question using BKT-derived difficulty and question type."""
        current_skill = str(skill).strip()
        safe_level = str(level).strip() or "easy"
        safe_type = str(question_type).strip() or "definition"
        safe_modality = str(modality).strip().lower() or "oral"
        previous = [q.strip() for q in previous_questions if str(q).strip()]
        previous_block = "\n".join(f"- {q}" for q in previous[-10:]) if previous else "- None"
        oral_block = (
            "This is an ORAL INTERVIEW.\n"
            "The candidate will answer verbally (not by writing code).\n"
            "Therefore:\n"
            "- Do NOT ask for full code or implementations\n"
            "- Do NOT require exact syntax\n"
            "- Questions must be answerable through explanation\n"
            "Focus on approach, logic, reasoning, and conceptual understanding."
        )
        type_instruction = self._question_type_instruction(safe_type)

        normalized_hint = str(weakness_hint or "").strip()
        if normalized_hint:
            prompt = (
                f"Generate a {safe_level} level interview question for {current_skill}.\n\n"
                f"Modality: {safe_modality}.\n"
                f"{oral_block}\n\n"
                f"Question-type guidance: {type_instruction}\n\n"
                "Focus on a concept similar to:\n"
                f'"{normalized_hint}"\n\n'
                "Do not repeat the exact same question.\n"
                "Make it precise and technical.\n\n"
                f"Previous questions:\n{previous_block}\n\n"
                "Return ONLY the final interview question.\n"
                "Do NOT include any explanation, prefix, labels, or meta-text.\n"
                "Do NOT include phrases like 'Here is', 'Here's', 'Rewritten', or 'Example'.\n"
                "Output must be exactly one complete question ending with a question mark.\n\n"
                "Output format:\n"
                "<question>"
            )
        else:
            prompt = (
                f"Generate a {safe_level} difficulty interview question for the skill: {current_skill}.\n"
                f"Question type: {safe_type}.\n\n"
                f"Modality: {safe_modality}.\n"
                f"{oral_block}\n\n"
                f"Question-type guidance: {type_instruction}\n\n"
                "The question must:\n\n"
                "- Focus on a specific concept\n"
                "- Be technical and precise\n"
                "- Not be vague\n"
                "- Not repeat previous questions\n\n"
                f"Previous questions:\n{previous_block}\n\n"
                "Return ONLY the final interview question.\n"
                "Do NOT include any explanation, prefix, labels, or meta-text.\n"
                "Do NOT include phrases like 'Here is', 'Here's', 'Rewritten', or 'Example'.\n"
                "Output must be exactly one complete question ending with a question mark.\n\n"
                "Output format:\n"
                "<question>"
            )

        candidate = ""
        try:
            response = self.llm_client.generate_response(prompt)
            candidate = clean_and_validate_question(self._clean_question_text(response)) or ""
        except Exception:
            candidate = ""

        if candidate and self._requires_exact_code_output(candidate):
            rewrite_prompt = (
                "Convert this into a verbal explanation-style interview question for an oral interview.\n"
                "Keep the same skill intent and difficulty.\n"
                "Do not ask for writing code, full implementation, or exact syntax.\n"
                f"Question type guidance: {type_instruction}\n"
                "Return ONLY the final interview question.\n"
                "Do NOT include any explanation, prefix, labels, or meta-text.\n"
                "Do NOT include phrases like 'Here is', 'Here's', 'Rewritten', or 'Example'.\n"
                "Output must be exactly one complete question ending with a question mark.\n\n"
                "Output format:\n"
                "<question>\n\n"
                f"Original question: \"{candidate}\""
            )
            try:
                rewrite_response = self.llm_client.generate_response(rewrite_prompt)
                rewritten = clean_and_validate_question(self._clean_question_text(rewrite_response))
                if rewritten:
                    candidate = rewritten
            except Exception:
                pass

        if candidate and self._requires_exact_code_output(candidate):
            softened = self._soften_to_oral(candidate, current_skill, safe_type)
            candidate = clean_and_validate_question(softened) or ""

        if candidate:
            self.last_source_type = "llm"
            return candidate

        self.last_source_type = "fallback"
        return self._fallback_question(
            skill=current_skill,
            level=safe_level,
            previous_questions=previous,
        )

    @staticmethod
    def _clean_question_text(response: object) -> str:
        """Extract a single question line from model output."""
        lines = [line.strip() for line in str(response).splitlines() if line.strip()]
        if not lines:
            return ""

        first = lines[0].strip().strip('"').strip()
        lowered = first.lower()
        if lowered.startswith("question:"):
            first = first.split(":", 1)[1].strip()
        return first

    @staticmethod
    def _question_type_instruction(question_type: str) -> str:
        """Return phrasing guidance so each question type remains verbal-first."""
        lowered = str(question_type).strip().lower()
        if lowered == "application":
            return "Ask how the candidate would approach solving the problem; do not ask them to write code."
        if lowered == "basic":
            return "Keep the question conceptual and explanation-driven."
        if lowered == "scenario":
            return "Keep the question as verbal reasoning through decisions and trade-offs."
        return "Keep the question answerable verbally using clear reasoning and conceptual depth."

    @staticmethod
    def _requires_exact_code_output(question: str) -> bool:
        """Detect prompts that request coding output rather than verbal reasoning."""
        lowered = str(question).strip().lower()
        code_markers = [
            "write a function",
            "write code",
            "implement",
            "code snippet",
            "exact syntax",
            "provide sql",
            "write a query",
            "output of this code",
            "complete the code",
        ]
        return any(marker in lowered for marker in code_markers)

    @staticmethod
    def _soften_to_oral(question: str, skill: str, question_type: str) -> str:
        """Fallback phrasing to keep a question oral-answerable when rewrite fails."""
        lowered_type = str(question_type).strip().lower()
        base = str(question).strip().rstrip("?.")
        if lowered_type == "application":
            return (
                f"For {skill}, how would you approach solving this problem and explain your reasoning: {base}?"
            )
        if lowered_type == "basic":
            return f"In {skill}, explain the concept behind this topic in your own words: {base}?"
        if lowered_type == "scenario":
            return f"Consider this {skill} scenario: {base}. How would you reason through your decision?"
        return f"For {skill}, explain your approach and reasoning for this topic: {base}?"

    def _fallback_question(self, skill: str, level: str, previous_questions: list[str]) -> str:
        """Select a static fallback question only when LLM output is unavailable."""
        level_to_difficulty = {
            "easy": 1,
            "easy-medium": 2,
            "medium": 3,
            "hard": 4,
            "extreme-hard": 5,
        }
        target = level_to_difficulty.get(str(level).strip(), 3)
        candidates = self.get_candidate_questions(skill)
        ranked = sorted(
            candidates,
            key=lambda q: abs(int(q.get("difficulty", target)) - target),
        )

        previous_set = {q.strip() for q in previous_questions if q.strip()}
        for row in ranked:
            question = str(row.get("question", "")).strip()
            cleaned = clean_and_validate_question(question)
            if cleaned and cleaned not in previous_set:
                return cleaned

        fallback_first = str(ranked[0].get("question", "")).strip() if ranked else ""
        return clean_and_validate_question(fallback_first) or ""

    @staticmethod
    def get_candidate_questions(skill: str) -> list[dict[str, Any]]:
        """Return structured, skill-filtered question candidates."""
        current_skill = str(skill).strip()
        question_bank = [
            {
                "question": f"How would you describe your experience with {current_skill}?",
                "skill": current_skill,
                "difficulty": 1,
            },
            {
                "question": f"What are common pitfalls in {current_skill} and how do you avoid them?",
                "skill": current_skill,
                "difficulty": 2,
            },
            {
                "question": f"How would you compare two approaches you use in {current_skill} for maintainable design?",
                "skill": current_skill,
                "difficulty": 3,
            },
            {
                "question": f"Can you describe a production incident involving {current_skill} and how you resolved it?",
                "skill": current_skill,
                "difficulty": 4,
            },
            {
                "question": f"How would you optimize and scale a complex {current_skill} workload?",
                "skill": current_skill,
                "difficulty": 5,
            },
        ]

        for row in question_bank:
            if not isinstance(row, dict) or not {"question", "skill", "difficulty"}.issubset(row.keys()):
                raise Exception("Invalid question bank entry")

        return [q for q in question_bank if q["skill"] == current_skill]

    @staticmethod
    def select_question_based_on_difficulty(
        candidate_questions: list[dict[str, Any]],
        difficulty_level: int,
    ) -> dict[str, Any]:
        """Select a skill-scoped question by nearest difficulty."""
        if not candidate_questions:
            raise Exception("No valid questions for current skill")

        bounded = max(1, min(5, int(difficulty_level)))
        selected = min(
            candidate_questions,
            key=lambda q: abs(int(q.get("difficulty", bounded)) - bounded),
        )
        return selected

    def generate_question(
        self,
        skill: str,
        previous_question: str,
        evaluation_summary: str,
        difficulty: int,
        weak_areas: list[str],
    ) -> str:
        """Generate question strictly from current-skill candidates."""
        _ = (previous_question, evaluation_summary, weak_areas)
        current_skill = str(skill).strip()
        candidate_questions = self.get_candidate_questions(current_skill)
        next_question = self.select_question_based_on_difficulty(candidate_questions, difficulty)

        print("Current Skill:", current_skill)
        print("Selected Question Skill:", next_question["skill"])

        if next_question["skill"] != current_skill:
            raise Exception(
                f"SKILL DRIFT ERROR: expected {current_skill}, got {next_question['skill']}"
            )

        return str(next_question["question"])
