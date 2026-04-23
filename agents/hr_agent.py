"""HR round agent with fixed interview questions and scoring support."""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List

from llm.groq_client import GroqClient
from llm.prompt_templates import build_hr_evaluation_prompt, build_hr_question_generation_prompt
from utils.helpers import clamp, safe_json_loads


class HRAgent:
    """Conduct a fixed HR round and score role-based soft skills."""

    METRIC_WEIGHTS = {
        "relevance": 0.20,
        "reasoning": 0.20,
        "clarity": 0.15,
        "specificity": 0.15,
        "outcome": 0.15,
        "authenticity": 0.10,
        "tone": 0.05,
    }

    HR_SKILLS = [
        "leadership",
        "problem_solving",
        "adaptability",
        "teamwork",
    ]

    FALLBACK_QUESTIONS = {
        "leadership": [
            "Tell me about a time you had to make a difficult decision for your team under uncertainty. How did you decide and what happened?",
            "Describe a situation where you guided others through a setback. What actions did you take and what was the outcome?",
            "Share a real example where you had to align people with different opinions. How did you lead the discussion and what result did you achieve?",
        ],
        "problem_solving": [
            "Tell me about a time you faced a difficult problem with limited resources. How did you reason through it and what was the impact?",
            "Describe a situation where your first approach did not work. How did you adapt your thinking and what outcome did you get?",
            "Share an example of solving a high-pressure challenge with constraints. How did you prioritize decisions and what result followed?",
        ],
        "adaptability": [
            "Tell me about a time priorities changed suddenly during important work. How did you adapt and what was the final outcome?",
            "Describe a situation where you had to learn something quickly due to change. How did you handle it and what impact did it have?",
            "Share an example of working through uncertainty or ambiguity. How did you stay effective and what happened in the end?",
        ],
        "teamwork": [
            "Tell me about a time you had a conflict with a teammate. How did you handle it and what was the outcome?",
            "Describe a situation where coordination broke down in a team. What did you do to restore collaboration and what result followed?",
            "Share an example where you had to work closely with people with different working styles. How did you align and what impact did that have?",
        ],
    }

    def __init__(self, llm_client: GroqClient | None = None) -> None:
        self.llm_client = llm_client or GroqClient()
        self.skill_questions = self._build_skill_questions()
        self.questions = [row["question"] for row in self.skill_questions]

    def _build_skill_questions(self) -> list[dict[str, str]]:
        """Generate one scenario-based HR question per skill with anti-repetition fallback."""
        asked: set[str] = set()
        generated: list[dict[str, str]] = []

        for skill in self.HR_SKILLS:
            question = self._generate_question_for_skill(skill=skill, asked_questions=asked)
            asked.add(question)
            generated.append({"skill": skill, "question": question})

        return generated

    def _generate_question_for_skill(self, skill: str, asked_questions: set[str]) -> str:
        """Generate a high-quality HR question and validate constraints before use."""
        previous = list(asked_questions)
        for _ in range(4):
            prompt = build_hr_question_generation_prompt(skill=skill, previous_questions=previous)
            response = self.llm_client.generate_response(prompt)
            candidate = self._clean_question_text(response)
            if self._is_valid_hr_question(candidate, skill=skill, asked_questions=asked_questions):
                return candidate

        fallback_pool = list(self.FALLBACK_QUESTIONS.get(skill, []))
        random.shuffle(fallback_pool)
        for question in fallback_pool:
            candidate = self._clean_question_text(question)
            if self._is_valid_hr_question(candidate, skill=skill, asked_questions=asked_questions):
                return candidate

        return fallback_pool[0] if fallback_pool else (
            "Tell me about a real situation where you used this soft skill. "
            "How did you handle it and what was the outcome?"
        )

    @staticmethod
    def _clean_question_text(response: object) -> str:
        """Extract a single plain-text question from LLM output."""
        text = str(response).strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""

        first = lines[0].strip().strip('"').strip()
        lowered = first.lower()
        if lowered.startswith("question:"):
            first = first.split(":", 1)[1].strip()

        if "?" in first:
            first = first.split("?")[0].strip() + "?"
        elif first:
            first = first.rstrip(".") + "?"

        return first

    def _is_valid_hr_question(self, question: str, skill: str, asked_questions: set[str]) -> bool:
        """Validate scenario quality, oral suitability, skill focus, and repetition constraints."""
        candidate = str(question or "").strip()
        if not candidate:
            return False
        if candidate in asked_questions:
            return False

        lowered = candidate.lower()
        banned_generic = [
            "what is",
            "why is",
            "define",
            "importance of",
        ]
        if any(lowered.startswith(prefix) for prefix in banned_generic):
            return False

        scenario_markers = [
            "tell me about a time",
            "describe a situation",
            "share an example",
            "walk me through",
            "when you",
        ]
        if not any(marker in lowered for marker in scenario_markers):
            return False

        skill_markers = {
            "communication": ["explain", "misunderstanding", "persuade", "communicat", "message"],
            "teamwork": ["team", "collabor", "conflict", "coordinate", "teammate"],
            "leadership": ["lead", "decision", "guide", "owner", "influence"],
            "problem_solving": ["problem", "challenge", "constraint", "resolve", "approach"],
            "adaptability": ["change", "adapt", "uncertainty", "shift", "unexpected"],
        }
        markers = skill_markers.get(skill, [skill])
        if not any(token in lowered for token in markers):
            return False

        evaluation_markers = ["how", "outcome", "result", "impact"]
        if not any(token in lowered for token in evaluation_markers):
            return False

        coding_markers = ["code", "algorithm", "syntax", "implement", "sql", "function"]
        if any(token in lowered for token in coding_markers):
            return False

        words = re.findall(r"\b\w+\b", candidate)
        if len(words) < 12 or len(words) > 45:
            return False

        return candidate.endswith("?")

    def get_questions(self) -> list[str]:
        """Return the fixed HR round question list."""
        return self.questions

    def get_skill_by_index(self, index: int) -> str:
        """Return the HR soft-skill label for a question index."""
        bounded = max(0, min(int(index), len(self.skill_questions) - 1))
        return str(self.skill_questions[bounded]["skill"])

    def run_hr_round(self) -> Dict[str, object]:
        """Run the full non-adaptive HR round and return structured results."""
        per_question_results: List[Dict[str, object]] = []
        all_responses: List[str] = []

        for index, question_row in enumerate(self.skill_questions, start=1):
            question = str(question_row["question"])
            skill = str(question_row["skill"])
            print(f"\nHR Question {index}: {question}")
            answer = input("Your Answer: ").strip()
            all_responses.append(answer)
            evaluation = self.evaluate_response(question=question, answer=answer, skill=skill)

            scores = {
                "leadership": 0.0,
                "problem_solving": 0.0,
                "adaptability": 0.0,
                "teamwork": 0.0,
            }
            if skill in scores:
                scores[skill] = float(evaluation["final_score"])

            per_question_results.append(
                {
                    "skill": skill,
                    "question": question,
                    "answer": answer,
                    "scores": scores,
                    "metrics": evaluation["metrics"],
                    "final_score": evaluation["final_score"],
                    "metric_justifications": evaluation["metric_justifications"],
                    "summary": evaluation["summary"],
                }
            )

        return {
            "questions_asked": len(self.questions),
            "per_question": per_question_results,
            "average_scores": self._compute_averages(per_question_results),
            "responses": all_responses,
        }

    def evaluate_response(self, question: str, answer: str, skill: str = "") -> Dict[str, object]:
        """Evaluate one HR answer using role-focused soft-skill dimensions."""
        prompt = build_hr_evaluation_prompt(question=question, answer=answer, skill=skill)
        response = self.llm_client.generate_response(prompt)
        parsed = safe_json_loads(response)

        metrics = self._extract_metrics(parsed)
        final_score = self._weighted_final_score(metrics)
        metric_justifications = self._extract_metric_justifications(parsed)
        summary = (
            str(parsed.get("summary", "HR evaluation completed.")).strip()
            if isinstance(parsed, dict)
            else "HR evaluation completed."
        )

        return {
            "metrics": metrics,
            "final_score": final_score,
            "metric_justifications": metric_justifications,
            "summary": summary,
        }

    @staticmethod
    def _to_score(value: object, default: float = 0.5) -> float:
        """Convert any numeric value into a bounded soft-skill score."""
        try:
            return clamp(float(value))
        except (TypeError, ValueError):
            return default

    def _extract_metrics(self, parsed: Any) -> Dict[str, float]:
        """Extract and clamp HR rubric metrics from LLM payload."""
        payload = parsed if isinstance(parsed, dict) else {}
        nested_metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
        source = nested_metrics if nested_metrics else payload

        return {
            "relevance": self._to_score(source.get("relevance"), default=0.5),
            "reasoning": self._to_score(source.get("reasoning"), default=0.5),
            "clarity": self._to_score(source.get("clarity"), default=0.5),
            "specificity": self._to_score(source.get("specificity"), default=0.5),
            "outcome": self._to_score(source.get("outcome"), default=0.5),
            "authenticity": self._to_score(source.get("authenticity"), default=0.5),
            "tone": self._to_score(source.get("tone"), default=0.5),
        }

    def _extract_metric_justifications(self, parsed: Any) -> Dict[str, str]:
        """Extract one-line per-metric rationale text when provided by the LLM."""
        payload = parsed if isinstance(parsed, dict) else {}
        nested = (
            payload.get("metric_justifications", {})
            if isinstance(payload.get("metric_justifications"), dict)
            else {}
        )

        result: Dict[str, str] = {}
        for metric in self.METRIC_WEIGHTS:
            text = str(nested.get(metric, "")).strip()
            if text:
                result[metric] = text
        return result

    def _weighted_final_score(self, metrics: Dict[str, float]) -> float:
        """Compute deterministic weighted final score for HR evaluation."""
        total = 0.0
        for metric, weight in self.METRIC_WEIGHTS.items():
            total += float(weight) * self._to_score(metrics.get(metric), default=0.5)
        return clamp(total)

    @staticmethod
    def _compute_averages(per_question_results: List[Dict[str, object]]) -> Dict[str, float]:
        """Compute final per-skill HR scores from one-question-per-skill results."""
        if not per_question_results:
            return {
                "leadership": 0.0,
                "problem_solving": 0.0,
                "adaptability": 0.0,
                "teamwork": 0.0,
            }

        scores = {
            "leadership": 0.0,
            "problem_solving": 0.0,
            "adaptability": 0.0,
            "teamwork": 0.0,
        }

        for row in per_question_results:
            skill = str(row.get("skill", "")).strip()
            final_score = row.get("final_score")
            if skill not in scores:
                continue
            try:
                scores[skill] = round(clamp(float(final_score)), 4)
            except (TypeError, ValueError):
                continue

        return scores
