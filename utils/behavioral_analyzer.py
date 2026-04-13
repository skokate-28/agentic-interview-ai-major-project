"""Global behavioral analysis across technical and HR interview responses."""

from __future__ import annotations

import re
from typing import Dict, List

from llm.groq_client import GroqClient
from llm.prompt_templates import build_communication_prompt
from utils.helpers import clamp


class BehavioralAnalyzer:
    """Compute stable interview-wide communication and confidence scores."""

    WEAK_PHRASES = [
        "maybe",
        "i think",
        "probably",
        "not sure",
        "i guess",
        "kind of",
        "sort of",
        "possibly",
    ]

    def __init__(self, llm_client: GroqClient | None = None) -> None:
        self.llm_client = llm_client or GroqClient()

    def analyze(self, responses: List[str]) -> Dict[str, float | int]:
        """Analyze all candidate responses for global confidence and communication."""
        cleaned_responses = [response.strip() for response in responses if response.strip()]
        total_words = self._count_words(cleaned_responses)
        weak_count = self._count_weak_phrases(cleaned_responses)
        weak_ratio = weak_count / max(1, total_words)

        confidence_score = clamp(1 - min(0.4, weak_ratio * 5))
        communication_score = self._communication_score(cleaned_responses, total_words)

        return {
            "confidence_score": round(confidence_score, 4),
            "communication_score": round(communication_score, 4),
            "weak_word_ratio": round(weak_ratio, 4),
            "total_responses": len(cleaned_responses),
        }

    def _communication_score(self, responses: List[str], total_words: int) -> float:
        """Blend LLM score with deterministic style and grammar heuristics."""
        llm_score = self._llm_communication_score(responses)
        total_sentences = self._count_sentences(responses)
        avg_sentence_length = total_words / max(1, total_sentences)

        length_score = self._sentence_length_score(avg_sentence_length)
        grammar_score = self._grammar_score(responses, total_sentences)

        return clamp((0.6 * llm_score) + (0.2 * length_score) + (0.2 * grammar_score))

    def _llm_communication_score(self, responses: List[str]) -> float:
        """Get communication quality estimate from the LLM, with safe fallback."""
        prompt = build_communication_prompt(responses)
        output = self.llm_client.generate_response(prompt)
        score = self._extract_float(output)
        if score is None:
            return 0.6
        return clamp(score)

    @staticmethod
    def _extract_float(text: str) -> float | None:
        """Extract the first numeric value from free-form text."""
        match = re.search(r"\d*\.?\d+", text.strip())
        if match is None:
            return None

        try:
            return float(match.group(0))
        except ValueError:
            return None

    @staticmethod
    def _count_words(responses: List[str]) -> int:
        """Count all word tokens across response list."""
        return sum(len(re.findall(r"\b\w+\b", response.lower())) for response in responses)

    def _count_weak_phrases(self, responses: List[str]) -> int:
        """Count weak language phrase occurrences across all responses."""
        joined = "\n".join(responses).lower()
        weak_count = 0
        for phrase in self.WEAK_PHRASES:
            weak_count += joined.count(phrase)
        return weak_count

    @staticmethod
    def _count_sentences(responses: List[str]) -> int:
        """Estimate sentence count using punctuation-based splitting."""
        sentences = 0
        for response in responses:
            parts = [part.strip() for part in re.split(r"[.!?]+", response) if part.strip()]
            sentences += len(parts)
        return max(1, sentences)

    @staticmethod
    def _sentence_length_score(avg_sentence_length: float) -> float:
        """Score sentence length while penalizing extremes smoothly."""
        if avg_sentence_length < 4:
            return 0.35
        if avg_sentence_length < 8:
            return 0.55 + ((avg_sentence_length - 4) / 4.0) * 0.25
        if avg_sentence_length <= 24:
            return 1.0 - min(0.2, abs(avg_sentence_length - 16) * 0.02)
        if avg_sentence_length <= 35:
            return max(0.55, 0.84 - ((avg_sentence_length - 24) * 0.02))
        return 0.5

    @staticmethod
    def _grammar_score(responses: List[str], total_sentences: int) -> float:
        """Compute lightweight grammar/structure heuristic from punctuation and sentence form."""
        joined = "\n".join(responses)
        punctuation_marks = len(re.findall(r"[.,;:!?]", joined))
        punctuation_score = min(1.0, punctuation_marks / max(1, total_sentences))

        response_count = max(1, len(responses))
        sentence_presence_score = min(1.0, total_sentences / response_count)

        capitalized_count = 0
        for response in responses:
            stripped = response.strip()
            if stripped and stripped[0].isalpha() and stripped[0].isupper():
                capitalized_count += 1
        capitalization_score = capitalized_count / response_count

        return clamp(
            (0.5 * punctuation_score)
            + (0.3 * sentence_presence_score)
            + (0.2 * capitalization_score)
        )
