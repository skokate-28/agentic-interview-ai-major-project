"""Global behavioral analysis across technical and HR interview responses."""

from __future__ import annotations

import re
from statistics import pstdev
from typing import Any, Dict, List

from llm.groq_client import GroqClient
from llm.prompt_templates import build_behavioral_prompt
from utils.helpers import clamp, safe_json_loads


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

    def analyze(self, responses: List[Any]) -> Dict[str, float | str]:
        """Analyze full interview transcript for global communication and confidence."""
        normalized_responses = self._normalize_responses(responses)
        full_transcript = " ".join([row["answer"] for row in normalized_responses])
        cleaned_answers = [row["answer"] for row in normalized_responses if row["answer"]]

        total_words = self._count_words(cleaned_answers)
        weak_count = self._count_weak_phrases(cleaned_answers)
        weak_ratio = weak_count / max(1, total_words)

        llm_eval = self._llm_behavioral_eval(full_transcript)
        llm_communication = self._to_optional_score(llm_eval.get("communication"))
        llm_confidence = self._to_optional_score(llm_eval.get("confidence"))

        communication_heuristic = self._communication_score(cleaned_answers, total_words)
        confidence_heuristic = self._confidence_score(cleaned_answers, weak_ratio)

        if llm_communication is None:
            communication_score = communication_heuristic
        else:
            communication_score = clamp((0.65 * llm_communication) + (0.35 * communication_heuristic))

        if llm_confidence is None:
            confidence_score = confidence_heuristic
        else:
            confidence_score = clamp((0.25 * llm_confidence) + (0.75 * confidence_heuristic))

        summary = self._build_summary(
            llm_summary=str(llm_eval.get("behavioral_summary", "")).strip(),
            communication=communication_score,
            confidence=confidence_score,
            weak_ratio=weak_ratio,
        )

        return {
            "communication": round(communication_score, 4),
            "confidence": round(confidence_score, 4),
            "behavioral_summary": summary,
        }

    @staticmethod
    def _normalize_responses(responses: List[Any]) -> List[Dict[str, str]]:
        """Normalize response payloads to a unified list[{'answer': ...}] shape."""
        normalized: List[Dict[str, str]] = []
        for item in responses:
            if isinstance(item, dict):
                answer = str(item.get("answer", "")).strip()
                phase = str(item.get("phase", "")).strip() or "unknown"
            else:
                answer = str(item).strip()
                phase = "unknown"

            if answer:
                normalized.append({"phase": phase, "answer": answer})
        return normalized

    def _communication_score(self, responses: List[str], total_words: int) -> float:
        """Blend LLM score with deterministic style and grammar heuristics."""
        total_sentences = self._count_sentences(responses)
        avg_sentence_length = total_words / max(1, total_sentences)

        length_score = self._sentence_length_score(avg_sentence_length)
        grammar_score = self._grammar_score(responses, total_sentences)
        coherence_score = self._coherence_score(responses)
        conciseness_score = self._conciseness_score(responses)

        return clamp(
            (0.35 * length_score)
            + (0.25 * grammar_score)
            + (0.2 * coherence_score)
            + (0.2 * conciseness_score)
        )

    def _llm_behavioral_eval(self, full_transcript: str) -> Dict[str, Any]:
        """Get JSON behavioral assessment from the LLM with safe fallback."""
        prompt = build_behavioral_prompt(full_transcript)
        output = self.llm_client.generate_response(prompt)
        parsed = safe_json_loads(output)
        if isinstance(parsed, dict):
            return parsed
        return {}

    @staticmethod
    def _to_optional_score(value: Any) -> float | None:
        """Convert optional value to bounded score; return None on parse failure."""
        try:
            return clamp(float(value))
        except (TypeError, ValueError):
            return None

    def _confidence_score(self, responses: List[str], weak_ratio: float) -> float:
        """Compute confidence using normalized weak language, assertiveness, and consistency."""
        weak_language_score = clamp(1.0 - min(0.6, weak_ratio * 8.0))
        assertiveness_score = self._assertiveness_score(responses)
        consistency_score = self._consistency_score(responses)

        return clamp(
            (0.5 * weak_language_score)
            + (0.25 * assertiveness_score)
            + (0.25 * consistency_score)
        )

    @staticmethod
    def _assertiveness_score(responses: List[str]) -> float:
        """Estimate assertiveness from hedging-vs-direct language ratio."""
        joined = " ".join(responses).lower()
        if not joined.strip():
            return 0.5

        hedging_markers = [
            "maybe",
            "i think",
            "probably",
            "not sure",
            "i guess",
            "kind of",
            "sort of",
            "possibly",
        ]
        direct_markers = ["i can", "i will", "i solved", "i led", "i built", "i delivered"]

        hedge_count = sum(joined.count(marker) for marker in hedging_markers)
        direct_count = sum(joined.count(marker) for marker in direct_markers)

        raw = 0.55 + (0.08 * direct_count) - (0.08 * hedge_count)
        return clamp(raw)

    def _consistency_score(self, responses: List[str]) -> float:
        """Estimate confidence consistency from per-response weak ratio variance."""
        ratios: List[float] = []
        for response in responses:
            words = re.findall(r"\b\w+\b", response.lower())
            if not words:
                continue
            weak_hits = 0
            lower = response.lower()
            for phrase in self.WEAK_PHRASES:
                weak_hits += lower.count(phrase)
            ratios.append(weak_hits / max(1, len(words)))

        if len(ratios) <= 1:
            return 0.65

        variation = pstdev(ratios)
        return clamp(1.0 - min(0.5, variation * 8.0))

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
    def _coherence_score(responses: List[str]) -> float:
        """Approximate coherence from keyword continuity between adjacent answers."""
        if len(responses) <= 1:
            return 0.7

        overlaps: List[float] = []
        for index in range(1, len(responses)):
            prev_tokens = set(re.findall(r"\b[a-z]{4,}\b", responses[index - 1].lower()))
            curr_tokens = set(re.findall(r"\b[a-z]{4,}\b", responses[index].lower()))
            if not prev_tokens or not curr_tokens:
                continue
            union = prev_tokens | curr_tokens
            if not union:
                continue
            overlaps.append(len(prev_tokens & curr_tokens) / len(union))

        if not overlaps:
            return 0.65
        return clamp(0.5 + min(0.4, sum(overlaps) / len(overlaps)))

    @staticmethod
    def _conciseness_score(responses: List[str]) -> float:
        """Score conciseness from average words per response using a soft target range."""
        if not responses:
            return 0.5

        lengths = [len(re.findall(r"\b\w+\b", response)) for response in responses]
        avg_words = sum(lengths) / max(1, len(lengths))

        if avg_words < 8:
            return 0.45
        if avg_words <= 45:
            return 0.95
        if avg_words <= 80:
            return max(0.6, 0.95 - ((avg_words - 45) * 0.01))
        return 0.55

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

    @staticmethod
    def _build_summary(
        llm_summary: str,
        communication: float,
        confidence: float,
        weak_ratio: float,
    ) -> str:
        """Build stable behavioral summary with LLM text fallback."""
        if llm_summary:
            return llm_summary

        communication_label = "clear" if communication >= 0.7 else "developing"
        confidence_label = "steady" if confidence >= 0.7 else "moderate"
        return (
            "Communication is "
            f"{communication_label}, while confidence appears {confidence_label}. "
            f"Weak-language ratio is {weak_ratio:.3f} across the full interview transcript."
        )
