"""Resume parser agent that extracts structured profile data using LLM."""

from __future__ import annotations

import logging
import re
from typing import Dict, List

from llm.groq_client import GroqClient
from llm.prompt_templates import RESUME_PARSE_PROMPT
from utils.helpers import normalize_resume_payload, safe_json_loads


class ResumeParserAgent:
    """Parse raw resume text into structured fields."""

    def __init__(self, llm_client: GroqClient | None = None) -> None:
        self.llm_client = llm_client or GroqClient()
        self.logger = logging.getLogger(self.__class__.__name__)

    def parse_resume(self, raw_resume_text: str) -> Dict[str, list[str]]:
        """Extract structured resume data using LLM, then fallback heuristics."""
        prompt = RESUME_PARSE_PROMPT.format(resume_text=raw_resume_text)
        response = self.llm_client.generate_response(prompt)
        parsed = safe_json_loads(response)
        normalized = normalize_resume_payload(parsed if isinstance(parsed, dict) else None)

        if any(normalized.values()):
            return normalized

        self.logger.info("Using fallback resume parser due to empty/invalid LLM output.")
        return self._fallback_parse(raw_resume_text)

    def _fallback_parse(self, text: str) -> Dict[str, list[str]]:
        """Fallback parser for basic sectioned resume formats."""
        result: Dict[str, list[str]] = {
            "skills": [],
            "projects": [],
            "experience": [],
            "achievements": [],
        }
        current_section: str | None = None

        for raw_line in text.splitlines():
            line = self._clean_line(raw_line)
            if not line:
                continue

            section = self._detect_section(line)
            if section:
                current_section = section
                inline_content = line.split(":", maxsplit=1)
                if len(inline_content) == 2 and inline_content[1].strip():
                    result[section].extend(self._split_items(inline_content[1]))
                continue

            if current_section:
                result[current_section].append(line)

        result["skills"] = self._deduplicate(result["skills"])
        result["projects"] = self._deduplicate(result["projects"])
        result["experience"] = self._deduplicate(result["experience"])
        result["achievements"] = self._deduplicate(result["achievements"])
        return result

    @staticmethod
    def _clean_line(line: str) -> str:
        """Normalize bullets/numbering and trim whitespace."""
        cleaned = line.strip()
        cleaned = re.sub(r"^[-*]\s+", "", cleaned)
        cleaned = re.sub(r"^\d+[\).]\s+", "", cleaned)
        return cleaned.strip()

    @staticmethod
    def _detect_section(line: str) -> str | None:
        """Map a heading line to a known resume section key."""
        lower = line.lower()
        if lower.startswith("skills:") or lower == "skills":
            return "skills"
        if lower.startswith("projects:") or lower == "projects":
            return "projects"
        if lower.startswith("experience:") or lower == "experience":
            return "experience"
        if lower.startswith("achievements:") or lower == "achievements":
            return "achievements"
        return None

    @staticmethod
    def _split_items(value: str) -> List[str]:
        """Split comma-separated values into normalized list items."""
        parts = [item.strip() for item in value.split(",")]
        return [item for item in parts if item]

    @staticmethod
    def _deduplicate(items: List[str]) -> List[str]:
        """Deduplicate while preserving original item order."""
        seen: set[str] = set()
        output: List[str] = []
        for item in items:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            output.append(item)
        return output
