"""Data models for persistence and transfer across layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class CandidateProfile:
    """Candidate record persisted across interview sessions."""

    candidate_id: str
    resume_data: Dict[str, Any]
    created_at: str | None = None


@dataclass
class SkillProgress:
    """Per-skill proficiency snapshot for a candidate."""

    candidate_id: str
    skill_name: str
    proficiency: float
    confidence: float
    updated_at: str | None = None
