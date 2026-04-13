"""Utility helpers for normalization and lightweight parsing."""

from __future__ import annotations

import json
import re
from typing import Any, Dict


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a float value to an inclusive range."""
    return max(low, min(high, value))


def extract_json_block(text: str) -> str:
    """Extract the first JSON object block from arbitrary model output."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else ""


def safe_json_loads(text: str) -> Any | None:
    """Safely parse JSON text; return None on parse failures."""
    candidate = extract_json_block(text) if text else ""
    if not candidate:
        return None

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def normalize_resume_payload(payload: Dict[str, Any] | None) -> Dict[str, list[str]]:
    """Normalize resume JSON payload to required schema with list[str] fields."""
    default_payload: Dict[str, list[str]] = {
        "skills": [],
        "projects": [],
        "experience": [],
        "achievements": [],
    }
    if not isinstance(payload, dict):
        return default_payload

    normalized: Dict[str, list[str]] = {}
    for key in default_payload:
        value = payload.get(key, [])
        if isinstance(value, list):
            normalized[key] = [str(item).strip() for item in value if str(item).strip()]
        elif isinstance(value, str) and value.strip():
            normalized[key] = [value.strip()]
        else:
            normalized[key] = []

    return normalized
