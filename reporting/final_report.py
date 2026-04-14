"""Final report generation utilities for technical and HR interview sessions."""

from __future__ import annotations

from typing import Any, Dict

from llm.groq_client import GroqClient


def _avg(values: list[float]) -> float:
    """Return average of numeric values with safe zero fallback."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def generate_summary(skill_scores: Dict[str, float], weak_areas: list[str]) -> str:
    """Generate compact technical summary using LLM with deterministic fallback."""
    strengths = sorted(skill_scores.items(), key=lambda item: item[1], reverse=True)
    strength_text = ", ".join([f"{name}: {score:.2f}" for name, score in strengths[:2]]) or "None"
    weakness_text = ", ".join(weak_areas[:2]) if weak_areas else "None"

    prompt = f"""
Generate a 3-4 line technical interview summary.
Mention strengths and 1-2 improvement areas.
Avoid fluff.

Skill Scores: {skill_scores}
Top Strengths: {strength_text}
Weak Areas: {weakness_text}
""".strip()

    llm = GroqClient()
    response = llm.generate_response(prompt)
    if response.strip():
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        return "\n".join(lines[:4])

    if strengths:
        top_strengths = ", ".join([name for name, _ in strengths[:2]])
    else:
        top_strengths = "fundamental understanding"

    if weak_areas:
        improvement = ", ".join(weak_areas[:2])
    else:
        improvement = "deeper practice on advanced scenarios"

    return (
        f"Strong performance in {top_strengths}.\n"
        f"Overall consistency is improving across assessed skills.\n"
        f"Focus next on {improvement} for better depth and reliability."
    )


def generate_final_report(user_data: Dict[str, Any], session_data: Dict[str, Any]) -> Dict[str, Any]:
    """Build final structured report including technical, HR, and trend graphs."""
    technical = session_data.get("technical", {})
    hr = session_data.get("hr", {})

    skill_scores_raw = technical.get("skill_scores", {})
    skill_scores = {
        str(skill): float(score)
        for skill, score in skill_scores_raw.items()
        if isinstance(score, (int, float))
    }
    session_weak_areas = [str(item) for item in technical.get("session_weak_areas", [])]

    technical_overall = _avg(list(skill_scores.values()))
    summary = generate_summary(skill_scores, session_weak_areas)

    technical_section = {
        "overall_score": round(technical_overall, 4),
        "skill_scores": {k: round(v, 4) for k, v in skill_scores.items()},
        "session_weak_areas": session_weak_areas,
        "summary": summary,
    }

    hr_metrics = {
        "communication": float(hr.get("communication", 0.0)),
        "confidence": float(hr.get("confidence", 0.0)),
        "leadership": float(hr.get("leadership", 0.0)),
        "problem_solving": float(hr.get("problem_solving", 0.0)),
    }
    hr_overall = _avg(list(hr_metrics.values()))
    hr_section = {
        "overall_score": round(hr_overall, 4),
        **{key: round(value, 4) for key, value in hr_metrics.items()},
    }

    existing_sessions = user_data.get("sessions", [])
    if not isinstance(existing_sessions, list):
        existing_sessions = []

    timeline_sessions = list(existing_sessions)
    session_id = str(session_data.get("session_id", ""))
    has_current_session = any(
        isinstance(item, dict) and str(item.get("session_id", "")) == session_id
        for item in existing_sessions
    )
    if not has_current_session:
        timeline_sessions.append(session_data)

    overall_history: list[dict[str, Any]] = []
    skill_histories: dict[str, list[dict[str, Any]]] = {}
    hr_histories: dict[str, list[dict[str, Any]]] = {
        "overall_score": [],
        "communication": [],
        "confidence": [],
        "leadership": [],
        "problem_solving": [],
    }

    for session in timeline_sessions:
        if not isinstance(session, dict):
            continue

        session_id = str(session.get("session_id", ""))
        timestamp = session.get("timestamp")

        tech = session.get("technical", {})
        if isinstance(tech, dict):
            session_overall = float(tech.get("overall_score", 0.0))
            overall_history.append(
                {
                    "session_id": session_id,
                    "timestamp": timestamp,
                    "score": round(session_overall, 4),
                }
            )

            session_skill_scores = tech.get("skill_scores", {})
            if isinstance(session_skill_scores, dict):
                for skill_name, score in session_skill_scores.items():
                    if not isinstance(score, (int, float)):
                        continue
                    skill_histories.setdefault(str(skill_name), []).append(
                        {
                            "session_id": session_id,
                            "timestamp": timestamp,
                            "score": round(float(score), 4),
                        }
                    )

        hr_block = session.get("hr", {})
        if isinstance(hr_block, dict):
            for key in hr_histories:
                value = hr_block.get(key)
                if isinstance(value, (int, float)):
                    hr_histories[key].append(
                        {
                            "session_id": session_id,
                            "timestamp": timestamp,
                            "score": round(float(value), 4),
                        }
                    )

    return {
        "technical": technical_section,
        "hr": hr_section,
        "graphs": {
            "overall_history": overall_history,
            "skill_histories": skill_histories,
            "hr_histories": hr_histories,
        },
    }
