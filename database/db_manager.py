"""SQLite manager for storing candidate and skill progress data."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from config import settings
from database.models import CandidateProfile


class DatabaseManager:
    """Database access layer for foundational interview system persistence."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or settings.DATABASE_PATH
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        """Create a SQLite connection."""
        return sqlite3.connect(self.db_path)

    def _initialize_schema(self) -> None:
        """Create required database tables if they do not exist."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS candidate_profiles (
                    candidate_id TEXT PRIMARY KEY,
                    resume_data TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT NOT NULL,
                    skill_name TEXT NOT NULL,
                    proficiency REAL NOT NULL,
                    confidence REAL NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_weak_areas (
                    candidate_id TEXT NOT NULL,
                    skill_name TEXT NOT NULL,
                    weak_areas TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (candidate_id, skill_name)
                )
                """
            )
            conn.commit()

    def upsert_candidate(self, candidate_id: str, resume_data: dict[str, Any]) -> None:
        """Insert or update candidate profile data."""
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO candidate_profiles (candidate_id, resume_data, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(candidate_id)
                DO UPDATE SET resume_data = excluded.resume_data
                """,
                (candidate_id, json.dumps(resume_data), now),
            )
            conn.commit()

    def get_candidate(self, candidate_id: str) -> CandidateProfile | None:
        """Fetch a candidate profile by candidate ID."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT candidate_id, resume_data, created_at
                FROM candidate_profiles
                WHERE candidate_id = ?
                """,
                (candidate_id,),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return CandidateProfile(
            candidate_id=row[0],
            resume_data=json.loads(row[1]),
            created_at=row[2],
        )

    def save_skill_progress(
        self,
        candidate_id: str,
        skill_name: str,
        proficiency: float,
        confidence: float,
    ) -> None:
        """Persist a skill proficiency snapshot for historical tracking."""
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO skill_progress (
                    candidate_id,
                    skill_name,
                    proficiency,
                    confidence,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (candidate_id, skill_name, proficiency, confidence, now),
            )
            conn.commit()

    def get_skill_weak_areas(
        self,
        candidate_id: str,
        skill_name: str,
    ) -> dict[str, dict[str, float | int]]:
        """Fetch weak-area metadata for a user-skill pair from DB source of truth."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT weak_areas
                FROM skill_weak_areas
                WHERE candidate_id = ? AND skill_name = ?
                """,
                (candidate_id, skill_name),
            )
            row = cursor.fetchone()

        if row is None:
            return {}

        try:
            parsed = json.loads(row[0])
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}

        if not isinstance(parsed, dict):
            return {}

        return parsed

    def save_skill_weak_areas(
        self,
        candidate_id: str,
        skill_name: str,
        weak_areas: dict[str, dict[str, float | int]],
    ) -> None:
        """Persist weak-area metadata for a user-skill pair."""
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO skill_weak_areas (
                    candidate_id,
                    skill_name,
                    weak_areas,
                    updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(candidate_id, skill_name)
                DO UPDATE SET
                    weak_areas = excluded.weak_areas,
                    updated_at = excluded.updated_at
                """,
                (candidate_id, skill_name, json.dumps(weak_areas), now),
            )
            conn.commit()
