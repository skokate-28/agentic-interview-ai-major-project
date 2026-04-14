"""MongoDB client for persistent multi-session interview storage."""

from __future__ import annotations

import os
from typing import Any, Dict

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()


class MongoDBManager:
    """MongoDB access layer for users, skills, and session history."""

    def __init__(self, mongo_uri: str | None = None) -> None:
        uri = mongo_uri or os.getenv("MONGO_URI")
        if not uri:
            raise ValueError("MONGO_URI not found in environment variables")

        self.client = MongoClient(uri)
        self.db = self.client["interview_db"]
        self.users_collection = self.db["users"]

    def get_user(self, user_id: str) -> Dict[str, Any] | None:
        """Fetch one user document by user_id."""
        return self.users_collection.find_one({"user_id": user_id}, {"_id": 0})

    def create_user(self, user_id: str, name: str) -> Dict[str, Any]:
        """Create a new user document if missing, otherwise return existing."""
        self.users_collection.update_one(
            {"user_id": user_id},
            {
                "$setOnInsert": {
                    "user_id": user_id,
                    "name": name,
                    "skills": {},
                    "sessions": [],
                }
            },
            upsert=True,
        )
        return self.get_user(user_id) or {
            "user_id": user_id,
            "name": name,
            "skills": {},
            "sessions": [],
        }

    def update_skill(self, user_id: str, skill_data: Dict[str, Dict[str, Any]]) -> None:
        """Merge per-skill proficiency, confidence, weak areas, and optional history."""
        user = self.get_user(user_id)
        if user is None:
            user = self.create_user(user_id, "Candidate")

        skills = user.get("skills", {})
        if not isinstance(skills, dict):
            skills = {}

        for skill_name, incoming in skill_data.items():
            existing = skills.get(skill_name, {})
            if not isinstance(existing, dict):
                existing = {}

            history = existing.get("history", [])
            if not isinstance(history, list):
                history = []

            session_id = incoming.get("session_id")
            score = incoming.get("score")
            if session_id is not None and score is not None:
                has_session = any(
                    isinstance(entry, dict) and entry.get("session_id") == session_id
                    for entry in history
                )
                if not has_session:
                    history.append({"session_id": str(session_id), "score": float(score)})

            skills[skill_name] = {
                "proficiency": float(incoming.get("proficiency", existing.get("proficiency", 0.0))),
                "confidence": float(incoming.get("confidence", existing.get("confidence", 0.0))),
                "weak_areas": incoming.get("weak_areas", existing.get("weak_areas", {})),
                "history": history,
            }

        self.users_collection.update_one(
            {"user_id": user_id},
            {"$set": {"skills": skills}},
            upsert=True,
        )

    def append_session(self, user_id: str, session_data: Dict[str, Any]) -> None:
        """Append one interview session object to user session history."""
        self.users_collection.update_one(
            {"user_id": user_id},
            {"$push": {"sessions": session_data}},
            upsert=True,
        )
