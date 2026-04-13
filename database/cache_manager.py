"""Redis cache layer for user-skill weak-area metadata."""

from __future__ import annotations

import importlib
import json
from typing import Any


class CacheManager:
    """Cache manager using Redis for weak-area metadata lookups."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        socket_timeout: int = 2,
    ) -> None:
        self.client: Any | None = None
        self._redis_module: Any | None = None

        try:
            self._redis_module = importlib.import_module("redis")
            self.client = self._redis_module.Redis(
                host=host,
                port=port,
                db=db,
                socket_timeout=socket_timeout,
                decode_responses=True,
            )
            self.client.ping()
        except Exception:
            self.client = None
            self._redis_module = None

    @staticmethod
    def _key(user_id: str, skill: str) -> str:
        """Generate user+skill namespaced cache key."""
        return f"candidate:{user_id}:skill:{skill}:weak_areas"

    def get_weak_areas(self, user_id: str, skill: str) -> dict[str, dict[str, float | int]] | None:
        """Get weak-area metadata from Redis cache by user and skill."""
        if self.client is None:
            return None

        try:
            raw = self.client.get(self._key(user_id, skill))
        except Exception:
            return None

        if raw is None:
            return None

        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

        if not isinstance(parsed, dict):
            return None

        return parsed

    def set_weak_areas(self, user_id: str, skill: str, data: dict[str, dict[str, float | int]]) -> None:
        """Set weak-area metadata in Redis cache by user and skill."""
        if self.client is None:
            return

        try:
            self.client.set(self._key(user_id, skill), json.dumps(data))
        except Exception:
            return
