"""Reusable Groq LLM client with retry and timeout handling."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

import requests

from config import settings


class GroqClient:
    """Class-based wrapper around the Groq Chat Completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        self.timeout = timeout or settings.REQUEST_TIMEOUT_SECONDS
        self.max_retries = max_retries or settings.MAX_RETRIES
        self.base_url = settings.GROQ_API_URL
        self.retry_backoff_seconds = settings.RETRY_BACKOFF_SECONDS
        self.session = requests.Session()
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_response(self, prompt: str) -> str:
        """Generate a text response from the Groq API for a single prompt."""
        if not prompt.strip():
            return ""

        if not self.api_key:
            self.logger.warning("GROQ_API_KEY is missing. Returning empty response.")
            return ""

        payload = self._build_payload(prompt)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                return self._extract_content(data)
            except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
                self.logger.warning(
                    "Groq request failed (attempt %s/%s): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * attempt)

        self.logger.error("Groq request failed after %s attempts.", self.max_retries)
        return ""

    def _build_payload(self, prompt: str) -> Dict[str, Any]:
        """Build API payload for chat completion requests."""
        return {
            "model": self.model,
            "messages": [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": prompt}
],
            "temperature": 0.2,
        }

    @staticmethod
    def _extract_content(response_data: Dict[str, Any]) -> str:
        """Extract generated text from Groq chat completion response."""
        content = response_data["choices"][0]["message"]["content"]
        return str(content).strip()
