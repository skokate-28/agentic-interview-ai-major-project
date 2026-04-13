"""Application settings and constants loaded from environment variables."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_URL: str = os.getenv(
    "GROQ_API_URL",
    "https://api.groq.com/openai/v1/chat/completions",
)

MAX_QUESTIONS_PER_SKILL: int = int(os.getenv("MAX_QUESTIONS_PER_SKILL", "5"))
BKT_ALPHA: float = float(os.getenv("BKT_ALPHA", "0.2"))
CONVERGENCE_THRESHOLD: float = float(os.getenv("CONVERGENCE_THRESHOLD", "0.85"))

REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BACKOFF_SECONDS: float = float(os.getenv("RETRY_BACKOFF_SECONDS", "1"))

DATABASE_PATH: str = os.getenv("DATABASE_PATH", "interview_ai.db")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")