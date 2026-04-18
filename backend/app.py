"""FastAPI backend for interview session APIs."""

from __future__ import annotations

import re

import pymupdf as fitz  # PyMuPDF
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.session_service import InterviewSessionService


class AnswerRequest(BaseModel):
    """Payload for submitting one answer."""

    session_id: str = Field(min_length=1)
    answer: str


class NextQuestionRequest(BaseModel):
    """Payload for requesting the pending next question."""

    session_id: str = Field(min_length=1)


app = FastAPI(title="AI Interview Backend", version="1.0.0")
service = InterviewSessionService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Health-check endpoint."""
    return {"status": "ok"}


def _extract_skills(text: str) -> list[str]:
    """Extract skills directly from the resume skills section."""
    if not text.strip():
        return []

    skills: list[str] = []
    lines = text.splitlines()

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        if "skills" in stripped.lower():
            parts = stripped.split(":", maxsplit=1)
            if len(parts) == 2:
                for token in re.split(r"[,/|;]", parts[1]):
                    cleaned = token.strip()
                    if cleaned:
                        skills.append(cleaned)
            else:
                for j in range(i + 1, min(i + 6, len(lines))):
                    next_line = lines[j].strip()
                    if not next_line:
                        break

                    for token in re.split(r"[,/|;]", next_line):
                        cleaned = token.strip()
                        if cleaned:
                            skills.append(cleaned)
            break

    seen: set[str] = set()
    final_skills: list[str] = []
    for skill in skills:
        key = skill.lower()
        if key not in seen:
            seen.add(key)
            final_skills.append(skill)

    print("FINAL SKILLS FROM RESUME:", final_skills)
    return final_skills[:8]


@app.post("/start")
@app.post("/start_interview")
async def start_interview(
    name: str = Form(...),
    user_id: str = Form(...),
    mode: str = Form(...),
    resume: UploadFile = File(...),
) -> dict[str, object]:
    """Non-blocking start endpoint that returns immediate first question."""
    print("START called - fast response")

    cleaned_name = name.strip()
    cleaned_user_id = user_id.strip()
    cleaned_mode = mode.strip().lower()

    if not cleaned_name:
        raise HTTPException(status_code=400, detail="Name is required")
    if not cleaned_user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    if cleaned_mode not in {"text", "audio"}:
        raise HTTPException(status_code=400, detail="Mode must be 'text' or 'audio'")

    if not resume:
        raise HTTPException(status_code=400, detail="Resume file is required")

    filename = (resume.filename or "").strip()
    print("Resume received:", filename)
    lower_name = filename.lower()
    if not lower_name.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF or TXT allowed")

    resume_bytes = b""
    try:
        resume_bytes = await resume.read()
    except Exception as exc:
        print("Resume read error:", exc)

    text = ""
    try:
        if filename.lower().endswith(".pdf"):
            doc = fitz.open(stream=resume_bytes, filetype="pdf")
            for page in doc:
                text += page.get_text()
        else:
            text = resume_bytes.decode("utf-8", errors="ignore")
    except Exception as exc:
        print("Resume parsing failed:", exc)
        text = ""

    print("Extracted text preview:", text[:500])

    skills = _extract_skills(text)
    if not skills:
        skills = ["Python"]

    print("DEBUG -> Extracted Skills:", skills)
    print("Parsed skills:", skills)

    start_result = service.start_session(
        user_id=cleaned_user_id,
        name=cleaned_name,
        mode=cleaned_mode,
        skills=skills,
        first_question=None,
    )

    session_id = str(start_result.get("session_id", ""))
    print("Created session_id:", session_id)
    if session_id:
        service.attach_resume(session_id=session_id, resume_bytes=resume_bytes, filename=filename)

    print("Sessions stored:", list(service.sessions.keys()))
    return {
        **start_result,
        "skills_debug": skills,
    }


@app.post("/answer")
async def submit_answer(request: AnswerRequest) -> dict:
    """Process answer and return next question or final report."""
    session_id = request.session_id.strip()
    answer = str(request.answer)

    print("Received session_id:", session_id)
    print("Received answer:", answer[:50])

    result = service.process_answer(
        session_id=session_id,
        answer=answer,
    )

    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message", "Invalid request"))

    return result


@app.post("/next_question")
async def next_question(request: NextQuestionRequest) -> dict:
    """Return next question after evaluation step."""
    session_id = request.session_id.strip()
    result = service.get_next_question(session_id=session_id)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message", "Invalid request"))
    return result


@app.get("/dashboard/{user_id}")
def get_dashboard(user_id: str) -> dict:
    """Return user dashboard history payload."""
    try:
        return service.get_dashboard(user_id.strip())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load dashboard: {exc}") from exc
