"""API-safe interview session service built on top of existing interview components."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import random
from threading import Lock
from typing import Any, Dict
from uuid import uuid4

from controller.langgraph_flow import LangGraphInterviewFlow
from core.bkt_engine import BKTModel
from core.cat_engine import CATEngine
from core.skill_graph import SkillGraph
from database.cache_manager import CacheManager
from database.mongo_client import MongoDBManager
from reporting.final_report import generate_final_report, generate_summary

try:
    from core.behavioral_analyzer import BehavioralAnalyzer
except ImportError:  # Backward-compatible import path.
    from utils.behavioral_analyzer import BehavioralAnalyzer

DEFAULT_SKILLS = ["Python", "SQL"]
MAX_QUESTIONS_PER_SKILL = 7
MAX_HR_QUESTIONS = 4
BASELINE_PROBABILITY = 0.5
BASELINE_DIFFICULTY = 1
MIN_QUESTIONS = 3
WIDTH_THRESHOLD = 0.35
QUESTION_TYPES_BY_LEVEL: dict[str, list[str]] = {
    "easy": ["definition", "basic concept", "simple example"],
    "easy-medium": ["concept explanation", "comparison", "use-case"],
    "medium": ["application", "scenario", "how/why"],
    "hard": ["optimization", "trade-offs", "edge cases"],
    "extreme-hard": ["system design", "failure analysis", "deep debugging"],
}


def compute_width(p: float, n: int, score: float) -> float:
    """Compute adaptive width with count-dominant decay and moderate score influence."""
    _ = float(p)  # Keep signature stable; width intentionally depends on count and score only.
    attempts = max(int(n), 0)
    bounded_score = max(0.0, min(1.0, float(score)))

    # Base decay is the dominant factor and changes smoothly with question count.
    base = 1.0 / math.sqrt(attempts + 1)

    # Extremeness is 0 near 0.5 and 1 at 0 or 1.
    extremeness = 4.0 * (bounded_score - 0.5) ** 2

    # Score impact is intentionally limited to a maximum 25% adjustment.
    score_factor = 1.0 - (0.25 * extremeness)

    width = base * score_factor

    # Prevent uncertainty from collapsing too early.
    return max(0.05, width)


class InterviewSessionService:
    """Manages active interview sessions for API clients."""

    def __init__(
        self,
        mongo_manager: MongoDBManager | None = None,
        cache_manager: CacheManager | None = None,
    ) -> None:
        try:
            self.mongo_manager = mongo_manager or MongoDBManager()
        except Exception as e:
            print("MongoDB init failed, running without DB:", e)
            self.mongo_manager = None
        self.cache_manager = cache_manager or CacheManager()
        self.sessions: dict[str, dict[str, Any]] = {}
        self.lock = Lock()

    def start_session(
        self,
        user_id: str,
        name: str,
        mode: str,
        skills: list[str] | None = None,
        first_question: str | None = None,
    ) -> dict[str, Any]:
        """Create lightweight session and return immediate first question."""
        with self.lock:
            # Keep start independent from MongoDB latency.
            normalized_skills = [str(skill).strip() for skill in (skills or []) if str(skill).strip()]
            if not normalized_skills:
                normalized_skills = list(DEFAULT_SKILLS)

            print("Parsed skills:", normalized_skills)
            current_skill = normalized_skills[0]
            initial_question = (
                str(first_question).strip()
                if first_question is not None and str(first_question).strip()
                else ""
            )
            print("MongoDB bypass enabled for start_session; using skills:", normalized_skills)

            session_id = str(uuid4())

            self.sessions[session_id] = {
                "session_id": session_id,
                "user_id": user_id,
                "name": name,
                "mode": mode,
                "modality": "oral",
                "flow": None,
                "skills": normalized_skills,
                "skill_index": 0,
                "skill_states": {},
                "technical_results": {},
                "hr_results": None,
                "all_responses": [],
                "session_weak_areas": set(),
                "phase": "technical",
                "current_question": initial_question,
                "current_skill": current_skill,
                "hr_index": 0,
                "hr_per_question": [],
                "runtime_ready": False,
                "resume_bytes": b"",
                "resume_filename": "",
                "asked_questions": {initial_question},
                "question_count": 0,
                "technical_scores": [],
                "hr_scores": [],
                "skill_graph": {},
                "awaiting_next_question": False,
                "pending_next_question": "",
                "pending_next_phase": "",
                "pending_next_skill": "",
                "pending_next_question_type": "",
                "last_evaluation": None,
            }

            self._initialize_runtime(self.sessions[session_id])
            initial_question = str(self.sessions[session_id].get("current_question", initial_question)).strip()

            print("Sessions stored:", list(self.sessions.keys()))

            return {
                "status": "in_progress",
                "session_id": session_id,
                "user_id": user_id,
                "mode": mode,
                "phase": "technical",
                "current_skill": current_skill,
                "question": initial_question,
                "skill": current_skill,
                "skills": normalized_skills,
            }

    def attach_resume(self, session_id: str, resume_bytes: bytes, filename: str = "") -> None:
        """Attach uploaded resume bytes to an active session for later processing."""
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                return
            session["resume_bytes"] = resume_bytes
            session["resume_filename"] = filename

    def process_answer(self, session_id: str, answer: str) -> dict[str, Any]:
        """Process one answer by session_id and return next question or final report."""
        with self.lock:
            print("Incoming session_id:", session_id)
            print("Active sessions:", list(self.sessions.keys()))
            session = self.sessions.get(session_id)
            if session is None:
                return {
                    "status": "error",
                    "message": "No active session for this session_id",
                }
            if "asked_questions" not in session or not isinstance(session["asked_questions"], set):
                session["asked_questions"] = set()

            if not bool(session.get("runtime_ready", False)):
                intro_answer = (answer or "").strip()
                if intro_answer:
                    session["all_responses"].append({"phase": "intro", "answer": intro_answer})

                self._initialize_runtime(session)

                return {
                    "status": "in_progress",
                    "phase": "technical",
                    "current_skill": session["current_skill"],
                    "question": session["current_question"],
                    "message": "Starting technical round.",
                }

            phase = session["phase"]
            if phase == "technical" and bool(session.get("awaiting_next_question", False)):
                return {
                    "status": "error",
                    "message": "Evaluation ready. Call /next_question before submitting another answer.",
                }
            if phase == "technical":
                return self._process_technical_answer(session, answer)
            if phase == "hr":
                return self._process_hr_answer(session, answer)

            return {
                "status": "complete",
                "message": "Interview already completed.",
                "final_report": session.get("final_report", {}),
            }

    def get_next_question(self, session_id: str) -> dict[str, Any]:
        """Return the pending next question after an evaluated answer step."""
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                return {
                    "status": "error",
                    "message": "No active session for this session_id",
                }

            if session.get("phase") == "done":
                return {
                    "status": "completed",
                    "end": True,
                    "technical": session.get("final_report", {}).get("technical", {}),
                    "global": session.get("final_report", {}).get("global", {}),
                    "hr": session.get("final_report", {}).get("hr", {}),
                    "final_report": session.get("final_report", {}),
                }

            if not bool(session.get("awaiting_next_question", False)):
                return {
                    "status": "error",
                    "message": "No pending next question. Submit an answer first.",
                }

            next_question = str(session.get("pending_next_question", "")).strip()
            if not next_question:
                return {
                    "status": "error",
                    "message": "No next question available.",
                }

            pending_phase = str(session.get("pending_next_phase", "")).strip() or str(session.get("phase", ""))
            pending_skill = str(session.get("pending_next_skill", "")).strip() or str(
                session.get("current_skill", "")
            )

            session["current_question"] = next_question
            session["awaiting_next_question"] = False
            session["pending_next_question"] = ""
            session["pending_next_phase"] = ""
            session["pending_next_skill"] = ""
            session["pending_next_question_type"] = ""

            return {
                "status": "next_question",
                "phase": pending_phase,
                "current_skill": pending_skill,
                "question": next_question,
            }

    def _initialize_runtime(self, session: dict[str, Any]) -> None:
        """Build heavy interview runtime objects lazily on first answer."""
        user_id = str(session.get("user_id", "")).strip()
        name = str(session.get("name", "Candidate")).strip() or "Candidate"

        # Keep runtime bootstrap independent from MongoDB reads.
        user_data: dict[str, Any] = {"skills": {}}
        skills = [str(skill).strip() for skill in session.get("skills", []) if str(skill).strip()]
        if not skills:
            skills = list(DEFAULT_SKILLS)
        flow = LangGraphInterviewFlow(
            user_id=user_id,
            resume_data={
                "name": name,
                "skills": skills,
                "projects": [],
                "experience": [],
                "achievements": [],
            },
            mongo_manager=self.mongo_manager,
            cache_manager=self.cache_manager,
        )
        flow.initialize_skill_graphs(skills)

        skill_states: dict[str, dict[str, Any]] = {}
        for skill in skills:
            existing_state = session.get("skill_states", {}).get(skill)
            if isinstance(existing_state, dict):
                skill_states[skill] = existing_state
                session.setdefault("skill_graph", {})
                session["skill_graph"].setdefault(skill, {"weaknesses": []})
                continue

            bkt = BKTModel(initial_probability=BASELINE_PROBABILITY)
            cat = CATEngine(initial_difficulty=BASELINE_DIFFICULTY)
            skill_graph = SkillGraph(skill_name=skill)

            cached = self.cache_manager.get_weak_areas(user_id, skill)
            if cached:
                skill_graph.weak_areas = cached
            else:
                persisted_skill = user_data.get("skills", {}).get(skill, {})
                if isinstance(persisted_skill, dict):
                    weak = persisted_skill.get("weak_areas", {})
                    if isinstance(weak, dict):
                        skill_graph.weak_areas = weak

            skill_states[skill] = {
                "bkt_model": bkt,
                "cat_engine": cat,
                "skill_graph": skill_graph,
                "previous_question": "",
                "evaluation_summary": "",
                "selected_weak_areas": [],
                "question_count": 0,
                "probability": bkt.current_probability,
                "prev_probability": bkt.current_probability,
                "difficulty": self._probability_to_level(bkt.current_probability),
                "current_question": "",
                "question_difficulty": self._difficulty_level_to_numeric(
                    self._probability_to_level(bkt.current_probability)
                ),
            }
            session.setdefault("skill_graph", {})
            session["skill_graph"].setdefault(skill, {"weaknesses": []})

        current_skill = skills[0]
        first_question = self._generate_question(
            flow,
            current_skill,
            skill_states[current_skill],
            session["asked_questions"],
            session,
        )
        if first_question is None:
            raise Exception("No valid questions for current skill")

        session["flow"] = flow
        session["skills"] = skills
        session["skill_index"] = 0
        session["skill_states"] = skill_states
        session["current_skill"] = current_skill
        session["current_question"] = first_question
        session["question_count"] = int(skill_states[current_skill]["question_count"])
        flow.current_probability = float(skill_states[current_skill]["probability"])
        flow.current_difficulty = "easy"
        session["runtime_ready"] = True

    def get_dashboard(self, user_id: str) -> dict[str, Any]:
        """Return dashboard-ready history data for user."""
        if self.mongo_manager:
            user_data = self.mongo_manager.get_user(user_id)
        else:
            user_data = None
        if user_data is None:
            return {
                "user_id": user_id,
                "sessions": [],
                "overall_score_history": [],
                "skill_histories": {},
                "hr_metrics_history": {
                    "overall_score": [],
                    "leadership": [],
                    "problem_solving": [],
                    "adaptability": [],
                    "teamwork": [],
                },
            }

        sessions = user_data.get("sessions", [])
        if not isinstance(sessions, list):
            sessions = []

        overall_history: list[dict[str, Any]] = []
        skill_histories: dict[str, list[dict[str, Any]]] = {}
        hr_histories: dict[str, list[dict[str, Any]]] = {
            "overall_score": [],
            "leadership": [],
            "problem_solving": [],
            "adaptability": [],
            "teamwork": [],
        }

        for session in sessions:
            if not isinstance(session, dict):
                continue

            session_id = str(session.get("session_id", ""))
            timestamp = session.get("timestamp")

            tech = session.get("technical", {})
            if isinstance(tech, dict):
                overall_score = float(tech.get("overall_score", 0.0))
                overall_history.append(
                    {
                        "session_id": session_id,
                        "timestamp": timestamp,
                        "score": round(overall_score, 4),
                    }
                )

                skill_scores = tech.get("skill_scores", {})
                if isinstance(skill_scores, dict):
                    for skill, score in skill_scores.items():
                        if not isinstance(score, (int, float)):
                            continue
                        skill_histories.setdefault(str(skill), []).append(
                            {
                                "session_id": session_id,
                                "timestamp": timestamp,
                                "score": round(float(score), 4),
                            }
                        )

            hr = session.get("hr", {})
            if isinstance(hr, dict):
                for metric in hr_histories:
                    value = hr.get(metric)
                    if isinstance(value, (int, float)):
                        hr_histories[metric].append(
                            {
                                "session_id": session_id,
                                "timestamp": timestamp,
                                "score": round(float(value), 4),
                            }
                        )

        return {
            "user_id": user_id,
            "name": user_data.get("name", "Candidate"),
            "sessions": sessions,
            "overall_score_history": overall_history,
            "skill_histories": skill_histories,
            "hr_metrics_history": hr_histories,
        }

    def _process_technical_answer(self, session: dict[str, Any], answer: str) -> dict[str, Any]:
        """Process one technical answer and route to next question/skill/HR."""
        cleaned_answer = (answer or "").strip()
        if not cleaned_answer:
            return {
                "status": "retry",
                "phase": "technical",
                "question": session["current_question"],
                "message": "No answer detected. Please respond or type 'skip'.",
            }

        flow: LangGraphInterviewFlow = session["flow"]
        skill = session["current_skill"]
        skill_state = session.get("skill_states", {}).get(skill)
        if not isinstance(skill_state, dict):
            skill_state = {
                "bkt_model": BKTModel(initial_probability=BASELINE_PROBABILITY),
                "cat_engine": CATEngine(initial_difficulty=BASELINE_DIFFICULTY),
                "skill_graph": SkillGraph(skill_name=skill),
                "previous_question": "",
                "evaluation_summary": "",
                "selected_weak_areas": [],
                "question_count": 0,
                "probability": BASELINE_PROBABILITY,
                "prev_probability": BASELINE_PROBABILITY,
                "difficulty": self._probability_to_level(BASELINE_PROBABILITY),
                "current_question": session.get("current_question", ""),
                "question_difficulty": self._difficulty_level_to_numeric(
                    self._probability_to_level(BASELINE_PROBABILITY)
                ),
            }
            session.setdefault("skill_states", {})
            session["skill_states"][skill] = skill_state

        print("Current skill index:", session["skill_index"])
        print("All skills:", session["skills"])
        print("Current skill:", session["current_skill"])

        session["all_responses"].append({"phase": "technical", "answer": cleaned_answer})

        skipped_question = cleaned_answer.lower() == "skip"
        weakness_output: dict[str, Any] | None = None
        if skipped_question:
            score = 0.25
            evaluation = {
                "summary": "Candidate skipped this question.",
                "weak_areas": [],
                "confidence": 0.2,
                "final_score": score,
                "skipped_question": True,
            }
            current_weak_areas: list[str] = []
            selected_weak_areas: list[str] = []
        else:
            evaluation = flow.evaluation_agent.evaluate_answer(
                question=skill_state["current_question"],
                answer=cleaned_answer,
                skill=skill,
            )
            score = float(evaluation.get("final_score", 0.0))
            if score < flow.WEAKNESS_THRESHOLD:
                current_weak_areas = flow._safe_weak_areas(evaluation.get("weak_areas", []))
            else:
                current_weak_areas = []

            selected_weak_areas = flow._select_weak_areas(
                current_weak_areas,
                skill_state["skill_graph"].weak_areas,
            )

        question_difficulty = skill_state["question_difficulty"]
        bkt: BKTModel = skill_state["bkt_model"]
        sg: SkillGraph = skill_state["skill_graph"]

        if "technical_scores" not in session or not isinstance(session["technical_scores"], list):
            session["technical_scores"] = []
        session["technical_scores"].append(float(score))
        print("Stored technical score:", score)

        print("DEBUG STATE BEFORE QUESTION:")
        print("Skill:", session["current_skill"])
        print("Probability (flow.current_probability):", getattr(flow, "current_probability", None))
        print("Probability (skill_state):", skill_state.get("probability"))
        print("BKT current_probability:", bkt.current_probability)
        print("Difficulty (flow.current_difficulty):", getattr(flow, "current_difficulty", None))
        print("Difficulty (skill_state):", skill_state.get("difficulty"))
        print("Question count (session):", session.get("question_count"))
        print("Question count (skill_state):", skill_state.get("question_count"))

        current_difficulty = str(skill_state.get("difficulty", "easy"))
        prev_probability = float(skill_state["probability"])
        print("OVERWRITING STATE HERE: bkt.current_probability before update =", bkt.current_probability)
        next_attempt = int(skill_state.get("question_count", 0)) + 1
        new_probability = bkt.update(float(score), n=next_attempt)
        difficulty_level = self._probability_to_level(new_probability)

        if score < 0.70:
            if "skill_graph" not in session:
                session["skill_graph"] = {}
            if skill not in session["skill_graph"]:
                session["skill_graph"][skill] = {"weaknesses": []}

            if selected_weak_areas:
                question_hint = str(selected_weak_areas[0]).strip()
            elif current_weak_areas:
                question_hint = str(current_weak_areas[0]).strip()
            else:
                question_hint = f"core concepts in {skill}"

            weakness_output = {
                "question_hint": question_hint,
                "difficulty": current_difficulty,
            }
            session["skill_graph"][skill]["weaknesses"].append(weakness_output)

        skill_state["question_count"] += 1
        question_count = int(skill_state["question_count"])
        skill_state["prev_probability"] = prev_probability
        skill_state["probability"] = new_probability
        skill_state["difficulty"] = difficulty_level
        skill_state["question_difficulty"] = self._difficulty_level_to_numeric(difficulty_level)
        skill_state["previous_question"] = skill_state["current_question"]
        skill_state["evaluation_summary"] = str(evaluation.get("summary", "")).strip()
        skill_state["selected_weak_areas"] = selected_weak_areas
        session["question_count"] = question_count
        session["skill_states"][skill] = skill_state
        print(f"Question count: {question_count}")
        flow.current_probability = float(new_probability)
        flow.current_difficulty = difficulty_level

        p = float(bkt.current_probability)
        n = int(skill_state["question_count"])
        s = max(0.0, min(1.0, float(score)))
        bkt_width = compute_width(p=p, n=n, score=s)
        skill_state["bkt_width"] = bkt_width
        session["skill_states"][skill] = skill_state
        print(f"[BKT WIDTH DEBUG] p={p}, n={n}, s={s}, width={bkt_width}")

        max_questions = 7

        sg.update(
            score=float(score),
            probability=new_probability,
            weak_areas=selected_weak_areas,
            question_count=question_count,
            max_questions=max_questions,
            question_difficulty=question_difficulty,
        )

        if selected_weak_areas:
            for area in selected_weak_areas:
                session["session_weak_areas"].add(area)

        flow.cache_manager.set_weak_areas(flow.user_id, skill, sg.weak_areas)
        # DB writes are intentionally skipped during live answer flow to avoid blocking.
        # Final persistence happens after interview completion.

        if question_count >= MAX_QUESTIONS_PER_SKILL or (
            question_count >= MIN_QUESTIONS and float(skill_state.get("bkt_width", 1.0)) < WIDTH_THRESHOLD
        ):
            move_next_skill = True
        else:
            move_next_skill = False

        if not move_next_skill:
            next_question = self._generate_question(
                flow,
                skill,
                skill_state,
                session["asked_questions"],
                session,
            )
            if next_question is None:
                print("All questions exhausted for skill:", skill)
                move_next_skill = True
            else:
                session["pending_next_question"] = next_question
                session["pending_next_phase"] = "technical"
                session["pending_next_skill"] = skill
                session["pending_next_question_type"] = str(skill_state.get("next_question_type", ""))
                session["awaiting_next_question"] = True
                session["last_evaluation"] = {
                    "score": round(float(score), 4),
                    "bkt_probability": round(float(bkt.current_probability), 4),
                    "bkt_width": round(float(skill_state.get("bkt_width", 0.0)), 4),
                    "next_difficulty": str(skill_state.get("difficulty", "easy")),
                    "next_question_type": str(skill_state.get("next_question_type", "")),
                }
                return {
                    "status": "evaluated",
                    "score": round(float(score), 4),
                    "bkt_probability": round(float(bkt.current_probability), 4),
                    "bkt_width": round(float(skill_state.get("bkt_width", 0.0)), 4),
                    "next_difficulty": str(skill_state.get("difficulty", "easy")),
                    "next_question_type": str(skill_state.get("next_question_type", "")),
                    "weakness": weakness_output,
                }

        session["technical_results"][skill] = {
            "skill": skill,
            "proficiency": round(sg.proficiency, 4),
            "confidence": round(sg.confidence, 4),
            "questions_asked": skill_state["question_count"],
            "weak_areas": sg.weak_areas,
        }

        session["skill_index"] += 1

        print("Moving to next skill...")
        print("New index:", session["skill_index"])

        if session["skill_index"] < len(session["skills"]):
            next_skill = session["skills"][session["skill_index"]]
            session["current_skill"] = next_skill

            print("SWITCHING TO NEW SKILL:", next_skill)
            next_state = session["skill_states"][next_skill]
            next_state["question_count"] = 0
            next_state["probability"] = BASELINE_PROBABILITY
            next_state["prev_probability"] = BASELINE_PROBABILITY
            next_state["previous_question"] = ""
            next_state["evaluation_summary"] = ""
            next_state["selected_weak_areas"] = []
            next_state["current_question"] = ""
            next_state["difficulty"] = self._probability_to_level(BASELINE_PROBABILITY)

            # Force baseline right on skill switch so no cross-skill carry-over can survive.
            next_state["bkt_model"].current_probability = BASELINE_PROBABILITY
            next_state["cat_engine"].difficulty = BASELINE_DIFFICULTY
            next_state["question_difficulty"] = self._difficulty_level_to_numeric(next_state["difficulty"])
            session["question_count"] = 0
            flow.current_probability = BASELINE_PROBABILITY
            flow.current_difficulty = next_state["difficulty"]

            print("Resetting state for new skill:", next_skill)
            print("QUESTION GENERATION INPUT:")
            print("Probability:", flow.current_probability)
            print("Difficulty:", getattr(flow, "current_difficulty", None))

            # Force test baseline immediately before generator call.
            flow.current_probability = BASELINE_PROBABILITY
            flow.current_difficulty = next_state["difficulty"]
            next_state["bkt_model"].current_probability = BASELINE_PROBABILITY
            next_state["cat_engine"].difficulty = BASELINE_DIFFICULTY
            next_state["difficulty"] = self._probability_to_level(BASELINE_PROBABILITY)
            next_state["question_difficulty"] = self._difficulty_level_to_numeric(next_state["difficulty"])

            print("Next skill:", next_skill)
            next_question = self._generate_question(
                flow,
                next_skill,
                next_state,
                session["asked_questions"],
                session,
            )
            if next_question is None:
                raise Exception("No valid questions for current skill")
            session["pending_next_question"] = next_question
            session["pending_next_phase"] = "technical"
            session["pending_next_skill"] = next_skill
            session["pending_next_question_type"] = str(next_state.get("next_question_type", ""))
            session["awaiting_next_question"] = True
            session["last_evaluation"] = {
                "score": round(float(score), 4),
                "bkt_probability": round(float(bkt.current_probability), 4),
                "bkt_width": round(float(skill_state.get("bkt_width", 0.0)), 4),
                "next_difficulty": str(next_state.get("difficulty", "easy")),
                "next_question_type": str(next_state.get("next_question_type", "")),
            }
            return {
                "status": "evaluated",
                "score": round(float(score), 4),
                "bkt_probability": round(float(bkt.current_probability), 4),
                "bkt_width": round(float(skill_state.get("bkt_width", 0.0)), 4),
                "next_difficulty": str(next_state.get("difficulty", "easy")),
                "next_question_type": str(next_state.get("next_question_type", "")),
                "weakness": weakness_output,
            }

        print("All skills completed -> moving to HR")
        session["phase"] = "hr"
        session["hr_index"] = 0
        session["hr_per_question"] = []
        first_hr_question = flow.hr_agent.questions[0]
        session["pending_next_question"] = first_hr_question
        session["pending_next_phase"] = "hr"
        session["pending_next_skill"] = ""
        session["pending_next_question_type"] = "hr"
        session["awaiting_next_question"] = True
        session["last_evaluation"] = {
            "score": round(float(score), 4),
            "bkt_probability": round(float(bkt.current_probability), 4),
            "bkt_width": round(float(skill_state.get("bkt_width", 0.0)), 4),
            "next_difficulty": "hr",
            "next_question_type": "hr",
        }

        return {
            "status": "evaluated",
            "score": round(float(score), 4),
            "bkt_probability": round(float(bkt.current_probability), 4),
            "bkt_width": round(float(skill_state.get("bkt_width", 0.0)), 4),
            "next_difficulty": "hr",
            "next_question_type": "hr",
            "weakness": weakness_output,
        }

    def _process_hr_answer(self, session: dict[str, Any], answer: str) -> dict[str, Any]:
        """Process one HR answer and finalize report if interview is complete."""
        cleaned_answer = (answer or "").strip()
        if not cleaned_answer:
            return {
                "status": "retry",
                "phase": "hr",
                "question": session["current_question"],
                "message": "No answer detected. Please respond or type 'skip'.",
            }

        flow: LangGraphInterviewFlow = session["flow"]
        total_hr_questions = min(MAX_HR_QUESTIONS, len(flow.hr_agent.questions))
        print("HR index:", session["hr_index"])
        print("Total HR questions:", total_hr_questions)

        if session["hr_index"] >= total_hr_questions:
            print("HR questions finished - ending interview")
            return self._finalize_session(session)

        if session["hr_index"] >= 1:
            return self._finalize_session(session)

        question = flow.hr_agent.questions[session["hr_index"]]

        evaluation = flow.hr_agent.evaluate_response(question=question, answer=cleaned_answer)
        session["all_responses"].append({"phase": "hr", "answer": cleaned_answer})

        per_answer_behavioral = BehavioralAnalyzer().analyze(
            [{"phase": "hr", "answer": cleaned_answer}]
        )
        communication_score = float(per_answer_behavioral.get("communication", 0.0))
        confidence_score = float(per_answer_behavioral.get("confidence", 0.0))
        leadership_score = float(evaluation.get("leadership", 0.0))
        problem_solving_score = float(evaluation.get("problem_solving", 0.0))
        adaptability_score = float(evaluation.get("adaptability", 0.0))
        teamwork_score = float(evaluation.get("teamwork", 0.0))

        if "hr_scores" not in session or not isinstance(session["hr_scores"], list):
            session["hr_scores"] = []
        session["hr_scores"].append(
            {
                "communication": communication_score,
                "confidence": confidence_score,
                "leadership": leadership_score,
                "problem_solving": problem_solving_score,
                "adaptability": adaptability_score,
                "teamwork": teamwork_score,
            }
        )

        session["hr_per_question"].append(
            {
                "question": question,
                "answer": cleaned_answer,
                "scores": {
                    "communication": communication_score,
                    "confidence": confidence_score,
                    "leadership": leadership_score,
                    "problem_solving": problem_solving_score,
                    "adaptability": adaptability_score,
                    "teamwork": teamwork_score,
                },
                "summary": evaluation["summary"],
            }
        )

        next_hr_index = session["hr_index"] + 1
        if next_hr_index < total_hr_questions:
            next_question = flow.hr_agent.questions[next_hr_index]
            session["current_question"] = next_question
            session["hr_index"] = next_hr_index
            return {
                "status": "in_progress",
                "phase": "hr",
                "question": next_question,
            }

        session["hr_index"] = next_hr_index

        return self._finalize_session(session)

    def build_final_report(self, session: dict[str, Any]) -> dict[str, Any]:
        """Build final report from stored session data for API response."""
        technical_results = session.get("technical_results", {})
        hr_scores = session.get("hr_results", {}).get("average_scores", {})

        technical = {
            "overall_score": sum([v["proficiency"] for v in technical_results.values()])
            / max(len(technical_results), 1),
            "skills": {
                k: v["proficiency"] for k, v in technical_results.items()
            },
        }

        global_scores = {
            "communication": hr_scores.get("communication", 0),
            "confidence": hr_scores.get("confidence", 0),
        }

        hr = {
            "leadership": hr_scores.get("leadership", 0),
            "problem_solving": hr_scores.get("problem_solving", 0),
            "adaptability": hr_scores.get("adaptability", 0),
            "teamwork": hr_scores.get("teamwork", 0),
        }

        hr_overall = sum(hr.values()) / max(len(hr), 1)

        return {
            "technical": technical,
            "global": global_scores,
            "hr": {
                "overall_score": hr_overall,
                **hr,
            },
        }

    def _finalize_session(self, session: dict[str, Any]) -> dict[str, Any]:
        """Aggregate all stored scores and build final report payload."""
        flow: LangGraphInterviewFlow = session["flow"]

        technical_scores = session.get("technical_results", {})
        if technical_scores:
            final_technical = sum(
                skill["proficiency"] for skill in technical_scores.values()
            ) / len(technical_scores)
        else:
            final_technical = 0

        hr_scores = [
            row
            for row in session.get("hr_scores", [])
            if isinstance(row, dict)
        ]
        communication = self._average([float(x.get("communication", 0.0)) for x in hr_scores])
        confidence = self._average([float(x.get("confidence", 0.0)) for x in hr_scores])
        leadership = self._average([float(x.get("leadership", 0.0)) for x in hr_scores])
        problem_solving = self._average([float(x.get("problem_solving", 0.0)) for x in hr_scores])
        adaptability = self._average([float(x.get("adaptability", 0.0)) for x in hr_scores])
        teamwork = self._average([float(x.get("teamwork", 0.0)) for x in hr_scores])

        hr_results = {
            "questions_asked": len(session.get("hr_per_question", [])),
            "per_question": session.get("hr_per_question", []),
            "average_scores": {
                "communication": round(communication, 4),
                "confidence": round(confidence, 4),
                "leadership": round(leadership, 4),
                "problem_solving": round(problem_solving, 4),
                "adaptability": round(adaptability, 4),
                "teamwork": round(teamwork, 4),
            },
            "responses": [
                item.get("answer", "")
                for item in session.get("hr_per_question", [])
                if isinstance(item, dict)
            ],
        }
        session["hr_results"] = hr_results

        transcript_rows = [
            row
            for row in session["all_responses"]
            if isinstance(row, dict) and str(row.get("answer", "")).strip()
        ]
        full_transcript = " ".join([r["answer"] for r in transcript_rows])
        behavioral_input = [{"phase": "full", "answer": full_transcript}] if full_transcript else []
        behavioral = BehavioralAnalyzer().analyze(behavioral_input)

        flow.session_id = session["session_id"]
        flow.session_weak_areas = set(session["session_weak_areas"])

        skill_scores = {
            skill: float(data.get("proficiency", 0.0))
            for skill, data in session["technical_results"].items()
        }
        session_data = {
            "session_id": session["session_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "technical": {
                "overall_score": final_technical,
                "scores": [
                    float(value)
                    for value in session.get("technical_scores", [])
                    if isinstance(value, (int, float))
                ],
                "skill_scores": skill_scores,
                "session_weak_areas": sorted(session["session_weak_areas"]),
                "summary": generate_summary(skill_scores, sorted(session["session_weak_areas"])),
            },
            "hr": {
                "overall_score": self._average(
                    [communication, confidence, leadership, problem_solving]
                ),
                "communication": communication,
                "confidence": confidence,
                "leadership": leadership,
                "problem_solving": problem_solving,
                "adaptability": adaptability,
                "teamwork": teamwork,
            },
            "behavioral": {
                "communication": float(behavioral.get("communication", 0.0)),
                "confidence": float(behavioral.get("confidence", 0.0)),
            },
        }

        # self.mongo_manager.append_session(session["user_id"], session_data)
        # user_data = self.mongo_manager.get_user(session["user_id"])
        user_data = {}
        _ = generate_final_report(user_data, session_data)

        final_report = self.build_final_report(session)
        print("FINAL REPORT:", final_report)

        session["phase"] = "done"
        session["final_report"] = final_report

        print("FINAL SESSION DATA:", session)
        print("FINAL TECH SCORES:", final_report["technical"].get("skills", {}))
        print("FINAL TECH AVG:", final_report["technical"].get("overall_score", 0))
        print("🔥 RETURNING FINAL REPORT")
        return {
            "status": "completed",
            "end": True,
            "technical": final_report["technical"],
            "global": final_report["global"],
            "hr": final_report["hr"],
            "final_report": final_report,
        }

    def _generate_question(
        self,
        flow: LangGraphInterviewFlow,
        skill: str,
        skill_state: dict[str, Any],
        asked_questions: set[str],
        session: dict[str, Any],
    ) -> str | None:
        """Generate next technical question from BKT probability and LLM prompting."""
        current_skill = str(skill).strip()
        modality = str(session.get("modality", "oral")).strip() or "oral"
        probability = float(skill_state.get("probability", BASELINE_PROBABILITY))
        difficulty_level = self._probability_to_level(probability)
        types_for_level = QUESTION_TYPES_BY_LEVEL.get(difficulty_level, QUESTION_TYPES_BY_LEVEL["easy"])

        weaknesses = session.get("skill_graph", {}).get(current_skill, {}).get("weaknesses", [])
        matching = [w for w in weaknesses if w.get("difficulty") == difficulty_level]
        if matching and random.random() < 0.7:
            selected_weakness = random.choice(matching)
            use_weakness = True
        else:
            selected_weakness = None
            use_weakness = False

        print("DEBUG STATE BEFORE QUESTION:")
        print("Skill:", current_skill)
        print("Probability:", probability)
        print("Difficulty:", difficulty_level)
        print("Question count:", int(skill_state.get("question_count", 0)))

        print("QUESTION GENERATION INPUT:")
        print("Probability:", probability)
        print("Difficulty:", difficulty_level)

        question = ""
        selected_question_type = ""
        selected_source_type = "fallback"
        for _ in range(5):
            question_type = random.choice(types_for_level)
            print("Selected Question Type:", question_type)
            candidate = flow.question_agent.generate_bkt_question(
                skill=current_skill,
                level=difficulty_level,
                question_type=question_type,
                modality=modality,
                previous_questions=list(asked_questions),
                weakness_hint=str(selected_weakness.get("question_hint", ""))
                if use_weakness and isinstance(selected_weakness, dict)
                else None,
            )
            candidate = str(candidate).strip()
            if candidate and candidate not in asked_questions:
                question = candidate
                selected_question_type = question_type
                selected_source_type = str(getattr(flow.question_agent, "last_source_type", "llm"))
                break

        if not question:
            print("All generated questions repeated for skill:", current_skill)
            return None

        print("Asked so far:", asked_questions)
        print("Selected:", question)
        print("Current Skill:", current_skill)
        print(
            f"[QUESTION SOURCE] source={selected_source_type}, difficulty={difficulty_level}, type={selected_question_type}"
        )

        asked_questions.add(question)
        skill_state["question_difficulty"] = self._difficulty_level_to_numeric(difficulty_level)
        skill_state["difficulty"] = difficulty_level
        skill_state["next_question_type"] = selected_question_type
        skill_state["current_question"] = question
        return question

    @staticmethod
    def _probability_to_level(probability: float) -> str:
        """Map BKT probability to named difficulty level."""
        p = max(0.0, min(1.0, float(probability)))
        if p <= 0.5:
            return "easy"
        if p <= 0.625:
            return "easy-medium"
        if p <= 0.75:
            return "medium"
        if p <= 0.875:
            return "hard"
        return "extreme-hard"

    @staticmethod
    def _difficulty_level_to_numeric(level: str) -> int:
        """Convert named difficulty level to numeric band for metadata compatibility."""
        mapping = {
            "easy": 1,
            "easy-medium": 2,
            "medium": 3,
            "hard": 4,
            "extreme-hard": 5,
        }
        return mapping.get(str(level).strip(), 1)

    @staticmethod
    def _resolve_skills(user_data: dict[str, Any]) -> list[str]:
        """Resolve user skills from persistent profile with defaults."""
        skills = user_data.get("skills", {})
        if isinstance(skills, dict) and skills:
            return [str(skill) for skill in skills.keys()]
        return list(DEFAULT_SKILLS)

    @staticmethod
    def _average(values: list[float]) -> float:
        """Compute average with safe zero fallback."""
        if not values:
            return 0.0
        return sum(values) / len(values)
