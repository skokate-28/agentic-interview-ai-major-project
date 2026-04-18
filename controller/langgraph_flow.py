"""LangGraph-based orchestration flow for the multi-agent interview pipeline."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, TypedDict
from uuid import uuid4

from agents.evaluation_agent import EvaluationAgent
from agents.hr_agent import HRAgent
from agents.question_agent import QuestionAgent
from config import settings
from core.bkt_engine import BKTModel
from core.cat_engine import CATEngine
from core.skill_graph import SkillGraph
from database.cache_manager import CacheManager
from database.mongo_client import MongoDBManager
from reporting.final_report import generate_final_report, generate_summary
from utils.behavioral_analyzer import BehavioralAnalyzer


def get_state_graph() -> Any:
    """Safely import and return LangGraph StateGraph class at runtime."""
    from langgraph.graph import StateGraph

    return StateGraph


class InterviewState(TypedDict, total=False):
    """Shared interview state across LangGraph nodes."""

    user_id: str
    skills: List[str]

    current_skill: str
    question: str
    answer: str

    evaluation: Dict[str, Any]
    score: float

    probability: float
    prev_probability: float
    difficulty: int
    question_difficulty: int

    weak_areas: List[str]
    selected_weak_areas: List[str]

    question_count: int
    max_questions: int
    done: bool
    skipped_question: bool

    skill_results: Dict[str, Any]
    all_responses: List[Dict[str, str]]

    hr_results: Dict[str, Any]
    behavioral_scores: Dict[str, Any]
    final_report: Dict[str, Any]
    session_id: str

    bkt_model: Any
    cat_engine: Any
    skill_graph: Any


class LangGraphInterviewFlow:
    """Graph-driven interview orchestrator preserving adaptive behavior and user isolation."""

    WEAKNESS_THRESHOLD = 0.4

    def __init__(
        self,
        user_id: str,
        resume_data: Dict[str, list[str]] | None = None,
        mongo_manager: MongoDBManager | None = None,
        cache_manager: CacheManager | None = None,
    ) -> None:
        self.user_id = user_id
        self.resume_data = resume_data or {
            "skills": [],
            "projects": [],
            "experience": [],
            "achievements": [],
        }
        self.mongo_manager = mongo_manager or MongoDBManager()
        self.db_manager = self.mongo_manager
        self.cache_manager = cache_manager or CacheManager()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.question_agent = QuestionAgent()
        self.evaluation_agent = EvaluationAgent()
        self.hr_agent = HRAgent()

        self._state_graph_cls = get_state_graph()

        self.skill_graphs: Dict[str, SkillGraph] = {}
        self.bkt_models: Dict[str, BKTModel] = {}
        self.cat_engines: Dict[str, CATEngine] = {}
        self.all_responses: list[dict[str, str]] = []
        self.session_weak_areas: set[str] = set()
        self.session_id: str = ""

        self.skill_app = self._build_skill_graph().compile()
        self.main_app = self._build_main_graph().compile()

    def load_user_profile(self) -> None:
        """Load persisted resume profile for the current user if available."""
        existing_user = self.mongo_manager.get_user(self.user_id)
        if existing_user is None:
            return

        existing_skills = existing_user.get("skills", {})
        if isinstance(existing_skills, dict):
            persisted_skill_names = [str(skill) for skill in existing_skills.keys()]
            if persisted_skill_names and not self.resume_data.get("skills"):
                self.resume_data["skills"] = persisted_skill_names

    def initialize_skill_graphs(self, skills: list[str]) -> None:
        """Initialize adaptive engine holders for each skill."""
        for skill in skills:
            self.skill_graphs[skill] = SkillGraph(skill_name=skill)
            self.bkt_models[skill] = BKTModel()
            self.cat_engines[skill] = CATEngine()

    def start_interview(self) -> Dict[str, object]:
        """Run full interview flow: technical skill loop, HR, then behavioral analysis."""
        self.all_responses = []
        self.session_weak_areas = set()
        self.session_id = str(uuid4())
        self.load_user_profile()

        candidate_name = str(self.resume_data.get("name", self.user_id))
        self.mongo_manager.create_user(self.user_id, candidate_name)

        skills = self.resume_data.get("skills", [])
        self.initialize_skill_graphs(skills)

        initial_state: InterviewState = {
            "user_id": self.user_id,
            "skills": skills,
            "skill_results": {},
            "all_responses": [],
            "done": False,
            "session_id": self.session_id,
        }

        final_state = self.main_app.invoke(initial_state)
        technical_round = final_state.get("skill_results", {})
        hr_round = final_state.get("hr_results", {})
        behavioral_scores = final_state.get("behavioral_scores", {})

        session_data = self._build_session_data(technical_round, hr_round, behavioral_scores)

        for skill, payload in technical_round.items():
            if not isinstance(payload, dict):
                continue
            proficiency = float(payload.get("proficiency", 0.0))
            confidence = float(payload.get("confidence", 0.0))
            weak_areas = payload.get("weak_areas", {})
            if not isinstance(weak_areas, dict):
                weak_areas = {}

            self.mongo_manager.update_skill(
                self.user_id,
                {
                    skill: {
                        "proficiency": proficiency,
                        "confidence": confidence,
                        "weak_areas": weak_areas,
                        "session_id": self.session_id,
                        "score": proficiency,
                    }
                },
            )

        self.mongo_manager.append_session(self.user_id, session_data)
        user_data = self.mongo_manager.get_user(self.user_id) or {}
        final_report = generate_final_report(user_data, session_data)
        final_state["final_report"] = final_report

        print("\nFinal Report:")
        print(final_report)

        return {
            "technical_round": technical_round,
            "hr_round": hr_round,
            "behavioral_scores": behavioral_scores,
            "final_report": final_report,
        }

    def run_technical_round(self, skills: list[str]) -> Dict[str, Dict[str, object]]:
        """Run the technical round skill loop using the LangGraph skill subgraph."""
        current_responses = list(self.all_responses)
        results: Dict[str, Dict[str, object]] = {}
        max_questions = max(1, settings.MAX_QUESTIONS_PER_SKILL)

        for skill in skills:
            bkt = BKTModel(initial_probability=0.5)
            cat = CATEngine(initial_difficulty=3)
            skill_graph = SkillGraph(skill_name=skill)

            self.bkt_models[skill] = bkt
            self.cat_engines[skill] = cat

            cached_weak_areas = self.cache_manager.get_weak_areas(self.user_id, skill)
            if cached_weak_areas:
                skill_graph.weak_areas = cached_weak_areas
            else:
                user_data = self.mongo_manager.get_user(self.user_id) or {}
                user_skills = user_data.get("skills", {})
                if isinstance(user_skills, dict):
                    persisted_skill = user_skills.get(skill, {})
                    if isinstance(persisted_skill, dict):
                        persisted_weak_areas = persisted_skill.get("weak_areas", {})
                        if isinstance(persisted_weak_areas, dict):
                            skill_graph.weak_areas = persisted_weak_areas

            print("\n--- DEBUG: CACHE ---")
            print("User:", self.user_id)
            print("Skill:", skill)
            print("Loaded Weak Areas:", skill_graph.weak_areas)
            print("Confidence:", skill_graph.confidence)

            self.skill_graphs[skill] = skill_graph
            initial_skill_state: InterviewState = {
                "user_id": self.user_id,
                "skills": skills,
                "current_skill": skill,
                "question": "",
                "answer": "",
                "evaluation": {},
                "score": 0.0,
                "probability": bkt.current_probability,
                "prev_probability": bkt.current_probability,
                "difficulty": cat.get_difficulty(),
                "question_difficulty": cat.get_difficulty(),
                "weak_areas": [],
                "selected_weak_areas": [],
                "question_count": 0,
                "max_questions": max_questions,
                "done": False,
                "skipped_question": False,
                "skill_results": results,
                "all_responses": current_responses,
                "bkt_model": bkt,
                "cat_engine": cat,
                "skill_graph": skill_graph,
                "session_id": self.session_id,
            }

            final_skill_state = self.skill_app.invoke(initial_skill_state)
            current_responses = list(final_skill_state.get("all_responses", current_responses))

            latest_graph = final_skill_state["skill_graph"]
            self.skill_graphs[skill] = latest_graph
            self.bkt_models[skill] = final_skill_state["bkt_model"]
            self.cat_engines[skill] = final_skill_state["cat_engine"]
            results[skill] = {
                "skill": skill,
                "proficiency": round(latest_graph.proficiency, 4),
                "confidence": round(latest_graph.confidence, 4),
                "questions_asked": final_skill_state.get("question_count", 0),
                "weak_areas": latest_graph.weak_areas,
            }

        self.all_responses = current_responses
        return results

    def run_hr_round(self) -> Dict[str, object]:
        """Run the fixed HR round and merge responses into global response history."""
        hr_results = self.hr_agent.run_hr_round()
        responses = hr_results.get("responses", [])
        if isinstance(responses, list):
            self.all_responses.extend(
                [
                    {"phase": "hr", "answer": str(response).strip()}
                    for response in responses
                    if str(response).strip()
                ]
            )
        return hr_results

    def question_node(self, state: InterviewState) -> InterviewState:
        """Generate next technical question for current skill and CAT difficulty."""
        current_skill = str(state["current_skill"]).strip()
        difficulty_level = int(state.get("difficulty", 3))
        state["question_difficulty"] = difficulty_level

        candidate_questions = self.question_agent.get_candidate_questions(current_skill)
        if not candidate_questions:
            raise Exception("No valid questions for current skill")

        next_question = self.question_agent.select_question_based_on_difficulty(
            candidate_questions,
            difficulty_level,
        )

        print("Current Skill:", current_skill)
        print("Selected Question Skill:", next_question["skill"])

        if next_question["skill"] != current_skill:
            raise Exception(
                f"SKILL DRIFT ERROR: expected {current_skill}, got {next_question['skill']}"
            )

        state["question"] = str(next_question["question"])
        return state

    def answer_node(self, state: InterviewState) -> InterviewState:
        """Collect candidate answer from terminal and add to global response list."""
        skill = state["current_skill"]
        question = state.get("question", "")
        answer = self._collect_answer(skill=skill, question=question)

        retries = 0
        while answer is None or answer.strip() == "":
            print("No answer detected. Please respond or type 'skip'.")
            retries += 1
            if retries >= 3:
                answer = "skip"
                break
            answer = self._collect_plain_answer()

        skipped_question = answer.strip().lower() == "skip"

        if "all_responses" not in state:
            state["all_responses"] = []
        state["all_responses"].append({"phase": "technical", "answer": answer})

        state["answer"] = answer
        state["skipped_question"] = skipped_question
        return state

    def evaluation_node(self, state: InterviewState) -> InterviewState:
        """Evaluate the candidate answer for the current question."""
        skill = state["current_skill"]
        question = state.get("question", "")
        answer = state.get("answer", "")

        if answer is None or str(answer).strip() == "":
            state["evaluation"] = {
                "summary": "No answer detected.",
                "weak_areas": [],
                "skipped_question": False,
            }
            state["score"] = 0.0
            state["weak_areas"] = []
            state["selected_weak_areas"] = []
            return state

        if bool(state.get("skipped_question", False)):
            skip_score = 0.25
            state["evaluation"] = {
                "summary": "Candidate skipped this question.",
                "weak_areas": [],
                "skipped_question": True,
                "confidence": 0.2,
                "final_score": skip_score,
            }
            state["score"] = skip_score
            state["weak_areas"] = []
            state["selected_weak_areas"] = []
            state["question_count"] = int(state.get("question_count", 0)) + 1
            return state

        evaluation = self.evaluation_agent.evaluate_answer(
            question=question,
            answer=answer,
            skill=skill,
        )

        score = self._safe_score(evaluation.get("final_score", 0.0))
        if score < self.WEAKNESS_THRESHOLD:
            current_weak_areas = self._safe_weak_areas(evaluation.get("weak_areas", []))
        else:
            current_weak_areas = []

        historical_weak_areas = state["skill_graph"].weak_areas
        selected_weak_areas = self._select_weak_areas(current_weak_areas, historical_weak_areas)

        state["evaluation"] = evaluation
        state["score"] = score
        state["weak_areas"] = current_weak_areas
        state["selected_weak_areas"] = selected_weak_areas
        state["question_count"] = int(state.get("question_count", 0)) + 1

        for area in selected_weak_areas:
            self.session_weak_areas.add(area)

        return state

    def bkt_node(self, state: InterviewState) -> InterviewState:
        """Update BKT mastery probability for current skill."""
        bkt: BKTModel = state["bkt_model"]

        state["prev_probability"] = float(state.get("probability", bkt.current_probability))
        question_count = int(state.get("question_count", 0))
        new_probability = bkt.update(float(state.get("score", 0.0)), n=question_count)

        state["probability"] = new_probability
        return state

    def cat_node(self, state: InterviewState) -> InterviewState:
        """Update CAT difficulty for current skill based on evaluation score."""
        cat: CATEngine = state["cat_engine"]
        difficulty = cat.update_difficulty(float(state.get("score", 0.0)))
        state["difficulty"] = difficulty
        return state

    def skillgraph_node(self, state: InterviewState) -> InterviewState:
        """Update skill graph metadata and persist weak-area cache/source-of-truth state."""
        skill = state["current_skill"]
        skill_graph: SkillGraph = state["skill_graph"]

        question_count = int(state.get("question_count", 0))
        max_questions = int(state.get("max_questions", max(1, settings.MAX_QUESTIONS_PER_SKILL)))
        score = float(state.get("score", 0.0))
        probability = float(state.get("probability", 0.0))
        selected_weak_areas = self._safe_weak_areas(state.get("selected_weak_areas", []))
        question_difficulty = int(state.get("question_difficulty", state.get("difficulty", 1)))

        skill_graph.update(
            score=score,
            probability=probability,
            weak_areas=selected_weak_areas,
            question_count=question_count,
            max_questions=max_questions,
            question_difficulty=question_difficulty,
        )

        self.cache_manager.set_weak_areas(self.user_id, skill, skill_graph.weak_areas)
        self.mongo_manager.update_skill(
            self.user_id,
            {
                skill: {
                    "proficiency": skill_graph.proficiency,
                    "confidence": skill_graph.confidence,
                    "weak_areas": skill_graph.weak_areas,
                }
            },
        )

        print("\n--- DEBUG: CACHE ---")
        print("User:", self.user_id)
        print("Skill:", skill)
        print("Loaded Weak Areas:", skill_graph.weak_areas)
        print("Confidence:", skill_graph.confidence)

        state["question_count"] = question_count
        return state

    @staticmethod
    def decision_state_node(state: InterviewState) -> InterviewState:
        """Pass-through state node used before conditional transition."""
        return state

    def decision_node(self, state: InterviewState) -> str:
        """Choose next graph transition: continue questioning or move to next skill."""
        probability = float(state.get("probability", 0.0))
        prev_probability = float(state.get("prev_probability", 0.0))
        question_count = int(state.get("question_count", 0))
        max_questions = int(state.get("max_questions", max(1, settings.MAX_QUESTIONS_PER_SKILL)))

        delta = abs(probability - prev_probability)
        decision = "continue"

        if question_count >= max_questions:
            decision = "next_skill"
        elif delta < 0.02 and question_count >= 3:
            decision = "next_skill"

        print("\n--- DEBUG LOOP ---")
        print("Skill:", state.get("current_skill", ""))
        print("Q Count:", question_count)
        print("Prob:", probability)
        print("Delta:", delta)
        print("Decision:", decision)
        print(f"[DEBUG] Decision: {decision}")
        print(f"[DEBUG] State ID: {id(state)}")

        return decision

    def technical_node(self, state: InterviewState) -> InterviewState:
        """Main-flow technical node that runs per-skill adaptive subgraph loop."""
        skills = state.get("skills", [])
        technical_results = self.run_technical_round(skills)

        state["skill_results"] = technical_results
        state["all_responses"] = list(self.all_responses)
        return state

    def hr_node(self, state: InterviewState) -> InterviewState:
        """Run HR round and persist outputs into shared state."""
        hr_results = self.run_hr_round()
        hr_answers = hr_results.get("responses", [])
        if "all_responses" not in state:
            state["all_responses"] = []
        if isinstance(hr_answers, list):
            state["all_responses"].extend(
                [
                    {"phase": "hr", "answer": str(answer).strip()}
                    for answer in hr_answers
                    if str(answer).strip()
                ]
            )
            self.all_responses = list(state["all_responses"])
        state["hr_results"] = hr_results
        return state

    def behavioral_node(self, state: InterviewState) -> InterviewState:
        """Run global behavioral analysis over all responses and store results."""
        analyzer = BehavioralAnalyzer()
        behavioral_scores = analyzer.analyze(list(state.get("all_responses", [])))
        state["behavioral_scores"] = behavioral_scores
        print("\nBehavioral Scores:")
        print(behavioral_scores)
        return state

    def _build_skill_graph(self) -> Any:
        """Build adaptive technical questioning subgraph for one skill."""
        graph = self._state_graph_cls(InterviewState)
        graph.add_node("question", self.question_node)
        graph.add_node("answer", self.answer_node)
        graph.add_node("evaluation", self.evaluation_node)
        graph.add_node("bkt", self.bkt_node)
        graph.add_node("cat", self.cat_node)
        graph.add_node("skillgraph", self.skillgraph_node)
        graph.add_node("decision", self.decision_state_node)

        graph.set_entry_point("question")
        graph.add_edge("question", "answer")
        graph.add_edge("answer", "evaluation")
        graph.add_edge("evaluation", "bkt")
        graph.add_edge("bkt", "cat")
        graph.add_edge("cat", "skillgraph")
        graph.add_edge("skillgraph", "decision")

        graph.add_conditional_edges(
            "decision",
            self.decision_node,
            {
                "continue": "question",
                "next_skill": "__end__",
            },
        )
        return graph

    def _build_main_graph(self) -> Any:
        """Build top-level flow graph: technical -> HR -> behavioral -> end."""
        graph = self._state_graph_cls(InterviewState)
        graph.add_node("technical", self.technical_node)
        graph.add_node("hr", self.hr_node)
        graph.add_node("behavioral", self.behavioral_node)

        graph.set_entry_point("technical")
        graph.add_edge("technical", "hr")
        graph.add_edge("hr", "behavioral")
        graph.add_edge("behavioral", "__end__")
        return graph

    def _build_session_data(
        self,
        technical_round: Dict[str, Any],
        hr_round: Dict[str, Any],
        behavioral_scores: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create one MongoDB session object from current interview outputs."""
        skill_scores: Dict[str, float] = {}
        for skill, data in technical_round.items():
            if isinstance(data, dict):
                skill_scores[skill] = float(data.get("proficiency", 0.0))

        technical_overall = self._average(list(skill_scores.values()))
        session_weak = sorted(self.session_weak_areas)
        technical_summary = generate_summary(skill_scores, session_weak)

        avg_scores = hr_round.get("average_scores", {})
        if not isinstance(avg_scores, dict):
            avg_scores = {}

        hr_block = {
            "overall_score": self._average(
                [
                    float(avg_scores.get("leadership", 0.0)),
                    float(avg_scores.get("problem_solving", 0.0)),
                    float(avg_scores.get("adaptability", 0.0)),
                    float(avg_scores.get("teamwork", 0.0)),
                ]
            ),
            "leadership": float(avg_scores.get("leadership", 0.0)),
            "problem_solving": float(avg_scores.get("problem_solving", 0.0)),
            "adaptability": float(avg_scores.get("adaptability", 0.0)),
            "teamwork": float(avg_scores.get("teamwork", 0.0)),
        }

        behavioral_block = {
            "communication": float(
                behavioral_scores.get("communication", behavioral_scores.get("communication_score", 0.0))
            ),
            "confidence": float(
                behavioral_scores.get("confidence", behavioral_scores.get("confidence_score", 0.0))
            ),
            "summary": str(
                behavioral_scores.get(
                    "behavioral_summary",
                    behavioral_scores.get("summary", "Behavioral analysis completed."),
                )
            ).strip(),
        }

        return {
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "technical": {
                "overall_score": round(technical_overall, 4),
                "skill_scores": {k: round(v, 4) for k, v in skill_scores.items()},
                "session_weak_areas": session_weak,
                "summary": technical_summary,
            },
            "hr": {
                "overall_score": round(hr_block["overall_score"], 4),
                "leadership": round(hr_block["leadership"], 4),
                "problem_solving": round(hr_block["problem_solving"], 4),
                "adaptability": round(hr_block["adaptability"], 4),
                "teamwork": round(hr_block["teamwork"], 4),
            },
            "behavioral": {
                "communication": round(behavioral_block["communication"], 4),
                "confidence": round(behavioral_block["confidence"], 4),
                "summary": behavioral_block["summary"],
            },
        }

    @staticmethod
    def _collect_answer(skill: str, question: str) -> str:
        """Collect candidate answer from terminal input safely."""
        try:
            return input(f"\n{skill} Question: {question}\nYour Answer: ").strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    @staticmethod
    def _collect_plain_answer() -> str:
        """Collect a follow-up answer prompt without question preface."""
        try:
            return input("Your Answer: ").strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    @staticmethod
    def _safe_score(value: object) -> float:
        """Convert evaluation score to bounded float."""
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = 0.0

        return max(0.0, min(1.0, score))

    @staticmethod
    def _safe_weak_areas(value: object) -> list[str]:
        """Normalize weak area list from evaluator output."""
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _average(values: list[float]) -> float:
        """Compute average with safe zero fallback."""
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _select_weak_areas(
        current_weak: list[str],
        historical_weak: dict[str, dict[str, float | int]],
    ) -> list[str]:
        """Select weak areas deterministically using current signal and historical metadata."""
        normalized_candidates: list[str] = []
        for item in current_weak:
            normalized = LangGraphInterviewFlow._normalize_weak_area_key(item)
            if normalized:
                normalized_candidates.append(normalized)

        normalized_current: list[str] = []
        seen: set[str] = set()
        for concept in normalized_candidates:
            cleaned = concept.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                normalized_current.append(cleaned)

        if normalized_current:
            ranked = sorted(
                normalized_current,
                key=lambda concept: (
                    -float(historical_weak.get(concept, {}).get("difficulty", 0.0)),
                    -int(historical_weak.get(concept, {}).get("frequency", 0)),
                    concept,
                ),
            )
            return ranked[:2]

        if not historical_weak:
            return []

        ranked_history = sorted(
            historical_weak.items(),
            key=lambda item: (
                -float(item[1].get("difficulty", 0.0)),
                -int(item[1].get("frequency", 0)),
                item[0],
            ),
        )
        return [ranked_history[0][0]]

    @staticmethod
    def _normalize_weak_area_key(concept: str) -> str:
        """Normalize weak-area text to reduce duplicate or near-duplicate concepts."""
        text = concept.strip().lower()
        if not text:
            return ""

        if "mutable" in text and "object" in text:
            return "mutable object handling"

        tokens = re.findall(r"\b[a-z]+\b", text)
        synonym_map = {
            "objects": "object",
            "handling": "handle",
            "handles": "handle",
            "handled": "handle",
            "cases": "case",
            "edges": "edge",
        }
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "with",
            "for",
            "of",
            "in",
            "to",
            "complex",
        }

        normalized_tokens: list[str] = []
        for token in tokens:
            mapped = synonym_map.get(token, token)
            if mapped in stop_words:
                continue
            if mapped not in normalized_tokens:
                normalized_tokens.append(mapped)

        if not normalized_tokens:
            return text

        return " ".join(normalized_tokens[:4])
