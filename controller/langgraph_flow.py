"""LangGraph-based orchestration flow for the multi-agent interview pipeline."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, TypedDict

from agents.evaluation_agent import EvaluationAgent
from agents.hr_agent import HRAgent
from agents.question_agent import QuestionAgent
from config import settings
from core.bkt_engine import BKTModel
from core.cat_engine import CATEngine
from core.skill_graph import SkillGraph
from database.cache_manager import CacheManager
from database.db_manager import DatabaseManager
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

    weak_areas: List[str]
    selected_weak_areas: List[str]

    question_count: int
    max_questions: int
    done: bool

    skill_results: Dict[str, Any]
    all_responses: List[str]

    hr_results: Dict[str, Any]
    behavioral_scores: Dict[str, Any]

    bkt_model: Any
    cat_engine: Any
    skill_graph: Any


class LangGraphInterviewFlow:
    """Graph-driven interview orchestrator preserving adaptive behavior and user isolation."""

    def __init__(
        self,
        user_id: str,
        resume_data: Dict[str, list[str]] | None = None,
        db_manager: DatabaseManager | None = None,
        cache_manager: CacheManager | None = None,
    ) -> None:
        self.user_id = user_id
        self.resume_data = resume_data or {
            "skills": [],
            "projects": [],
            "experience": [],
            "achievements": [],
        }
        self.db_manager = db_manager or DatabaseManager()
        self.cache_manager = cache_manager or CacheManager()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.question_agent = QuestionAgent()
        self.evaluation_agent = EvaluationAgent()
        self.hr_agent = HRAgent()

        self._state_graph_cls = get_state_graph()

        self.skill_graphs: Dict[str, SkillGraph] = {}
        self.bkt_models: Dict[str, BKTModel] = {}
        self.cat_engines: Dict[str, CATEngine] = {}
        self.all_responses: list[str] = []

        self.skill_app = self._build_skill_graph().compile()
        self.main_app = self._build_main_graph().compile()

    def load_user_profile(self) -> None:
        """Load persisted resume profile for the current user if available."""
        existing_profile = self.db_manager.get_candidate(self.user_id)
        if existing_profile is not None:
            self.resume_data = existing_profile.resume_data  # type: ignore[assignment]

    def initialize_skill_graphs(self, skills: list[str]) -> None:
        """Initialize adaptive engine holders for each skill."""
        for skill in skills:
            self.skill_graphs[skill] = SkillGraph(skill_name=skill)
            self.bkt_models[skill] = BKTModel()
            self.cat_engines[skill] = CATEngine()

    def start_interview(self) -> Dict[str, object]:
        """Run full interview flow: technical skill loop, HR, then behavioral analysis."""
        self.all_responses = []
        self.load_user_profile()
        self.db_manager.upsert_candidate(self.user_id, self.resume_data)

        skills = self.resume_data.get("skills", [])
        self.initialize_skill_graphs(skills)

        initial_state: InterviewState = {
            "user_id": self.user_id,
            "skills": skills,
            "skill_results": {},
            "all_responses": [],
            "done": False,
        }

        final_state = self.main_app.invoke(initial_state)
        return {
            "technical_round": final_state.get("skill_results", {}),
            "hr_round": final_state.get("hr_results", {}),
            "behavioral_scores": final_state.get("behavioral_scores", {}),
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
                db_weak_areas = self.db_manager.get_skill_weak_areas(self.user_id, skill)
                if db_weak_areas:
                    skill_graph.weak_areas = db_weak_areas

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
                "weak_areas": [],
                "selected_weak_areas": [],
                "question_count": 0,
                "max_questions": max_questions,
                "done": False,
                "skill_results": results,
                "all_responses": current_responses,
                "bkt_model": bkt,
                "cat_engine": cat,
                "skill_graph": skill_graph,
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
                [str(response).strip() for response in responses if str(response).strip()]
            )
        return hr_results

    def question_node(self, state: InterviewState) -> InterviewState:
        """Generate next technical question for current skill and CAT difficulty."""
        skill = state["current_skill"]
        question = self.question_agent.generate_question(
            skill=skill,
            previous_question=state.get("question", ""),
            evaluation_summary=str(state.get("evaluation", {}).get("summary", "")),
            difficulty=int(state.get("difficulty", 3)),
            weak_areas=list(state.get("selected_weak_areas", [])),
        )

        state["question"] = question
        return state

    def answer_node(self, state: InterviewState) -> InterviewState:
        """Collect candidate answer from terminal and add to global response list."""
        skill = state["current_skill"]
        question = state.get("question", "")
        answer = self._collect_answer(skill=skill, question=question)

        if "all_responses" not in state:
            state["all_responses"] = []
        state["all_responses"].append(answer)

        state["answer"] = answer
        return state

    def evaluation_node(self, state: InterviewState) -> InterviewState:
        """Evaluate the candidate answer for the current question."""
        skill = state["current_skill"]
        question = state.get("question", "")
        answer = state.get("answer", "")

        evaluation = self.evaluation_agent.evaluate_answer(
            question=question,
            answer=answer,
            skill=skill,
        )

        score = self._safe_score(evaluation.get("final_score", 0.0))
        current_weak_areas = self._safe_weak_areas(evaluation.get("weak_areas", []))
        historical_weak_areas = state["skill_graph"].weak_areas
        selected_weak_areas = self._select_weak_areas(current_weak_areas, historical_weak_areas)

        state["evaluation"] = evaluation
        state["score"] = score
        state["weak_areas"] = current_weak_areas
        state["selected_weak_areas"] = selected_weak_areas
        state["question_count"] = int(state.get("question_count", 0)) + 1
        return state

    def bkt_node(self, state: InterviewState) -> InterviewState:
        """Update BKT mastery probability for current skill."""
        bkt: BKTModel = state["bkt_model"]

        state["prev_probability"] = float(state.get("probability", bkt.current_probability))
        new_probability = bkt.update(float(state.get("score", 0.0)))

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

        skill_graph.update(
            score=score,
            probability=probability,
            weak_areas=selected_weak_areas,
            question_count=question_count,
            max_questions=max_questions,
        )

        self.cache_manager.set_weak_areas(self.user_id, skill, skill_graph.weak_areas)
        self.db_manager.save_skill_weak_areas(self.user_id, skill, skill_graph.weak_areas)
        self.db_manager.save_skill_progress(
            self.user_id,
            skill,
            skill_graph.proficiency,
            skill_graph.confidence,
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
                [str(answer).strip() for answer in hr_answers if str(answer).strip()]
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

    @staticmethod
    def _collect_answer(skill: str, question: str) -> str:
        """Collect candidate answer from terminal input safely."""
        try:
            return input(f"\n{skill} Question: {question}\nYour Answer: ").strip()
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
    def _select_weak_areas(
        current_weak: list[str],
        historical_weak: dict[str, dict[str, float | int]],
    ) -> list[str]:
        """Select weak areas deterministically using current signal and historical metadata."""
        normalized_current = []
        seen: set[str] = set()
        for concept in current_weak:
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
