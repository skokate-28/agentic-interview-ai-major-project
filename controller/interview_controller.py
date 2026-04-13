"""Compatibility wrapper exposing the LangGraph-based interview workflow."""

from __future__ import annotations

from typing import Dict

from database.cache_manager import CacheManager
from database.db_manager import DatabaseManager

from controller.langgraph_flow import LangGraphInterviewFlow


class InterviewController:
    """Facade over the LangGraph interview flow for backward-compatible usage."""

    def __init__(
        self,
        user_id: str,
        resume_data: Dict[str, list[str]] | None = None,
        db_manager: DatabaseManager | None = None,
        cache_manager: CacheManager | None = None,
    ) -> None:
        self.flow = LangGraphInterviewFlow(
            user_id=user_id,
            resume_data=resume_data,
            db_manager=db_manager,
            cache_manager=cache_manager,
        )
        self._sync_public_refs()

    def _sync_public_refs(self) -> None:
        """Expose commonly used flow internals for compatibility."""
        self.user_id = self.flow.user_id
        self.resume_data = self.flow.resume_data
        self.db_manager = self.flow.db_manager
        self.cache_manager = self.flow.cache_manager
        self.question_agent = self.flow.question_agent
        self.evaluation_agent = self.flow.evaluation_agent
        self.hr_agent = self.flow.hr_agent
        self.skill_graphs = self.flow.skill_graphs
        self.bkt_models = self.flow.bkt_models
        self.cat_engines = self.flow.cat_engines
        self.all_responses = self.flow.all_responses

    def load_user_profile(self) -> None:
        """Load persisted resume profile for this user."""
        self.flow.load_user_profile()
        self._sync_public_refs()

    def initialize_skill_graphs(self, skills: list[str]) -> None:
        """Initialize adaptive model holders for provided skills."""
        self.flow.initialize_skill_graphs(skills)
        self._sync_public_refs()

    def run_technical_round(self, skills: list[str]) -> Dict[str, Dict[str, object]]:
        """Run graph-based technical round."""
        result = self.flow.run_technical_round(skills)
        self._sync_public_refs()
        return result

    def run_hr_round(self) -> Dict[str, object]:
        """Run graph-compatible HR round."""
        result = self.flow.run_hr_round()
        self._sync_public_refs()
        return result

    def start_interview(self) -> Dict[str, object]:
        """Run full LangGraph interview pipeline."""
        result = self.flow.start_interview()
        self._sync_public_refs()
        return result
