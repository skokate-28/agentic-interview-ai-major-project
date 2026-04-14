"""Main entry point for the interview system foundation."""

from __future__ import annotations

import json
import logging

from agents.resume_parser_agent import ResumeParserAgent
from controller.interview_controller import InterviewController


DUMMY_RESUME_TEXT = """
John Doe
Email: john.doe@example.com

Skills: Python, SQL

Projects:
1) Adaptive Quiz Engine for student assessment
2) Flask-based Task Automation API

Experience:
Software Engineer at ExampleTech (2022-Present)

Achievements:
- Winner, National Hackathon 2024
- Employee of the Quarter (Q3 2025)
""".strip()


def main() -> None:
    """Run technical and HR interview rounds and print consolidated results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser_agent = ResumeParserAgent()
    parsed_resume = parser_agent.parse_resume(DUMMY_RESUME_TEXT)

    controller = InterviewController(
        user_id="demo_candidate",
        resume_data=parsed_resume,
    )
    interview_results = controller.start_interview()
    technical_results = interview_results.get("technical_round", {})
    hr_results = interview_results.get("hr_round", {})
    behavioral_scores = interview_results.get("behavioral_scores", {})
    final_report = interview_results.get("final_report", {})

    print("\nTechnical Results:")
    print(json.dumps(technical_results, indent=2))

    print("\nHR Results:")
    print(json.dumps(hr_results, indent=2))

    print("\nBehavioral Scores:")
    print(json.dumps(behavioral_scores, indent=2))

    print("\nFinal Report:")
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()