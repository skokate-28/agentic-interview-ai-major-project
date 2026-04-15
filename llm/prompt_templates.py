"""Prompt templates used across agents."""

from __future__ import annotations


def _format_weak_areas(weak_areas: list[str]) -> str:
	"""Render weak areas as a readable comma-separated list."""
	cleaned = [item.strip() for item in weak_areas if item.strip()]
	return ", ".join(cleaned) if cleaned else "None"

RESUME_PARSE_PROMPT = """
You are an expert resume parser.
Extract structured information from the resume text below.
Return strict JSON with exactly these keys:
- skills: list[str]
- projects: list[str]
- experience: list[str]
- achievements: list[str]

Resume text:
{resume_text}
""".strip()

QUESTION_GENERATION_PROMPT = """
Generate one technical interview question.
Skill: {skill}
Difficulty level (1-5): {difficulty}
Return only the question text.
""".strip()


def build_question_prompt(
	skill: str,
	previous_question: str,
	evaluation_summary: str,
	difficulty: int,
	weak_areas: list[str],
) -> str:
	"""Build a strict prompt for adaptive next-question generation."""
	previous_text = previous_question.strip() or "None"
	summary_text = evaluation_summary.strip() or "No prior evaluation summary provided."
	weak_areas_text = _format_weak_areas(weak_areas)

	return f"""
You are an expert technical interviewer.

Skill: {skill}

Previous Question:
{previous_text}

Evaluation Summary:
{summary_text}

Weak Areas:
{weak_areas_text}

Difficulty Level: {difficulty} (1-5)

Instructions:
- Ask ONE new question only
- Do NOT repeat previous questions
- Stay strictly within the skill
- Focus on ONE concept
- Match difficulty level strictly:
	1 = definition
	2 = explanation
	3 = comparison
	4 = scenario
	5 = edge case/problem solving
- Follow adaptation logic:
	- If answer was correct, increase challenge
	- If answer was partial, keep same level and target weak areas
	- If answer was wrong, simplify the question

Return ONLY the question text.
""".strip()

ANSWER_EVALUATION_PROMPT = """
Evaluate the candidate answer for the given skill.
Return strict JSON with keys:
- score: float between 0 and 1
- weak_areas: list[str]

Skill: {skill}
Question: {question}
Answer: {answer}
""".strip()


def build_evaluation_prompt(question: str, answer: str, skill: str) -> str:
		"""Build a strict JSON-only evaluation prompt for 7-parameter scoring."""
		question_text = question.strip() or "N/A"
		answer_text = answer.strip() or ""
		skill_text = skill.strip() or "General"

		return f"""
You are an expert evaluator.

Skill: {skill_text}

Question:
{question_text}

Candidate Answer:
{answer_text}

Evaluate using these parameters (0 to 1):

- correctness
- concept_coverage
- depth
- clarity
- confidence (penalize words like maybe, guess)
- error_penalty (higher = more errors)
- overall_reasoning

Return STRICT JSON:

{{
	"correctness": 0-1,
	"concept_coverage": 0-1,
	"depth": 0-1,
	"clarity": 0-1,
	"confidence": 0-1,
	"error_penalty": 0-1,
	"overall_reasoning": 0-1,
	"weak_areas": ["concept1", "concept2"],
	"final_score": 0-1,
	"summary": "short explanation"
}}

Do NOT return anything outside JSON.
""".strip()

HR_EVALUATION_PROMPT = """
Evaluate this HR response for role-based soft skills.
Return strict JSON with keys:
- leadership: float between 0 and 1
- problem_solving: float between 0 and 1
- adaptability: float between 0 and 1
- teamwork: float between 0 and 1
- summary: string

Question: {question}
Answer: {answer}
""".strip()


def build_hr_evaluation_prompt(question: str, answer: str) -> str:
	"""Build a strict JSON-only HR evaluation prompt for soft skill scoring."""
	question_text = question.strip() or "N/A"
	answer_text = answer.strip() or ""

	return f"""
You are an expert HR evaluator.

Question:
{question_text}

Candidate Answer:
{answer_text}

Evaluate the candidate on the following (0 to 1):

- leadership (initiative, ownership)
- problem_solving (logical thinking)
	- adaptability (learning agility, handling change)
	- teamwork (collaboration, empathy, alignment)

Return STRICT JSON:

{{
	"leadership": 0-1,
	"problem_solving": 0-1,
	"adaptability": 0-1,
	"teamwork": 0-1,
	"summary": "short explanation"
}}

Do NOT return anything else.
""".strip()


def build_behavioral_prompt(full_transcript: str) -> str:
	"""Build strict JSON prompt for full-transcript behavioral evaluation."""
	transcript = full_transcript.strip() or "-"

	return f"""
You are an expert interview behavioral evaluator.

Evaluate the FULL interview transcript.

Transcript:
{transcript}

Score from 0 to 1 using these rules:

Communication:
- clarity
- structure
- coherence
- conciseness

Confidence:
- weak word ratio (normalized over the full transcript)
- assertiveness
- consistency across answers

Important constraints:
- Do NOT over-penalize a single weak word
- Evaluate globally across the entire transcript

Return STRICT JSON:

{{
	"communication": 0-1,
	"confidence": 0-1,
	"behavioral_summary": "short explanation"
}}

Do NOT return anything outside JSON.
""".strip()


def build_communication_prompt(responses: list[str]) -> str:
	"""Backward-compatible wrapper for behavioral prompt creation."""
	transcript = " ".join(response.strip() for response in responses if response.strip())
	return build_behavioral_prompt(transcript)
