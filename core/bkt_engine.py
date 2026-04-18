"""Bayesian Knowledge Tracing (BKT) model skeleton."""

from __future__ import annotations

import math


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a float value to an inclusive range."""
    return max(low, min(high, value))


def update_bkt_probability(
    p: float,
    s: float,
    n: int,
    k: float = 0.6,
    alpha: float = 1.5,
    tau: float = 2.0,
) -> float:
    """Update mastery probability with confidence-scaled score responsiveness."""
    # Keep probability and score in valid bounds.
    p_old = _clamp(float(p), low=0.01, high=0.99)
    score = _clamp(float(s))
    attempts = max(int(n), 1)

    # Confidence increases smoothly with question count; early updates stay conservative.
    confidence_factor = 1.0 - math.exp(-attempts / float(tau))

    # Update strength scales with distance between score and current probability.
    score_gap = score - p_old
    delta = float(k) * score_gap * (1.0 + float(alpha) * abs(score_gap)) * confidence_factor

    # Keep probability away from hard 0/1 lock-in.
    p_next = _clamp(p_old + delta, low=0.01, high=0.99)
    return p_next


class BKTModel:
    """Simple BKT-style probability updater."""

    def __init__(
        self,
        initial_probability: float = 0.0,
        k: float = 0.6,
        alpha: float = 1.5,
        tau: float = 2.0,
    ) -> None:
        self.current_probability = _clamp(initial_probability, low=0.01, high=0.99)
        self.k = k
        self.alpha = alpha
        self.tau = tau

    def update(self, score: float, n: int) -> float:
        """Update mastery probability using score-distance and confidence growth by attempts."""
        p_old = _clamp(self.current_probability, low=0.01, high=0.99)
        bounded_score = _clamp(score)
        attempts = max(int(n), 1)

        confidence_factor = 1.0 - math.exp(-attempts / float(self.tau))
        score_gap = bounded_score - p_old
        delta = (
            float(self.k)
            * score_gap
            * (1.0 + float(self.alpha) * abs(score_gap))
            * confidence_factor
        )
        p_new = update_bkt_probability(
            p=p_old,
            s=bounded_score,
            n=attempts,
            k=self.k,
            alpha=self.alpha,
            tau=self.tau,
        )

        self.current_probability = p_new
        print(
            "[BKT UPDATE DEBUG] "
            f"p_old={p_old:.4f}, s={bounded_score:.4f}, "
            f"n={attempts}, confidence={confidence_factor:.4f}, "
            f"delta={delta:.4f}, p_new={p_new:.4f}"
        )
        return self.current_probability
