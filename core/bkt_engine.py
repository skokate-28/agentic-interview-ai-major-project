"""Bayesian Knowledge Tracing (BKT) model skeleton."""

from __future__ import annotations

import math


BETA = 0.75
K = 0.5
GAMMA = 0.4
RHO = 0.825


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a float value to an inclusive range."""
    return max(low, min(high, value))


def update_bkt_probability(
    p: float,
    s: float,
    n: int,
    sum_s: float,
    beta: float = BETA,
    k: float = K,
    gamma: float = GAMMA,
    rho: float = RHO,
) -> float:
    """Update probability using the cumulative-score formula."""
    p_old = _clamp(float(p), low=0.0, high=1.0)
    score = _clamp(float(s))
    attempts = max(int(n), 1)
    running_sum = float(sum_s)
    mean_s = running_sum / attempts

    alpha = 1.0 / (1.0 + float(beta) * attempts)
    error = score - p_old
    bounded_error = math.tanh(float(k) * error)
    extremeness = abs(2.0 * score - 1.0)
    extreme_factor = 1.0 + float(gamma) * extremeness
    consistency_base = mean_s - 0.5
    activation = attempts / (attempts + 1.0)
    consistency_factor = 1.0 + float(rho) * consistency_base * activation

    delta = alpha * bounded_error * extreme_factor * consistency_factor
    p_next = p_old + delta
    p_next = _clamp(p_next, low=0.0, high=1.0)
    return p_next


class BKTModel:
    """Simple BKT-style probability updater."""

    def __init__(
        self,
        initial_probability: float = 0.5,
        k: float = K,
        alpha: float = 0.0,
        tau: float = 2.0,
    ) -> None:
        self.current_probability = _clamp(initial_probability, low=0.0, high=1.0)
        self.k = float(k)
        self.alpha = alpha
        self.tau = tau
        self.n = 0
        self.sum_s = 0.0

    def update(self, score: float, n: int) -> float:
        """Update probability in-place using cumulative score state."""
        p_old = _clamp(self.current_probability, low=0.0, high=1.0)
        bounded_score = _clamp(score)
        attempts = max(int(n), 1)
        self.n = attempts
        self.sum_s = float(self.sum_s) + bounded_score
        mean_s = self.sum_s / self.n

        alpha = 1.0 / (1.0 + BETA * self.n)
        error = bounded_score - p_old
        bounded_error = math.tanh(self.k * error)
        extremeness = abs(2.0 * bounded_score - 1.0)
        extreme_factor = 1.0 + GAMMA * extremeness
        consistency_base = mean_s - 0.5
        activation = self.n / (self.n + 1.0)
        consistency_factor = 1.0 + RHO * consistency_base * activation
        delta = alpha * bounded_error * extreme_factor * consistency_factor

        p_new = update_bkt_probability(
            p=p_old,
            s=bounded_score,
            n=attempts,
            sum_s=self.sum_s,
            beta=BETA,
            k=self.k,
            gamma=GAMMA,
            rho=RHO,
        )

        self.current_probability = p_new
        print(
            "[BKT UPDATE DEBUG] "
            f"p_old={p_old:.4f}, s={bounded_score:.4f}, n={attempts}, "
            f"mean_s={mean_s:.4f}, delta={delta:.4f}, p_new={p_new:.4f}"
        )
        return self.current_probability
