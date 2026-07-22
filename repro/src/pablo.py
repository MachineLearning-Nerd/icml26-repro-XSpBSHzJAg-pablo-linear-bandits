"""Numerical primitives for PABLO (arXiv:2603.28201, Algorithm 1).

The implementation deliberately exposes the full finite perturbation support.
That makes the estimator identities in Proposition 2.1 testable to floating-
point precision instead of relying on a noisy Monte Carlo approximation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PerturbationOutcome:
    coordinate: int
    sign: int
    action: np.ndarray
    feedback: float
    estimate: np.ndarray
    eigenvalue: float


def _symmetric_factors(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return eigenvalues/vectors and symmetric square-root factors of an SPD matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if np.any(eigenvalues <= 0):
        raise ValueError("H must be positive definite")
    sqrt_h = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    inv_sqrt_h = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
    return eigenvalues, eigenvectors, sqrt_h, inv_sqrt_h


def enumerate_outcomes(w: np.ndarray, loss: np.ndarray, h: np.ndarray) -> list[PerturbationOutcome]:
    """Enumerate all 2d equiprobable outcomes in Algorithm 1."""
    d = len(w)
    eigenvalues, eigenvectors, sqrt_h, inv_sqrt_h = _symmetric_factors(h)
    outcomes: list[PerturbationOutcome] = []
    for coordinate in range(d):
        for sign in (-1, 1):
            s = sign * eigenvectors[:, coordinate]
            action = w + inv_sqrt_h @ s
            feedback = float(loss @ action)
            estimate = d * (sqrt_h @ s) * feedback
            outcomes.append(
                PerturbationOutcome(
                    coordinate=coordinate,
                    sign=sign,
                    action=action,
                    feedback=feedback,
                    estimate=estimate,
                    eigenvalue=float(eigenvalues[coordinate]),
                )
            )
    return outcomes


def proposition_21_quantities(w: np.ndarray, loss: np.ndarray, h: np.ndarray) -> dict[str, float]:
    """Compute both sides of every identity/bound in Proposition 2.1 exactly."""
    outcomes = enumerate_outcomes(w, loss, h)
    estimates = np.stack([outcome.estimate for outcome in outcomes])
    mean_estimate = estimates.mean(axis=0)
    observed_second_moment = float(np.mean(np.sum(estimates * estimates, axis=1)))
    predicted_second_moment = float(
        len(w) * (loss @ loss) + len(w) * (loss @ w) ** 2 * np.trace(h)
    )

    bound_ratios = []
    for outcome in outcomes:
        lhs = float(outcome.estimate @ outcome.estimate)
        rhs = float(
            len(w) ** 2
            * (loss @ loss)
            * (np.sqrt(outcome.eigenvalue) * np.linalg.norm(w) + 1.0) ** 2
        )
        bound_ratios.append(lhs / rhs if rhs > 0 else 0.0)

    return {
        "bias_l2": float(np.linalg.norm(mean_estimate - loss)),
        "second_moment_observed": observed_second_moment,
        "second_moment_predicted": predicted_second_moment,
        "second_moment_relative_error": float(
            abs(observed_second_moment - predicted_second_moment)
            / max(predicted_second_moment, np.finfo(float).tiny)
        ),
        "max_almost_sure_bound_ratio": float(max(bound_ratios)),
    }


def pablo_ogd(
    losses: np.ndarray,
    eta: float,
    seed: int,
    epsilon: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PABLO with isotropic Eq. (4) perturbations and Euclidean OGD.

    This is used only to instantiate Proposition 2.3 with a subroutine whose
    deterministic regret certificate is elementary and independently known.
    """
    rng = np.random.default_rng(seed)
    t_horizon, d = losses.shape
    w = np.zeros(d)
    centers = np.zeros((t_horizon, d))
    played = np.zeros((t_horizon, d))
    estimates = np.zeros((t_horizon, d))
    for t, loss in enumerate(losses):
        centers[t] = w
        coordinate = int(rng.integers(d))
        sign = -1 if int(rng.integers(2)) == 0 else 1
        s = np.zeros(d)
        s[coordinate] = sign
        scale_sq = max(float(w @ w), epsilon**2)
        h_scalar = 1.0 / (d * scale_sq)
        played[t] = w + s / np.sqrt(h_scalar)
        feedback = float(loss @ played[t])
        estimates[t] = d * np.sqrt(h_scalar) * s * feedback
        w = w - eta * estimates[t]
    return centers, played, estimates


def ogd_static_certificate(comparator: np.ndarray, gradients: np.ndarray, eta: float) -> float:
    """Standard OGD certificate ||u||^2/(2 eta) + eta/2 sum ||g_t||^2."""
    return float(
        (comparator @ comparator) / (2.0 * eta)
        + 0.5 * eta * np.sum(gradients * gradients)
    )
