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


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return np.zeros_like(vector) if norm == 0.0 else vector / norm


class ParameterFreeMirrorDescent:
    """Jacobsen--Cutkosky (2022), Algorithm 4.

    This is the exact unconstrained closed-form update cited by Theorem 3.1.
    ``gradient_bound`` is the algorithm's required bound G and ``epsilon`` is
    its parameter-free origin-regret parameter.
    """

    def __init__(self, dimension: int, gradient_bound: float, epsilon: float) -> None:
        if gradient_bound <= 0.0 or epsilon <= 0.0:
            raise ValueError("gradient_bound and epsilon must be positive")
        self.gradient_bound = float(gradient_bound)
        self.epsilon = float(epsilon)
        self.variance = 4.0 * self.gradient_bound**2
        self.theta = np.zeros(dimension)
        self.w = np.zeros(dimension)

    def decision(self) -> np.ndarray:
        return self.w.copy()

    def update(self, gradient: np.ndarray) -> None:
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm > self.gradient_bound * (1.0 + 1e-10):
            raise ValueError(
                f"gradient norm {gradient_norm} exceeds bound {self.gradient_bound}"
            )
        self.theta -= gradient
        self.variance += gradient_norm**2
        theta_norm = float(np.linalg.norm(self.theta))
        if theta_norm == 0.0:
            self.w.fill(0.0)
            return

        log_variance = np.log(self.variance / self.gradient_bound**2)
        alpha = (
            self.epsilon
            * self.gradient_bound
            / (np.sqrt(self.variance) * log_variance**2)
        )
        if theta_norm <= 6.0 * self.variance / self.gradient_bound:
            exponent = theta_norm**2 / (36.0 * self.variance)
        else:
            exponent = (
                theta_norm / (3.0 * self.gradient_bound)
                - self.variance / self.gradient_bound**2
            )
        if exponent > 700.0:
            raise FloatingPointError("PFMD exponent exceeds floating-point range")
        self.w = alpha * _unit(self.theta) * np.expm1(exponent)


def pfmd_static_certificate(
    comparator: np.ndarray,
    gradients: np.ndarray,
    gradient_bound: float,
    epsilon: float,
) -> float:
    """Explicit finite-horizon certificate from PFMD Theorem 1 (k=3)."""
    variance = 4.0 * gradient_bound**2 + float(np.sum(gradients * gradients))
    log_variance = np.log(variance / gradient_bound**2)
    alpha = epsilon * gradient_bound / (np.sqrt(variance) * log_variance**2)
    comparator_norm = float(np.linalg.norm(comparator))
    log_term = np.log1p(comparator_norm / alpha) + 1.0
    adaptive_term = max(
        np.sqrt(variance * log_term),
        gradient_bound * log_term,
    )
    return float(4.0 * gradient_bound * epsilon + 6.0 * comparator_norm * adaptive_term)


def pablo_pfmd(
    losses: np.ndarray,
    seed: int,
    epsilon: float = 0.2,
    perturbation_floor: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """PABLO using the PFMD subroutine named in Theorem 3.1."""
    rng = np.random.default_rng(seed)
    horizon, dimension = losses.shape
    loss_bound = float(np.max(np.linalg.norm(losses, axis=1)))
    # One bound handles both estimator gradients and the ghost deltas in
    # Proposition 2.3: ||ell-hat|| <= (2d+1)G under Corollary 2.2.
    olo_bound = (2.0 * dimension + 1.0) * loss_bound
    learner = ParameterFreeMirrorDescent(
        dimension=dimension,
        gradient_bound=olo_bound,
        epsilon=epsilon / dimension,
    )
    centers = np.zeros_like(losses)
    played = np.zeros_like(losses)
    estimates = np.zeros_like(losses)
    for t, loss in enumerate(losses):
        w = learner.decision()
        centers[t] = w
        coordinate = int(rng.integers(dimension))
        sign = -1 if int(rng.integers(2)) == 0 else 1
        s = np.zeros(dimension)
        s[coordinate] = sign
        scale_sq = max(float(w @ w), perturbation_floor**2)
        h_scalar = 1.0 / (dimension * scale_sq)
        played[t] = w + s / np.sqrt(h_scalar)
        feedback = float(loss @ played[t])
        estimates[t] = dimension * np.sqrt(h_scalar) * s * feedback
        learner.update(estimates[t])
    return centers, played, estimates, olo_bound


class RefinedDynamicBase:
    """Unconstrained specialization of the paper's Algorithm 5."""

    def __init__(
        self,
        dimension: int,
        eta: float,
        alpha: float,
        gamma: float,
        k: float = 4.0,
    ) -> None:
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.k = float(k)
        self.w = np.zeros(dimension)

    def decision(self) -> np.ndarray:
        return self.w.copy()

    def update(self, gradient: np.ndarray) -> None:
        w_norm = float(np.linalg.norm(self.w))
        mirror_gradient = (
            np.zeros_like(self.w)
            if w_norm == 0.0
            else (self.k / self.eta) * np.log1p(w_norm / self.alpha) * _unit(self.w)
        )
        theta = mirror_gradient - gradient
        theta_norm = float(np.linalg.norm(theta))
        radial = theta_norm - 0.5 * self.eta * float(gradient @ gradient) - self.gamma
        exponent = self.eta * radial / self.k
        magnitude = 0.0 if exponent <= 0.0 else self.alpha * np.expm1(exponent)
        self.w = _unit(theta) * magnitude


def dynamic_step_sizes(horizon: int, gradient_bound: float) -> list[float]:
    """The distinct grid S in Algorithm 6."""
    values: list[float] = []
    i = 0
    while True:
        eta = min(2.0**i / (horizon * gradient_bound), 1.0 / gradient_bound)
        if not values or eta != values[-1]:
            values.append(float(eta))
        if eta == 1.0 / gradient_bound:
            return values
        i += 1


class RefinedDynamicLearner:
    """The paper's Algorithm 6: sum a logarithmic grid of Algorithm 5 bases."""

    def __init__(
        self,
        dimension: int,
        horizon: int,
        gradient_bound: float,
        epsilon: float,
    ) -> None:
        self.gradient_bound = float(gradient_bound)
        self.epsilon = float(epsilon)
        self.horizon = int(horizon)
        alpha = epsilon / horizon
        gamma = gradient_bound / horizon
        self.bases = [
            RefinedDynamicBase(dimension, eta, alpha, gamma)
            for eta in dynamic_step_sizes(horizon, gradient_bound)
        ]

    def decision(self) -> np.ndarray:
        return np.sum([base.decision() for base in self.bases], axis=0)

    def update(self, gradient: np.ndarray) -> None:
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm > self.gradient_bound * (1.0 + 1e-10):
            raise ValueError(
                f"gradient norm {gradient_norm} exceeds bound {self.gradient_bound}"
            )
        for base in self.bases:
            base.update(gradient)


def phi(value: float, scale: float) -> float:
    return float(value * np.log1p(scale * value))


def dynamic_certificate(
    comparators: np.ndarray,
    gradients: np.ndarray,
    gradient_bound: float,
    epsilon: float,
) -> float:
    """Explicit Algorithm 6 certificate from Theorem E.2."""
    horizon = len(comparators)
    endpoint = phi(float(np.linalg.norm(comparators[-1])), horizon / epsilon)
    path = sum(
        phi(
            float(np.linalg.norm(comparators[t] - comparators[t - 1])),
            4.0 * horizon**3 / epsilon,
        )
        for t in range(1, horizon)
    )
    complexity = endpoint + path
    max_norm = float(np.max(np.linalg.norm(comparators, axis=1)))
    weighted_variance = float(
        np.sum(np.sum(gradients * gradients, axis=1) * np.linalg.norm(comparators, axis=1))
    )
    grid_size = len(dynamic_step_sizes(horizon, gradient_bound))
    return float(
        4.0 * gradient_bound * (grid_size * epsilon + max_norm + complexity)
        + 2.0 * np.sqrt(2.0 * complexity * weighted_variance)
    )


def pablo_dynamic(
    losses: np.ndarray,
    seed: int,
    epsilon: float = 0.2,
    perturbation_floor: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """PABLO using the refined dynamic learner in Theorem 3.3."""
    rng = np.random.default_rng(seed)
    horizon, dimension = losses.shape
    loss_bound = float(np.max(np.linalg.norm(losses, axis=1)))
    olo_bound = (2.0 * dimension + 1.0) * loss_bound
    learner = RefinedDynamicLearner(
        dimension=dimension,
        horizon=horizon,
        gradient_bound=olo_bound,
        epsilon=epsilon / dimension,
    )
    centers = np.zeros_like(losses)
    played = np.zeros_like(losses)
    estimates = np.zeros_like(losses)
    for t, loss in enumerate(losses):
        w = learner.decision()
        centers[t] = w
        coordinate = int(rng.integers(dimension))
        sign = -1 if int(rng.integers(2)) == 0 else 1
        s = np.zeros(dimension)
        s[coordinate] = sign
        scale_sq = max(float(w @ w), perturbation_floor**2)
        h_scalar = 1.0 / (dimension * scale_sq)
        played[t] = w + s / np.sqrt(h_scalar)
        feedback = float(loss @ played[t])
        estimates[t] = dimension * np.sqrt(h_scalar) * s * feedback
        learner.update(estimates[t])
    return centers, played, estimates, olo_bound
