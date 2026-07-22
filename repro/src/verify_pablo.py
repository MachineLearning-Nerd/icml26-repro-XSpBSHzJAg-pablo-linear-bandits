"""Evidence-first checks for PABLO's finite-support estimator and reduction.

Every reported assessment is tied to a paper statement. The script prints a
machine-readable summary because local OpenResearch run logs are the evidence
channel used to compare experiment branches.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import pablo


SEED = 260328201


def random_spd(rng: np.random.Generator, d: int, max_eigenvalue: float | None = None) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    upper = 2.0 if max_eigenvalue is None else max_eigenvalue
    lower = max(upper * 1e-3, np.finfo(float).eps)
    eigenvalues = np.exp(rng.uniform(np.log(lower), np.log(upper), size=d))
    return (q * eigenvalues) @ q.T


def check_proposition_21() -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    max_bias = 0.0
    max_second_error = 0.0
    max_bound_ratio = 0.0
    instances = 0
    for d in (1, 2, 4, 8, 16):
        for _ in range(16):
            w = rng.normal(size=d)
            loss = rng.normal(size=d)
            h = random_spd(rng, d)
            values = pablo.proposition_21_quantities(w, loss, h)
            max_bias = max(max_bias, float(values["bias_l2"]))
            max_second_error = max(max_second_error, float(values["second_moment_relative_error"]))
            max_bound_ratio = max(max_bound_ratio, float(values["max_almost_sure_bound_ratio"]))
            instances += 1
    aligned = max_bias < 1e-10 and max_second_error < 1e-10 and max_bound_ratio <= 1.0 + 1e-12
    return {
        "paper_claim": "Proposition 2.1: unbiased estimator, exact conditional second moment, and an almost-sure norm bound.",
        "observed": {
            "instances_exhaustively_enumerated": instances,
            "max_bias_l2": max_bias,
            "max_second_moment_relative_error": max_second_error,
            "max_almost_sure_bound_ratio": max_bound_ratio,
        },
        "assessment": "aligned" if aligned else "not aligned in this finite check",
    }


def check_corollary_22() -> dict[str, object]:
    rng = np.random.default_rng(SEED + 1)
    epsilon = 0.1
    max_second_ratio = 0.0
    max_almost_sure_ratio = 0.0
    instances = 0
    for d in (1, 2, 4, 8, 16, 32):
        for _ in range(16):
            w = rng.normal(size=d)
            loss = rng.normal(size=d)
            cap = 1.0 / (d * max(float(w @ w), epsilon**2))
            h = random_spd(rng, d, max_eigenvalue=cap)
            outcomes = pablo.enumerate_outcomes(w, loss, h)
            squared_norms = np.array([outcome.estimate @ outcome.estimate for outcome in outcomes])
            loss_sq = float(loss @ loss)
            max_second_ratio = max(
                max_second_ratio,
                float(squared_norms.mean() / (2.0 * d * loss_sq)),
            )
            max_almost_sure_ratio = max(
                max_almost_sure_ratio,
                float(squared_norms.max() / (4.0 * d**2 * loss_sq)),
            )
            instances += 1
    aligned = max_second_ratio <= 1.0 + 1e-12 and max_almost_sure_ratio <= 1.0 + 1e-12
    return {
        "paper_claim": "Corollary 2.2: Eq. (4)-compatible H controls the estimator's conditional second moment and support.",
        "observed": {
            "instances_exhaustively_enumerated": instances,
            "max_ratio_to_second_moment_bound": max_second_ratio,
            "max_ratio_to_almost_sure_bound": max_almost_sure_ratio,
        },
        "assessment": "aligned" if aligned else "not aligned in this finite check",
    }


def check_proposition_23() -> dict[str, object]:
    rng = np.random.default_rng(SEED + 2)
    d, horizon, repetitions = 5, 250, 1200
    eta = 0.01

    # Fixed before the learner's perturbations: an oblivious, bounded sequence.
    losses = rng.normal(size=(horizon, d))
    losses /= np.linalg.norm(losses, axis=1, keepdims=True)
    losses = 0.35 * losses + 0.05 * np.eye(1, d, 0).repeat(horizon, axis=0)
    losses /= np.maximum(1.0, np.linalg.norm(losses, axis=1, keepdims=True))
    cumulative_loss = losses.sum(axis=0)
    comparator = -cumulative_loss / np.linalg.norm(cumulative_loss)

    margins = np.empty(repetitions)
    lhs_values = np.empty(repetitions)
    rhs_values = np.empty(repetitions)
    for repetition in range(repetitions):
        centers, played, estimates = pablo.pablo_ogd(
            losses=losses,
            eta=eta,
            seed=SEED + 10_000 + repetition,
        )
        deltas = losses - estimates
        lhs = float(np.sum(losses * (played - comparator)))
        rhs = pablo.ogd_static_certificate(comparator, estimates, eta)
        rhs += pablo.ogd_static_certificate(comparator, deltas, eta)
        lhs_values[repetition] = lhs
        rhs_values[repetition] = rhs
        margins[repetition] = rhs - lhs

    mean_margin = float(margins.mean())
    standard_error = float(margins.std(ddof=1) / np.sqrt(repetitions))
    lower_95 = mean_margin - 1.96 * standard_error
    aligned = lower_95 > 0.0
    return {
        "paper_claim": "Proposition 2.3: expected bandit regret is controlled by two OLO certificates.",
        "observed": {
            "d": d,
            "T": horizon,
            "perturbation_repetitions": repetitions,
            "mean_bandit_regret": float(lhs_values.mean()),
            "mean_two_certificate_bound": float(rhs_values.mean()),
            "mean_bound_minus_regret": mean_margin,
            "margin_standard_error": standard_error,
            "margin_95pct_lower": lower_95,
        },
        "assessment": "aligned" if aligned else "inconclusive under this setup",
        "scope": "Finite Monte Carlo instantiation with Euclidean OGD; this checks the reduction, not the PFMD rates in Theorem 3.1.",
    }


def check_dimension_gap_mechanism() -> dict[str, object]:
    dimensions = np.array([2, 4, 8, 16, 32, 64, 128], dtype=float)
    rms_values = []
    support_values = []
    epsilon = 0.1
    for dimension_value in dimensions:
        d = int(dimension_value)
        w = np.zeros(d)
        loss = np.zeros(d)
        loss[0] = 1.0
        h = np.eye(d) / (d * epsilon**2)
        outcomes = pablo.enumerate_outcomes(w, loss, h)
        norms = np.array([np.linalg.norm(outcome.estimate) for outcome in outcomes])
        rms_values.append(float(np.sqrt(np.mean(norms**2))))
        support_values.append(float(norms.max()))

    rms_slope, _ = np.polyfit(np.log(dimensions), np.log(rms_values), 1)
    support_slope, _ = np.polyfit(np.log(dimensions), np.log(support_values), 1)
    max_rms_normalized_error = float(
        np.max(np.abs(np.array(rms_values) / np.sqrt(dimensions) - 1.0))
    )
    max_support_normalized_error = float(
        np.max(np.abs(np.array(support_values) / dimensions - 1.0))
    )
    aligned = (
        abs(float(rms_slope) - 0.5) < 1e-10
        and abs(float(support_slope) - 1.0) < 1e-10
        and max_rms_normalized_error < 1e-12
        and max_support_normalized_error < 1e-12
    )
    return {
        "paper_claim": "Theorem 3.1 mechanism: second-moment control contributes sqrt(d), while trajectory-coupled control uses a worst-case d scale.",
        "observed": {
            "dimensions": [int(value) for value in dimensions],
            "conditional_rms": rms_values,
            "support_maximum": support_values,
            "conditional_rms_loglog_slope": float(rms_slope),
            "support_maximum_loglog_slope": float(support_slope),
            "max_rms_over_sqrt_d_error": max_rms_normalized_error,
            "max_support_over_d_error": max_support_normalized_error,
        },
        "assessment": "aligned" if aligned else "not aligned in this controlled family",
        "scope": "Tests the estimator-moment mechanism behind the sqrt(d) separation, not the full PFMD theorem with hidden polylogarithmic constants.",
    }


def main() -> None:
    results = {
        "paper": "A Perturbation Approach to Unconstrained Linear Bandits",
        "arxiv": "2603.28201",
        "seed": SEED,
        "compute": "local CPU",
        "claims": {
            "proposition_2_1": check_proposition_21(),
            "corollary_2_2": check_corollary_22(),
            "proposition_2_3": check_proposition_23(),
            "theorem_3_1_dimension_mechanism": check_dimension_gap_mechanism(),
        },
    }
    print("PABLO EXACT-IDENTITY AND REDUCTION CHECK")
    for key, claim in results["claims"].items():
        print(f"{key}: {claim['assessment']}")
        print(json.dumps(claim["observed"], sort_keys=True))
    print("RESULT_JSON=" + json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
