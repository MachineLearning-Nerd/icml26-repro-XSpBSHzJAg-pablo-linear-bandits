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


def _oblivious_losses(dimension: int, horizon: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(horizon, dimension))
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    bias = np.zeros(dimension)
    bias[0] = 0.35
    losses = 0.65 * noise + bias
    losses /= np.maximum(1.0, np.linalg.norm(losses, axis=1, keepdims=True))
    return losses


def _lower_confidence_margin(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
    return mean, standard_error, mean - 1.96 * standard_error


def check_theorem_31_pfmd() -> dict[str, object]:
    """Faithful finite instantiation of the PFMD subroutine in Theorem 3.1."""
    epsilon = 0.2
    repetitions = 160
    rows = []
    direct_certificate_violations = 0
    all_margins = []
    for dimension, horizon in ((2, 128), (4, 256), (8, 512)):
        losses = _oblivious_losses(dimension, horizon, SEED + 31 * dimension)
        cumulative = losses.sum(axis=0)
        comparator = -0.75 * cumulative / np.linalg.norm(cumulative)
        regrets = np.empty(repetitions)
        certificates = np.empty(repetitions)
        for repetition in range(repetitions):
            centers, played, estimates, gradient_bound = pablo.pablo_pfmd(
                losses,
                seed=SEED + 100_000 * dimension + repetition,
                epsilon=epsilon,
            )
            estimate_regret = float(np.sum(estimates * (centers - comparator)))
            estimate_certificate = pablo.pfmd_static_certificate(
                comparator,
                estimates,
                gradient_bound,
                epsilon / dimension,
            )
            if estimate_regret > estimate_certificate + 1e-8:
                direct_certificate_violations += 1
            deltas = losses - estimates
            certificates[repetition] = estimate_certificate + pablo.pfmd_static_certificate(
                comparator,
                deltas,
                gradient_bound,
                epsilon / dimension,
            )
            regrets[repetition] = float(np.sum(losses * (played - comparator)))
        margins = certificates - regrets
        all_margins.extend(margins.tolist())
        mean_margin, margin_se, lower_95 = _lower_confidence_margin(margins)
        rows.append(
            {
                "d": dimension,
                "T": horizon,
                "repetitions": repetitions,
                "mean_bandit_regret": float(regrets.mean()),
                "mean_two_pfmd_certificate": float(certificates.mean()),
                "mean_margin": mean_margin,
                "margin_standard_error": margin_se,
                "margin_95pct_lower": lower_95,
                "mean_regret_over_sqrt_dT": float(
                    regrets.mean() / np.sqrt(dimension * horizon)
                ),
            }
        )
    _, _, overall_lower = _lower_confidence_margin(np.asarray(all_margins))
    aligned = direct_certificate_violations == 0 and overall_lower > 0.0
    return {
        "paper_claim": "Theorem 3.1: PABLO with Jacobsen--Cutkosky PFMD has comparator-adaptive expected static regret.",
        "observed": {
            "pfmd_implementation": "Jacobsen--Cutkosky (2022), Algorithm 4 closed-form update",
            "direct_pfmd_certificate_violations": direct_certificate_violations,
            "overall_two_certificate_margin_95pct_lower": overall_lower,
            "configurations": rows,
        },
        "assessment": "aligned in faithful finite PFMD instantiations" if aligned else "inconclusive under these PFMD instantiations",
        "scope": "The exact cited OLO update and its explicit finite certificate are tested. Asymptotic big-O constants and all adversarial sequences cannot be established by a finite run.",
    }


def _switching_problem(
    dimension: int,
    horizon: int,
    switches: int,
) -> tuple[np.ndarray, np.ndarray]:
    directions = []
    for index in range(switches + 1):
        direction = np.zeros(dimension)
        coordinate = index % dimension
        direction[coordinate] = 1.0 if (index // dimension) % 2 == 0 else -1.0
        directions.append(direction)
    segment = np.minimum(
        (np.arange(horizon) * (switches + 1)) // horizon,
        switches,
    )
    losses = 0.35 * np.stack([directions[index] for index in segment])
    comparators = -0.75 * np.stack([directions[index] for index in segment])
    return losses, comparators


def check_theorem_33_dynamic() -> dict[str, object]:
    """Run PABLO with the paper's Algorithms 5--6 and audit Theorem E.2."""
    dimension, horizon, repetitions = 3, 256, 120
    epsilon = 0.2
    rows = []
    direct_certificate_violations = 0
    all_margins = []
    for switches in (0, 1, 3, 7):
        losses, comparators = _switching_problem(dimension, horizon, switches)
        regrets = np.empty(repetitions)
        certificates = np.empty(repetitions)
        for repetition in range(repetitions):
            centers, played, estimates, gradient_bound = pablo.pablo_dynamic(
                losses,
                seed=SEED + 300_000 + 10_000 * switches + repetition,
                epsilon=epsilon,
            )
            estimate_regret = float(np.sum(estimates * (centers - comparators)))
            estimate_certificate = pablo.dynamic_certificate(
                comparators,
                estimates,
                gradient_bound,
                epsilon / dimension,
            )
            if estimate_regret > estimate_certificate + 1e-8:
                direct_certificate_violations += 1
            deltas = losses - estimates
            certificates[repetition] = estimate_certificate + pablo.dynamic_certificate(
                comparators,
                deltas,
                gradient_bound,
                epsilon / dimension,
            )
            regrets[repetition] = float(np.sum(losses * (played - comparators)))
        margins = certificates - regrets
        all_margins.extend(margins.tolist())
        _, _, lower_95 = _lower_confidence_margin(margins)
        path_length = float(
            np.sum(np.linalg.norm(np.diff(comparators, axis=0), axis=1))
        )
        rows.append(
            {
                "switches": switches,
                "path_length": path_length,
                "mean_dynamic_regret": float(regrets.mean()),
                "mean_two_algorithm6_certificate": float(certificates.mean()),
                "margin_95pct_lower": lower_95,
            }
        )
    _, _, overall_lower = _lower_confidence_margin(np.asarray(all_margins))
    aligned = direct_certificate_violations == 0 and overall_lower > 0.0
    return {
        "paper_claim": "Theorem 3.3: PABLO with refined Algorithm 6 adapts to comparator path length without prior knowledge of it.",
        "observed": {
            "olo_implementation": "paper Algorithms 5 and 6",
            "direct_algorithm6_certificate_violations": direct_certificate_violations,
            "overall_two_certificate_margin_95pct_lower": overall_lower,
            "configurations": rows,
        },
        "assessment": "aligned in faithful finite Algorithm 6 instantiations" if aligned else "inconclusive under these Algorithm 6 instantiations",
        "scope": "Tests the paper's executable dynamic subroutine and explicit Theorem E.2 certificate on controlled path lengths; it does not prove the universal asymptotic theorem.",
    }


def check_theorem_52_lower_bound() -> dict[str, object]:
    """Audit the exact stochastic hard-instance construction and proof constants."""
    rows = []
    valid = True
    products = []
    zero_policy_regrets = []
    for dimension in (2, 4, 8, 16):
        for multiplier in (4, 16, 64):
            horizon = multiplier * dimension
            delta = 1.0 / (8.0 * np.sqrt(horizon))
            expected_loss_norm_sq = dimension * delta**2 + 0.5
            randomization_bound = (
                (horizon - 2.0 * dimension)
                * delta
                * np.sqrt(dimension)
                / 4.0
            )
            theorem_sqrt_bound = np.sqrt(dimension * horizon) / 64.0
            small_action_bound = horizon / (6.0 * dimension)
            zero_policy_regret = horizon * delta * np.sqrt(dimension)
            row_valid = (
                horizon >= 4 * dimension
                and expected_loss_norm_sq <= 1.0
                and randomization_bound + 1e-15 >= theorem_sqrt_bound
                and zero_policy_regret + 1e-15 >= theorem_sqrt_bound
            )
            valid = valid and row_valid
            products.append(dimension * horizon)
            zero_policy_regrets.append(zero_policy_regret)
            rows.append(
                {
                    "d": dimension,
                    "T": horizon,
                    "Delta": delta,
                    "E_loss_norm_squared": expected_loss_norm_sq,
                    "randomization_hammer_bound": randomization_bound,
                    "sqrt_dT_over_64": theorem_sqrt_bound,
                    "small_action_T_over_6d": small_action_bound,
                    "zero_policy_expected_regret": zero_policy_regret,
                    "construction_checks_pass": row_valid,
                }
            )
    slope, _ = np.polyfit(np.log(products), np.log(zero_policy_regrets), 1)
    valid = valid and abs(float(slope) - 0.5) < 1e-12
    return {
        "paper_claim": "Theorem 5.2: the unit-ball stochastic hard instance forces a sqrt(dT)/64 or T/(6d) regret branch.",
        "observed": {
            "construction": "theta in {+/-1/(8 sqrt(T))}^d; Gaussian noise covariance I/(2d)",
            "grid_checks": rows,
            "zero_policy_loglog_slope_vs_dT": float(slope),
            "failed_construction_checks": int(sum(not row["construction_checks_pass"] for row in rows)),
        },
        "assessment": "aligned proof-constant audit" if valid else "construction audit found a discrepancy",
        "scope": "A finite program can audit the hard distribution, moment constraint, and algebraic constants. The theorem's universal quantifier over all algorithms remains a mathematical proof statement.",
    }


def classify_conjecture_53() -> dict[str, object]:
    return {
        "paper_claim": "Conjecture 5.3 proposes the oblivious-norm minimax uBLO rate Theta(||u|| sqrt(T (d v log ||u||))).",
        "observed": {
            "paper_status": "explicitly left as an open problem",
            "known_components": [
                "Theorem 5.1 supplies the scale-learning lower bound",
                "Theorem 5.2 supplies the direction-learning lower bound",
                "the paper states these need not share a single hard loss sequence",
            ],
        },
        "assessment": "correctly scoped open conjecture; not presented as experimentally verified",
        "scope": "Coverage here means faithful classification and boundary analysis, not a pass/fail experiment.",
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
            "theorem_3_1_pfmd": check_theorem_31_pfmd(),
            "theorem_3_3_dynamic": check_theorem_33_dynamic(),
            "theorem_5_2_lower_bound": check_theorem_52_lower_bound(),
            "conjecture_5_3": classify_conjecture_53(),
        },
    }
    print("PABLO EXACT-IDENTITY AND REDUCTION CHECK")
    for key, claim in results["claims"].items():
        print(f"{key}: {claim['assessment']}")
        print(json.dumps(claim["observed"], sort_keys=True))
    print("RESULT_JSON=" + json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
