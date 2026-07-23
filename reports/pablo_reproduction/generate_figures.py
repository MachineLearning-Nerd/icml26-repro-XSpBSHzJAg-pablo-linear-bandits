"""Regenerate the report's five figures from the final immutable run log."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).with_name("images")
OUT.mkdir(parents=True, exist_ok=True)
NAVY = "#17324d"
BLUE = "#2878b5"
ORANGE = "#f28e2b"
GREEN = "#2a9d8f"
RED = "#d1495b"
PURPLE = "#6f65a8"


def appendix_f_counterexample() -> None:
    labels = ["d=4\nT=64", "d=4\nT=1k", "d=8\nT=2k", "d=16\nT=8k", "d=64\nT=1M"]
    actual = np.array([2.0, 7.905694, 16.0, 45.254834, 1000.0])
    claimed = np.array([10.666667, 166.666667, 341.333333, 1365.333333, 166666.666667])
    corrected = np.array([0.333333, 1.317616, 2.666667, 7.542472, 166.666667])
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10.2, 5.1), constrained_layout=True)
    ax.bar(x - width, claimed, width, color=RED, label="paper step: S_tau / 6")
    ax.bar(x, actual, width, color=BLUE, label="zero-policy actual regret")
    ax.bar(x + width, corrected, width, color=GREEN, label="corrected: Delta sqrt(d) S_tau / 6")
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("regret lower-bound quantity (log scale)")
    ax.set_title("Appendix F: the displayed proof step exceeds a legal policy's actual regret")
    ax.text(0.02, 0.96, "5/5 counterexamples; corrected step holds 5/5", transform=ax.transAxes,
            va="top", color=NAVY, fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.11))
    ax.grid(axis="y", which="both", alpha=0.2)
    fig.savefig(OUT / "appendix_f_counterexample.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def reduction_martingales() -> None:
    labels = ["d=5\nT=250", "d=8\nT=512", "d=16\nT=2048", "d=32\nT=4096"]
    exploration = [0.696, 0.877, 0.368, 0.398]
    estimator = [0.541, 1.108, 0.042, 1.215]
    ghost = [0.515, 1.236, 0.859, 1.169]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    for offset, values, label, color, marker in (
        (-0.16, exploration, "exploration", BLUE, "o"),
        (0.0, estimator, "estimator-center", ORANGE, "s"),
        (0.16, ghost, "ghost iterate", GREEN, "^"),
    ):
        ax.scatter(x + offset, values, s=70, color=color, marker=marker, label=label)
    ax.axhline(1.96, color=RED, linestyle="--", label="two-sided 95% threshold")
    ax.axhline(3.0, color=NAVY, linestyle=":", label="predeclared 3-sigma guardrail")
    ax.set_xticks(x, labels)
    ax.set_ylabel("absolute z-score for zero mean")
    ax.set_title("PABLO reduction: all 12 martingale diagnostics are within 1.24 sigma")
    ax.set_ylim(0, 3.25)
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(OUT / "reduction_martingales.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def static_dynamic_scale() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    static_labels = ["2/128", "4/256", "8/512", "16/1024", "32/2048"]
    static_regret = np.array([44.743, 65.196, 136.466, 268.123, 534.248])
    static_certificate = np.array([320.690, 652.773, 1391.412, 2904.375, 5965.840])
    x = np.arange(5)
    axes[0].plot(x, static_regret, "o-", color=BLUE, label="mean regret")
    axes[0].plot(x, static_certificate, "s-", color=GREEN, label="two-PFMD certificate")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, static_labels, rotation=20)
    axes[0].set_xlabel("d / T")
    axes[0].set_title("Static PFMD: 0 violations")
    axes[0].legend(frameon=False)
    axes[0].grid(which="both", alpha=0.2)

    path = np.array([8.4853, 33.9411, 135.7645])
    dynamic_regret = np.array([2117.878, 2149.906, 2150.483])
    dynamic_certificate = np.array([37474.586, 124602.233, 454153.643])
    axes[1].plot(path, dynamic_regret, "o-", color=BLUE, label="mean regret")
    axes[1].plot(path, dynamic_certificate, "s-", color=ORANGE, label="Algorithm 6 certificate")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("comparator path length")
    axes[1].set_title("Dynamic Algorithm 6: 0 violations")
    axes[1].legend(frameon=False)
    axes[1].grid(which="both", alpha=0.2)
    fig.suptitle("Faithful cited learners at d up to 32 and T up to 8,192", color=NAVY,
                 fontweight="bold")
    fig.savefig(OUT / "static_dynamic_scale.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def high_probability() -> None:
    horizons = np.array([1024, 2048, 4096])
    required = {
        "biased spherical": [0.6570, 0.9762, 1.4215],
        "block rotations": [1.2571, 1.8773, 2.7598],
        "dense chirp": [1.1777, 1.7586, 2.5855],
    }
    fig, ax = plt.subplots(figsize=(8.8, 4.9), constrained_layout=True)
    for (label, values), color, marker in zip(
        required.items(), [BLUE, ORANGE, GREEN], ["o", "s", "^"]
    ):
        ax.plot(horizons, values, marker=marker, linewidth=2, color=color, label=label)
    ax.axhline(1.0, color=RED, linestyle="--", label="coefficient-one proxy")
    ax.plot(horizons, np.log(horizons / 0.1), color=NAVY, linestyle=":", linewidth=2,
            label="declared hidden-log envelope (delta=0.1)")
    ax.set_xticks(horizons)
    ax.set_xlabel("horizon T (d=16)")
    ax.set_ylabel("target quantile / displayed rate")
    ax.set_title("Theorem 4.2: 27/27 declared envelopes cover; tight proxy diverges")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    fig.savefig(OUT / "high_probability_scaled.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def conjecture_boundary() -> None:
    d_over_t = np.array([4, 16, 64, 256], dtype=float)
    cap_ratio = 1.0 / np.sqrt(d_over_t)
    proof_ratio = 2.0 / (3.0 * d_over_t)
    fig, ax = plt.subplots(figsize=(8.7, 4.9), constrained_layout=True)
    ax.loglog(d_over_t, cap_ratio, "o-", color=PURPLE, linewidth=2,
              label="zero-action cap / Conjecture 5.3 scale")
    ax.loglog(d_over_t, proof_ratio, "s-", color=RED, linewidth=2,
              label="corrected Appendix F branch / sqrt(dT)")
    ax.set_xticks(d_over_t, labels=["4", "16", "64", "256"])
    ax.set_xlabel("dimension-to-horizon ratio d/T")
    ax.set_ylabel("ratio to displayed target")
    ax.set_title("Two exact boundary certificates vanish outside the intended regime")
    ax.text(0.04, 0.10, "slopes: -1/2 (conjecture cap), -1 (proof branch)",
            transform=ax.transAxes, color=NAVY, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(which="both", alpha=0.2)
    fig.savefig(OUT / "conjecture_boundary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    appendix_f_counterexample()
    reduction_martingales()
    static_dynamic_scale()
    high_probability()
    conjecture_boundary()
