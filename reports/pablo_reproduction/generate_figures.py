"""Regenerate the report's four figures from immutable run-log metrics."""
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


def headline() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    discrepancies = np.array([4.677549914984633e-14, 5.927973308535068e-15, 4.440892098500626e-16])
    labels = ["Estimator\nbias", "Second-moment\nidentity", "Support-bound\nroundoff"]
    axes[0].bar(labels, discrepancies, color=[BLUE, GREEN, ORANGE])
    axes[0].axhline(1e-10, color=RED, linestyle="--", label="acceptance tolerance")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("absolute / relative discrepancy")
    axes[0].set_title("Exact finite-support checks")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    values = [16.205403430173575, 101.54590809712488]
    axes[1].bar(["Mean bandit\nregret", "Mean OLO\ncertificate"], values, color=[BLUE, GREEN])
    axes[1].set_ylabel("cumulative loss units")
    axes[1].set_title("Proposition 2.3 instantiation")
    axes[1].text(0.5, 0.62, "95% lower margin = 85.27", transform=axes[1].transAxes,
                 ha="center", color=NAVY, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("PABLO's tested reduction matches its finite-computation predictions", color=NAVY, fontweight="bold")
    fig.savefig(OUT / "headline_result.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def dimension_gap() -> None:
    dimensions = np.array([2, 4, 8, 16, 32, 64, 128])
    rms = np.sqrt(dimensions)
    support = dimensions.astype(float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.loglog(dimensions, rms, "o-", color=BLUE, linewidth=2, label="conditional RMS (slope 0.500)")
    ax.loglog(dimensions, support, "s-", color=ORANGE, linewidth=2, label="support maximum (slope 1.000)")
    ax.set_xlabel("dimension d")
    ax.set_ylabel("estimator norm scale")
    ax.set_title("Expectation and trajectory-wise control separate by √d")
    ax.set_xticks(dimensions, labels=[str(value) for value in dimensions])
    ax.grid(which="both", alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(OUT / "dimension_gap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def h_ablation() -> None:
    multipliers = np.array([0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64])
    second = np.array([0.5625, 0.625, 0.75, 1.0, 1.5, 2.5, 4.5, 8.5, 16.5, 32.5])
    support = np.array([0.0572533, 0.0703125, 0.0910692, 0.125, 0.182138, 0.28125, 0.458027, 0.78125, 1.38480, 2.53125])
    fig, ax = plt.subplots(figsize=(7.4, 4.7), constrained_layout=True)
    ax.plot(multipliers, second, "o-", color=BLUE, label="second-moment / bound")
    ax.plot(multipliers, support, "s-", color=ORANGE, label="support / bound")
    ax.axhline(1, color=RED, linestyle="--", label="claimed limit")
    ax.axvline(1, color=NAVY, linestyle=":", label="largest admissible H")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("H spectral cap multiplier")
    ax.set_ylabel("normalized bound ratio")
    ax.set_title("Negative control: the H condition does real work")
    ax.grid(which="both", alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(OUT / "h_ablation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def support_distribution() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.bar(["norm = 0\n(14 outcomes)", "norm = 8\n(2 outcomes)"], [14 / 16, 2 / 16],
           color=["#b8c7d9", ORANGE])
    ax.set_ylim(0, 1)
    ax.set_ylabel("probability on the 2d support")
    ax.set_title("Why RMS is √d while the support maximum is d (d = 8)")
    ax.text(0.5, 0.62, "RMS = √8 = 2.83\nmaximum = 8", transform=ax.transAxes,
            ha="center", va="center", color=NAVY, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(OUT / "support_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    headline()
    dimension_gap()
    h_ablation()
    support_distribution()
