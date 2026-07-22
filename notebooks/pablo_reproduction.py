# /// script
# dependencies = ["marimo>=0.14"]
# ///
"""Self-contained PABLO tutorial with formal run evidence embedded."""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import marimo as mo

    return math, mo


@app.cell
def _(mo):
    evidence = {
        "score": 12,
        "max_score": 12,
        "max_bias": 4.677549914984633e-14,
        "max_second_error": 5.927973308535068e-15,
        "mean_regret": 16.205403430173575,
        "mean_certificate": 101.54590809712488,
        "pfmd_margin": 669.8739743565591,
        "dynamic_margin": 1466.3344315010604,
        "hp_settings": 27,
        "hp_coverage": 1.0,
        "hp_wilson": 0.963005192523998,
        "hp_tight_max": 2.108464120049903,
        "fixed_point_residual": 8.146325789063808e-31,
        "lower_slope": 0.5000000000000003,
    }
    bars = "".join(
        f'<rect x="20" y="{18 + 29 * index}" width="360" height="19" rx="4" fill="{color}"/>'
        f'<text x="30" y="{32 + 29 * index}" fill="white" font-size="12">{label}: 2/2</text>'
        for index, (label, color) in enumerate(
            [
                ("PABLO reduction", "#2878b5"),
                ("Static PFMD", "#2a9d8f"),
                ("Dynamic regret", "#f28e2b"),
                ("High probability", "#6f65a8"),
                ("Lower bound", "#d1495b"),
                ("Open status", "#17324d"),
            ]
        )
    )
    chart = mo.Html(
        f'<svg viewBox="0 0 400 205" style="max-width:650px;width:100%">{bars}</svg>'
    )
    intro = mo.md(
        f"""
        # PABLO, claim by claim

        **Already-produced formal evidence: {evidence['score']}/{evidence['max_score']} transparent claim coverage.**

        Exact estimator bias was **{evidence['max_bias']:.2e}** and second-moment error was
        **{evidence['max_second_error']:.2e}**. The static and dynamic certificate margins were
        **{evidence['pfmd_margin']:.3f}** and **{evidence['dynamic_margin']:.3f}**. All
        **{evidence['hp_settings']}** final high-probability settings achieved empirical coverage
        **{evidence['hp_coverage']:.1f}**.

        The evidence is embedded below. Opening this notebook does not rerun any formal experiment.
        """
    )
    mo.vstack([intro, chart])
    return (evidence,)


@app.cell
def _(mo):
    mo.md(r"""
    ## The one-scalar-to-vector construction

    An OLO learner proposes a center $w$. PABLO samples one signed eigenvector
    $s \in \{\pm v_i\}_{i=1}^d$, plays

    $$\widetilde w = w + H^{-1/2}s,$$

    observes one scalar $\langle \ell,\widetilde w\rangle$, and returns

    $$\widetilde\ell = dH^{1/2}s\langle \ell,\widetilde w\rangle.$$

    The formal verifier enumerates all $2d$ choices of $s$. That makes estimator
    unbiasedness and its conditional second moment finite sums rather than noisy estimates.
    The same estimator then drives faithful static PFMD, dynamic Algorithm 6, and
    high-probability optimistic learners.
    """)
    return


@app.cell
def _(evidence, mo):
    rows = [
        {"anchor": "PABLO reduction", "paper": "exact identities / two certificates", "observed": f"bias {evidence['max_bias']:.2e}; regret {evidence['mean_regret']:.3f} ≤ {evidence['mean_certificate']:.3f}", "assessment": "aligned"},
        {"anchor": "Static PFMD", "paper": "comparator-adaptive expected regret", "observed": f"95% certificate margin {evidence['pfmd_margin']:.3f}", "assessment": "aligned finite audit"},
        {"anchor": "Dynamic regret", "paper": "path-length adaptation", "observed": f"95% certificate margin {evidence['dynamic_margin']:.3f}", "assessment": "aligned finite audit"},
        {"anchor": "High probability", "paper": "coverage ≥ 1−3δ", "observed": f"27/27 coverage 1.0; Wilson lower {evidence['hp_wilson']:.3f}", "assessment": "aligned declared envelope"},
        {"anchor": "Lower bound", "paper": "√(dT)/64 or T/(6d)", "observed": f"12/12 checks; slope {evidence['lower_slope']:.3f}", "assessment": "aligned construction audit"},
        {"anchor": "Conjecture 5.3", "paper": "explicitly open", "observed": "classified, not experimentally proved", "assessment": "correctly scoped"},
    ]
    mo.md("## Embedded claim matrix")
    mo.ui.table(rows, selection=None)
    return


@app.cell
def _(mo):
    dimension = mo.ui.slider(2, 128, value=8, step=1, label="Dimension d")
    dimension
    return (dimension,)


@app.cell
def _(dimension, math, mo):
    d_value = dimension.value
    rms_value = math.sqrt(d_value)
    mo.md(
        rf"""
        ## Why two dimension scales appear

        In the sparse diagnostic family at **d = {d_value}**, conditional RMS is
        $\sqrt{{d}}$ = **{rms_value:.3f}**, while the support maximum is $d$ =
        **{d_value}**. The formal branch fixed seven dimensions and recovered log--log
        slopes **0.500** and **1.000**. This bounded interaction evaluates the closed form
        only; it does not create new reproduction evidence.
        """
    )
    return


@app.cell
def _(mo):
    horizon = mo.ui.slider(96, 384, value=192, step=96, label="Horizon T")
    delta = mo.ui.dropdown([0.10, 0.05, 0.02], value=0.05, label="Confidence δ")
    mo.hstack([horizon, delta])
    return delta, horizon


@app.cell
def _(delta, evidence, horizon, math, mo):
    hidden_log = math.log(horizon.value / delta.value)
    target = 1.0 - 3.0 * delta.value
    mo.md(
        f"""
        ## Read the high-probability result honestly

        At **T = {horizon.value}** and **δ = {delta.value:.2f}**, the theorem target is
        **1−3δ = {target:.2f}** and the declared hidden-log factor is
        **log(T/δ) = {hidden_log:.3f}**. The final run's worst empirical quantile needed
        only **{evidence['hp_tight_max']:.3f}×** the unexpanded displayed rate.

        The earlier coefficient-one proxy failed on longer structured losses. It remains a
        negative control; the notebook does not rewrite that divergence into a success.
        The implicit update itself was solved to residual **{evidence['fixed_point_residual']:.2e}**.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Scope and provenance

    All formal runs used a local CPU and the fixed command
    `uv run --no-cache --with numpy==2.1.3 python repro/src/verify_pablo.py`.
    Finite dimensions, horizons, seeded losses, and Monte Carlo repetitions substitute for
    universal quantifiers. The lower-bound program checks the hard construction, not every
    possible algorithm; Conjecture 5.3 remains open.

    Read the complete [illustrated report on GitHub](https://github.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/blob/main/reports/pablo_reproduction/report.md)
    for figures, exact branch lineage, compute cost, and the retained negative results.
    """)
    return


if __name__ == "__main__":
    app.run()
