# /// script
# dependencies = ["marimo>=0.14"]
# ///
"""Self-contained tutorial for the PABLO reproduction; formal evidence is embedded."""

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
        "max_bias": 4.677549914984633e-14,
        "max_second_error": 5.927973308535068e-15,
        "corollary_second_ratio": 0.8723636598630653,
        "corollary_support_ratio": 0.8676697555233535,
        "mean_regret": 16.205403430173575,
        "mean_certificate": 101.54590809712488,
        "margin_lower_95": 85.26638210495742,
        "rms_slope": 0.4999999999999998,
        "support_slope": 0.9999999999999998,
    }
    mo.md(
    f"""
    # PABLO: seeing a vector through one scalar

    **Already-produced evidence:** exhaustive enumeration found maximum estimator bias
    **{evidence['max_bias']:.2e}** and second-moment identity error
    **{evidence['max_second_error']:.2e}**. In the generic reduction, mean bandit
    regret was **{evidence['mean_regret']:.3f}** versus a two-certificate bound of
    **{evidence['mean_certificate']:.3f}**; the lower 95% margin was
    **{evidence['margin_lower_95']:.3f}**.

    This notebook embeds the formal run results. Opening it does not rerun the 1,200
    repetitions or claim that an interactive calculation is new reproduction evidence.
    """
    )
    return (evidence,)


@app.cell
def _(mo):
    mo.md(r"""
    ## The construction

    An online linear optimization learner proposes a center $w$. PABLO selects one
    signed eigenvector $s \in \{\pm v_i\}_{i=1}^d$, plays

    $$\widetilde w = w + H^{-1/2}s,$$

    observes the single scalar $\langle \ell, \widetilde w\rangle$, and returns

    $$\widetilde\ell = dH^{1/2}s\langle \ell,\widetilde w\rangle.$$

    Averaging over the complete finite support gives
    $\mathbb{E}[\widetilde\ell\mid\mathcal F_{t-1}]=\ell$. The formal run enumerated
    that support rather than estimating the expectation by sampling.
    """)
    return


@app.cell
def _(evidence, mo):
    rows = [
        {"check": "Estimator bias", "paper": "0", "observed": f"{evidence['max_bias']:.3e}", "assessment": "aligned"},
        {"check": "Second-moment error", "paper": "0", "observed": f"{evidence['max_second_error']:.3e}", "assessment": "aligned"},
        {"check": "Corollary second ratio", "paper": "≤ 1", "observed": f"{evidence['corollary_second_ratio']:.3f}", "assessment": "aligned"},
        {"check": "Corollary support ratio", "paper": "≤ 1", "observed": f"{evidence['corollary_support_ratio']:.3f}", "assessment": "aligned"},
        {"check": "RMS dimension slope", "paper": "0.5", "observed": f"{evidence['rms_slope']:.3f}", "assessment": "aligned mechanism"},
        {"check": "Support dimension slope", "paper": "1.0", "observed": f"{evidence['support_slope']:.3f}", "assessment": "aligned mechanism"},
    ]
    mo.md("## Evidence table")
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
    support_value = float(d_value)
    mo.md(
    rf"""
    ## Bounded interaction: the dimension mechanism

    For the sparse-loss family used in the formal branch, at **d = {d_value}**:

    - conditional RMS = $\sqrt{{d}}$ = **{rms_value:.3f}**;
    - support maximum = $d$ = **{support_value:.0f}**;
    - their ratio is $\sqrt{{d}}$ = **{support_value / rms_value:.3f}**.

    The slider evaluates the closed-form family only. The formal evidence used the
    fixed dimensions 2, 4, 8, 16, 32, 64, and 128 and found slopes 0.500 and 1.000.
    """
    )
    return


@app.cell
def _(mo):
    cap_multiplier = mo.ui.slider(0.125, 8.0, value=1.0, step=0.125, label="H cap multiplier")
    cap_multiplier
    return (cap_multiplier,)


@app.cell
def _(cap_multiplier, mo):
    multiplier_value = cap_multiplier.value
    second_ratio = (1.0 + multiplier_value) / 2.0
    status = "inside the stated cap" if multiplier_value <= 1.0 else "outside the theorem's stated domain"
    mo.md(
    f"""
    ## Negative control

    This setting is **{status}**. For the fixed diagnostic vectors, the normalized
    conditional second moment is **{second_ratio:.3f}**. The claimed limit is 1.

    At the boundary the ratio is exactly 1; above it, this family immediately violates
    the conditional second-moment guarantee. This is why Equation (4)'s matrix condition
    is a substantive part of the result.
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## What remains

    The finite estimator reduction is aligned. The full PFMD static theorem, dynamic
    meta-algorithm, high-probability optimistic learners, and universal lower-bound proof
    were not replaced by proxy pass/fail tests. See the
    [illustrated report](../reports/pablo_reproduction/report.md) for the complete claim
    matrix, substitutions, compute cost, and experiment-branch provenance.
    """)
    return


if __name__ == "__main__":
    app.run()
