# Evidence


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_b85a6e6dbd3a", "created_at": "2026-07-21T15:58:42+00:00", "title": "Observed evidence"}
-->
## Headline result

![Exact estimator identities and the generic reduction agree with the paper's finite-computation predictions.](https://raw.githubusercontent.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/main/reports/pablo_reproduction/images/headline_result.png)

Across 80 random positive-definite geometries, exhaustive enumeration of every `2d` perturbation outcome found:

- maximum estimator bias: `4.6775e-14`;
- maximum second-moment relative error: `5.9280e-15`;
- maximum support-bound ratio: `1.0000000000000004` (floating-point roundoff).

The Corollary 2.2 stress test covered 96 additional instances. Maximum utilization was `0.8724` of the conditional-second-moment bound and `0.8677` of the support bound.

## Dimension mechanism

![Conditional RMS and support maximum have slopes one-half and one.](https://raw.githubusercontent.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/main/reports/pablo_reproduction/images/dimension_gap.png)

For the controlled sparse-loss family over `d = 2,…,128`, conditional RMS grew with fitted exponent `0.500`, while the support maximum grew with exponent `1.000`. This is the estimator mechanism behind the paper's √d comparator-regime separation.

## Negative control

![The H condition is consequential.](https://raw.githubusercontent.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/main/reports/pablo_reproduction/images/h_ablation.png)

Both bounds held through the stated spectral cap. The second-moment ratio was exactly `1.0` at the boundary and increased to `1.5` at `2×`; the support ratio also exceeded one under larger violations. The aligned in-domain result is therefore not produced by a test that always passes.
