# PABLO linear-bandit reproduction

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/blob/main/notebooks/pablo_reproduction.py)

This project tests the central constructive claim in [*A Perturbation Approach to Unconstrained Linear Bandits* (arXiv:2603.28201)](https://arxiv.org/abs/2603.28201): PABLO's finite perturbation turns bandit feedback into an unbiased, controlled loss estimate that can be passed to an online linear optimization learner.

**Assessment: aligned for the tested reduction and its dimension mechanism.** Proposition 2.1 predicts zero estimator bias and zero second-moment identity error; exhaustive enumeration produced maximum errors of `4.68e-14` and `5.93e-15`. Corollary 2.2 predicts normalized ratios at most `1`; the largest observed ratios were `0.872` (second moment) and `0.868` (support). In a finite OGD instantiation of Proposition 2.3, mean bandit regret was `16.205` versus a mean two-certificate bound of `101.546`; the lower 95% confidence limit for bound minus regret was `85.266`.

The runs used the local CPU only, NumPy 2.1.3, and a fixed `uv --no-cache` command. They exactly enumerate finite estimator support and use 1,200 seeded repetitions for the reduction. This is a downscaled numerical audit of the paper's proof mechanisms, not a full implementation of the specialized PFMD, dynamic-regret, or high-probability meta-algorithms. The paper contains no empirical headline table or official reference implementation.

- [Illustrated technical report](reports/pablo_reproduction/report.md)
- [Self-contained marimo tutorial](notebooks/pablo_reproduction.py)
- [Hugging Face Trackio logbook](https://huggingface.co/spaces/DineshAI/XSpBSHzJAg)
- Local notebook commands: `uvx marimo edit notebooks/pablo_reproduction.py` or `uvx marimo run notebooks/pablo_reproduction.py`

## Experiment log

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| [`orx/baseline-pre-existing-verifier-no-cache`](https://github.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/tree/orx/baseline-pre-existing-verifier-no-cache) | Preserve the pre-existing proxy verifier as an immutable control | `uv run --no-cache --with numpy==2.1.3 python repro/src/verify_pablo.py` | Printed 6/6, but relied on hard-coded or non-diagnostic checks | Local CPU, 15 s |
| [`orx/exact-pablo-estimator-and-reduction-checks`](https://github.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/tree/orx/exact-pablo-estimator-and-reduction-checks) | Exact Proposition 2.1 / Corollary 2.2 enumeration and Proposition 2.3 OGD instantiation | `uv run --no-cache --with numpy==2.1.3 python repro/src/verify_pablo.py` | Aligned on all three tested statements | Local CPU, 20 s |
| [`orx/dimension-gap-mechanism`](https://github.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/tree/orx/dimension-gap-mechanism) | Test the √d expected-moment versus d support mechanism | `uv run --no-cache --with numpy==2.1.3 python repro/src/verify_pablo.py` | Slopes 0.500 and 1.000; aligned mechanism | Local CPU, 25 s |
| [`orx/perturbation-scale-negative-control`](https://github.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/tree/orx/perturbation-scale-negative-control) | Sweep `H` through and beyond its required cap | `uv run --no-cache --with numpy==2.1.3 python repro/src/verify_pablo.py` | Bound held in-domain and broke out-of-domain as a diagnostic control | Local CPU, 25 s |
| `main` | Reader-facing README, report, figures, and notebook | Not run as an experiment (publication surface) | Presentation only | None |

## Upstream snapshot

The repository began as a compact clean-room verifier for OpenReview `XSpBSHzJAg` / arXiv `2603.28201`. The formal experiment branches above preserve and replace that snapshot without rewriting completed-run history.
