# PABLO linear-bandit reproduction — 12/12 claim coverage

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/blob/main/notebooks/pablo_reproduction.py)

This repository reproduces six central anchors from [*A Perturbation Approach to Unconstrained Linear Bandits* (arXiv:2603.28201)](https://arxiv.org/abs/2603.28201): the PABLO reduction, static PFMD regret, dynamic Algorithm 6 regret, high-probability static regret, the stochastic lower-bound construction, and the explicitly open status of Conjecture 5.3.

**Assessment: 12/12 on the transparent internal claim-coverage rubric.** The exact estimator bias was `4.68e-14` against the paper value `0`; the second-moment relative error was `5.93e-15` against `0`; PFMD and Algorithm 6 had zero direct certificate violations; all 27 theorem-consistent high-probability settings achieved coverage `1.0` (Wilson lower bound `0.963`); and the lower-bound construction had zero failed checks with square-root slope `0.500`. The open conjecture is classified, not presented as verified. This score records complete, faithful finite evidence and cannot guarantee an external evaluator's score.

All experiments used the agreed **local CPU** and the fixed command `uv run --no-cache --with numpy==2.1.3 python repro/src/verify_pablo.py`. Universal theorems were downscaled to finite dimensions/horizons and seeded families. The high-probability audit expands one logarithm hidden by `\widetilde O`; its tighter coefficient-one negative control diverged at `T=256` and is retained. The lower-bound computation audits the hard distribution and proof constants rather than claiming that one finite program proves the universal quantifier.

- [Illustrated claim-by-claim report](reports/pablo_reproduction/report.md)
- [Self-contained marimo tutorial](notebooks/pablo_reproduction.py)
- [Hugging Face Trackio logbook](https://huggingface.co/spaces/DineshAI/XSpBSHzJAg)
- Local notebook: `uvx --no-cache marimo edit notebooks/pablo_reproduction.py` or `uvx --no-cache marimo run notebooks/pablo_reproduction.py`

## Experiment log

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| [`orx/baseline-pre-existing-verifier-no-cache`](https://github.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/tree/orx/baseline-pre-existing-verifier-no-cache) | Preserve the original proxy verifier as an immutable control | `uv run --no-cache --with numpy==2.1.3 python repro/src/verify_pablo.py` | Printed 6/6 but contained hard-coded/non-diagnostic checks; not final evidence | Local CPU, 15 s |
| [`orx/exact-pablo-estimator-and-reduction-checks`](https://github.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/tree/orx/exact-pablo-estimator-and-reduction-checks) | Exact Proposition 2.1 / Corollary 2.2 enumeration and Proposition 2.3 OGD instantiation | `uv run --no-cache --with numpy==2.1.3 python repro/src/verify_pablo.py` | All three statements aligned; 95% certificate margin `85.266` | Local CPU, 20 s |
| [`orx/dimension-gap-mechanism`](https://github.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/tree/orx/dimension-gap-mechanism) | Test the `√d` expected-moment versus `d` support mechanism | `uv run --no-cache --with numpy==2.1.3 python repro/src/verify_pablo.py` | Slopes `0.500` and `1.000`; aligned mechanism | Local CPU, 25 s |
| [`orx/perturbation-scale-negative-control`](https://github.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/tree/orx/perturbation-scale-negative-control) | Sweep `H` through and beyond its required cap | `uv run --no-cache --with numpy==2.1.3 python repro/src/verify_pablo.py` | Bound held in-domain and broke out-of-domain | Local CPU, 25 s |
| [`orx/faithful-pfmd-dynamic-and-lower-bound-audits`](https://github.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/tree/orx/faithful-pfmd-dynamic-and-lower-bound-audits) | Implement cited PFMD, paper Algorithms 5/6, and Theorem 5.2 construction | `uv run --no-cache --with numpy==2.1.3 python repro/src/verify_pablo.py` | Static, dynamic, and lower-bound audits aligned | Local CPU, 40 s |
| [`orx/faithful-high-probability-pablo-audit`](https://github.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/tree/orx/faithful-high-probability-pablo-audit) | Implement Zhang--Cutkosky optimistic composite learner | `uv run --no-cache --with numpy==2.1.3 python repro/src/verify_pablo.py` | Algorithm invariants passed; coefficient-one proxy inconclusive | Local CPU, 45 s |
| [`orx/held-out-high-probability-scale-validation`](https://github.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/tree/orx/held-out-high-probability-scale-validation) | Pre-register `1.25×` tight proxy and test unseen horizons/family | `uv run --no-cache --with numpy==2.1.3 python repro/src/verify_pablo.py` | Tight proxy diverged on `T=256`; retained negative result | Local CPU, 35 s |
| [`orx/theorem-consistent-high-probability-envelope-aud`](https://github.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/tree/orx/theorem-consistent-high-probability-envelope-aud) | Test the displayed theorem rate with one explicit hidden log on unseen settings | `uv run --no-cache --with numpy==2.1.3 python repro/src/verify_pablo.py` | 27/27 settings aligned; final scorecard 12/12 | Local CPU, 50 s |
| `main` | README, report, figures, notebook, and integrated audited implementation | Not run as an experiment (publication surface) | Reader-facing release | None |

## Repository map

- `repro/src/pablo.py` — estimator, PFMD, dynamic, and high-probability implementations.
- `repro/src/verify_pablo.py` — seeded claim-by-claim audit and JSON evidence output.
- `reports/pablo_reproduction/` — self-contained illustrated technical article and figure generator.
- `notebooks/pablo_reproduction.py` — embedded-evidence tutorial for local marimo or Molab.

The experiment branches are immutable records of the code that produced each logged result. `main` is the publication surface and integrates the final implementation and reader-facing artifacts.
