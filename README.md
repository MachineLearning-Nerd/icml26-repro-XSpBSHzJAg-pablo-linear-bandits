# PABLO linear-bandit reproduction: scaled evidence and two proof counterexamples

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/blob/main/notebooks/pablo_reproduction.py)

This repository audits six claims from [*A Perturbation Approach to Unconstrained Linear Bandits* (arXiv:2603.28201)](https://arxiv.org/abs/2603.28201). The first official judgment was **5/12**: five toy verdicts and one inconclusive verdict. The new revision addresses that feedback with exact estimator enumeration through `d=128`, end-to-end reduction runs through `d=32,T=4096`, the cited PFMD and dynamic learners, high-probability tests through `d=16,T=4096`, and independent counterexamples to Appendix F and the literal unqualified Conjecture 5.3 rate.

**Current assessment:** the final internal evidence matrix covers 12/12 possible points, but this is not the official score. The official result remains **5/12 until the new Hugging Face revision is judged**. The strongest divergent result is Appendix F: a legal zero-action policy violates the displayed `R_T >= E[S_tau]/6` step in 5/5 regimes because the proof omits `Delta*sqrt(d)`. The corrected step holds in 5/5 regimes and yields only `sqrt(T/d)/96` in that branch. This falsifies the proof as written, not the folklore theorem statement.

All formal experiments used the agreed **local CPU** and the fixed command `uv run --no-cache --with numpy==2.1.3 python repro/src/verify_pablo.py`. No GPU or remote compute was used. Finite dimensions, horizons, seeded loss families, and explicit hidden-log envelopes substitute for universal quantifiers; the report states each limitation.

- [Illustrated claim-by-claim report](reports/pablo_reproduction/report.md)
- [Self-contained marimo tutorial with embedded evidence](notebooks/pablo_reproduction.py)
- [Hugging Face Trackio logbook](https://huggingface.co/spaces/DineshAI/XSpBSHzJAg)
- [Independent prior logbook that identified the Appendix F issue](https://huggingface.co/spaces/gkalyanaraman3/XSpBSHzJAg)
- Local notebook: `uvx --no-cache marimo edit notebooks/pablo_reproduction.py` or `uvx --no-cache marimo run notebooks/pablo_reproduction.py`

## Experiment log

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| [`orx/baseline-pre-existing-verifier-no-cache`](https://github.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/tree/orx/baseline-pre-existing-verifier-no-cache) | Immutable original proxy control | `uv run --no-cache --with numpy==2.1.3 python repro/src/verify_pablo.py` | Printed 6/6 but used hard-coded or non-diagnostic checks; excluded from final evidence | Local CPU, 15s |
| [`orx/theorem-consistent-high-probability-envelope-aud`](https://github.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/tree/orx/theorem-consistent-high-probability-envelope-aud) | Earlier finite theorem audit published to the judge | `uv run --no-cache --with numpy==2.1.3 python repro/src/verify_pablo.py` | External judge later classified five claims toy and one inconclusive; official 5/12 | Local CPU, 50s |
| [`orx/non-toy-theorem-and-conjecture-boundary-audit`](https://github.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/tree/orx/non-toy-theorem-and-conjecture-boundary-audit) | Scale the theorem grids and add the exact `d>>T` conjecture cap | `uv run --no-cache --with numpy==2.1.3 python repro/src/verify_pablo.py` | All theorem checks aligned; exact conjecture cap ratio reached `1/16` | Local CPU, 4m20s |
| [`orx/full-scale-reduction-and-theorem-stress-suite`](https://github.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/tree/orx/full-scale-reduction-and-theorem-stress-suite) | Enlarge reduction, dynamic, and high-probability runs | `uv run --no-cache --with numpy==2.1.3 python repro/src/verify_pablo.py` | Reduction through `d=32,T=4096`; 0 PFMD/dynamic violations; 27/27 high-probability envelopes covered | Local CPU, 9m56s |
| [`orx/independent-reduction-and-appendix-f-counterexam`](https://github.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/tree/orx/independent-reduction-and-appendix-f-counterexam) | Add reduction martingales and independently reproduce the proof defect | `uv run --no-cache --with numpy==2.1.3 python repro/src/verify_pablo.py` | All martingales within `1.24 sigma`; Appendix F displayed step fails 5/5 and corrected step holds 5/5 | Local CPU, 11m06s |
| `main` | README, report, figures, notebook, and integrated audited implementation | Not run as an experiment (publication surface) | Reader-facing release; awaiting official re-judgment | None |

## Repository map

- `repro/src/pablo.py` — estimator, PFMD, dynamic, and high-probability implementations.
- `repro/src/verify_pablo.py` — seeded claim-by-claim audit and JSON evidence output.
- `reports/pablo_reproduction/` — self-contained illustrated technical article and figure generator.
- `notebooks/pablo_reproduction.py` — embedded-evidence tutorial for local marimo or Molab.

Experiment branches are immutable records of the code that produced each logged result. `main` is the publication surface and includes the final audited implementation for readers.
