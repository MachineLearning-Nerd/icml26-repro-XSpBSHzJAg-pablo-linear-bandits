# Verification runs


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_bfcc7be5236c", "created_at": "2026-07-21T15:58:46+00:00", "title": "Formal experiment provenance"}
-->
## Formal experiment provenance

| Branch | Purpose | Outcome | Local CPU time |
|---|---|---|---:|
| [`orx/baseline-pre-existing-verifier-no-cache`](https://github.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/tree/orx/baseline-pre-existing-verifier-no-cache) | Preserve the original verifier | Non-diagnostic 6/6 control | 15 s |
| [`orx/exact-pablo-estimator-and-reduction-checks`](https://github.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/tree/orx/exact-pablo-estimator-and-reduction-checks) | Exact Proposition 2.1 / Corollary 2.2 enumeration and Proposition 2.3 OGD instantiation | Aligned | 20 s |
| [`orx/dimension-gap-mechanism`](https://github.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/tree/orx/dimension-gap-mechanism) | Test √d expected-moment versus d support | Slopes 0.500 and 1.000 | 25 s |
| [`orx/perturbation-scale-negative-control`](https://github.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/tree/orx/perturbation-scale-negative-control) | Sweep `H` through and beyond its cap | Held in-domain; broke out-of-domain | 25 s |

Every branch inherited this exact command from `orx exp status`:

```text
uv run --no-cache --with numpy==2.1.3 python repro/src/verify_pablo.py
```

The three evidence runs used deterministic seeds. Proposition 2.3 used 1,200 perturbation repetitions with `d=5` and `T=250`. The lower endpoint of the 95% confidence interval for `certificate − regret` was `85.266`.

Successful scientific compute totaled 85 CPU seconds. Two zero-second dependency/setup attempts occurred before Python started and are excluded from evidence.
