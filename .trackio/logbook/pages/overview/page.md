# Overview


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_5302ca3b4709", "created_at": "2026-07-21T15:58:39+00:00", "title": "PABLO reproduction overview"}
-->
# PABLO under the microscope

This logbook reproduces the central constructive claim in **A Perturbation Approach to Unconstrained Linear Bandits**:

- [arXiv 2603.28201](https://arxiv.org/abs/2603.28201)
- [OpenReview XSpBSHzJAg](https://openreview.net/forum?id=XSpBSHzJAg)
- [Full illustrated report](https://github.com/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/blob/main/reports/pablo_reproduction/report.md)
- [Interactive marimo tutorial](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-XSpBSHzJAg-pablo-linear-bandits/blob/main/notebooks/pablo_reproduction.py)

**Central question.** Can a learner that observes only one scalar loss per round construct a useful full loss-vector estimate for an online linear optimization learner?

**Assessment.** Aligned for the tested finite-support reduction and its dimension mechanism. The specialized PFMD, dynamic-regret, high-probability meta-algorithms, and universal lower-bound proof were not replaced by proxy pass/fail tests.

All formal runs used the local CPU and the same command:

```text
uv run --no-cache --with numpy==2.1.3 python repro/src/verify_pablo.py
```
