# Claims


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_4dd71e46b6f7", "created_at": "2026-07-21T15:58:41+00:00", "title": "Claim-by-claim assessment"}
-->
## Claim-by-claim assessment

| Paper statement | Paper prediction | Observed evidence | Assessment |
|---|---:|---:|---|
| Proposition 2.1, unbiasedness | bias `= 0` | maximum L2 bias `4.68e-14` | **Aligned** |
| Proposition 2.1, second moment | relative identity error `= 0` | maximum `5.93e-15` | **Aligned** |
| Proposition 2.1, support bound | normalized ratio `≤ 1` | maximum `1 + 4.4e-16` (roundoff) | **Aligned** |
| Corollary 2.2 | moment and support ratios `≤ 1` | `0.872` and `0.868` | **Aligned** |
| Proposition 2.3 with OGD | expected regret `≤` two OLO certificates | `16.205` vs `101.546`; lower 95% margin `85.266` | **Aligned in this instantiation** |
| Theorem 3.1 dimension mechanism | RMS `O(√d)`, support `O(d)` | fitted slopes `0.500`, `1.000` | **Aligned mechanism** |
| Theorem 3.1 full PFMD rate | comparator-regime bounds up to polylogs | PFMD was not reimplemented | **Not attempted at full scale** |
| Theorem 3.3 | dynamic regret adapts as `√P_T` | specialized dynamic meta-algorithm not reimplemented | **Not attempted** |
| Theorems 4.2–4.3 | high-probability static and dynamic bounds | estimator prerequisites tested; full optimistic algorithms not reimplemented | **Partially covered by prerequisites** |
| Theorem 5.2 | universal `Ω(√dT)` lower bound | no finite test can verify the universal proof quantifier | **Not attempted** |
| Conjecture 5.3 | proposed minimax uBLO rate | explicitly open in the paper | **Not scored as a result** |

The original repository baseline printed “6/6” using hard-coded or non-diagnostic checks. It is retained as a control, not treated as reproduction evidence.
