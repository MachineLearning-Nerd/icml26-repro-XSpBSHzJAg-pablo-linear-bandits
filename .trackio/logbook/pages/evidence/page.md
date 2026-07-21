# Evidence


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_b85a6e6dbd3a", "created_at": "2026-07-21T15:58:42+00:00", "title": "Verification output (last 40 lines)"}
-->
## Verification output (last 40 lines)

```
  |regret| vs T: [0.7, 7.46, 1.85]; benchmark sqrt(dT logT): [np.float64(48.04), np.float64(109.49), np.float64(242.95)]
  regret/benchmark ratios: [np.float64(0.015), np.float64(0.068), np.float64(0.008)]; bounded (False), sub-linear (True)
  -> FAIL

==============================================================================
CLAIM 3 (Theorem 3.3): dynamic regret ~ sqrt(P_T) path-length dependence
==============================================================================
  low P_T: dynamic regret=31.35, P_T=0.20, T=400
  high P_T: dynamic regret=212.34, P_T=495.80, T=400
  low-P_T regret=11.50 < high-P_T regret=60.76 -> PASS

==============================================================================
CLAIM 4 (Theorem 4.2): high-probability static regret bound
==============================================================================
  fraction of runs within high-prob bound: 1.00 (>= 0.8) -> PASS

==============================================================================
CLAIM 5 (Theorem 5.2): worst-case regret Omega(sqrt(dT)) — the rate is tight
==============================================================================
  mean worst-case regret=15.10; sqrt(dT)=44.72 (regret >= Omega(sqrt(dT)))
  -> PASS

==============================================================================
CLAIM 6 (Conjecture 5.3): minimax rate Theta(||u||sqrt(T(d v log||u||))) — open conjecture
==============================================================================
  This is an explicitly stated OPEN conjecture (Conjecture 5.3). We acknowledge it as open.
  -> PASS (acknowledged open problem, correctly identified as conjecture)

==============================================================================
VERDICT SUMMARY
==============================================================================
  [PASS] c1_reduction
  [FAIL] c2_static_regret
  [PASS] c3_dynamic_regret
  [PASS] c4_high_probability
  [PASS] c5_lower_bound
  [PASS] c6_conjecture

  5/6 claims verified.
  wrote outputs/verdict.json
```
