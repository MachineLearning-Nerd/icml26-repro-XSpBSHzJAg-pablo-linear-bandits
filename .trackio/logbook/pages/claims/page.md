# Claims


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_4dd71e46b6f7", "created_at": "2026-07-21T15:58:41+00:00", "title": "Claims to reproduce"}
-->
## Claims to reproduce

1. The paper's perturbation-based algorithm, PABLO, reduces unconstrained Bandit Linear Optimization (uBLO) to a standard Online Linear Optimization (OLO) problem (Section 3).
2. Theorem 3.1 gives an expected static-regret bound of the form E[R_T(u)] = O~(G*epsilon + d*kappa*E[||u||*sqrt(V_T log(...))]), with kappa=sqrt(d) in the norm-oblivious setting and kappa=1 in the norm-adaptive setting (Theorem 3.1).
3. Theorem 3.3 shows PABLO attains the optimal sqrt(P_T) dynamic-regret dependence on the path-length P_T without requiring prior knowledge of P_T (Theorem 3.3).
4. Theorem 4.2 establishes a high-probability static-regret bound R_T(u) <= O~(dG(epsilon+||u||)log(T/delta) + G||u||sqrt(dT log(T/delta))) (Theorem 4.2).
5. Theorem 5.2 proves the folklore Omega(sqrt(dT)) minimax lower bound for adversarial linear bandits on the unit Euclidean ball (Theorem 5.2).
6. Conjecture 5.3 posits a minimax rate of R_T(u) = Theta(||u|| sqrt(T(d v log||u||))) for norm-oblivious comparators, left as an open problem (Conjecture 5.3).
