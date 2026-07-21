"""Clean-room PABLO (Perturbation Approach to unconstrained Bandit Linear Optimization)
from "A Perturbation Approach to Unconstrained Linear Bandits" (arXiv 2603.28201). numpy, CPU.

Algorithm 1: H_t=I => s_t = ±e_i (random coordinate), play w_hat = w + s_t, observe <l_t, w_hat>,
  loss estimate l_hat = d * s_t * <l_t, w_hat> (1-point bandit gradient estimate). OLO = OGD.
Theorem 3.1: E[R_T(u)] = O(d^0.5 * ||u|| * sqrt(T log T)) static regret (norm-oblivious, kappa=sqrt(d)).
"""
from __future__ import annotations
import numpy as np


def pablo(losses, d, T, eta, seed=0):
    """PABLO with coordinate perturbation (H=I) + OGD as OLO. Returns (actions, regrets_vs_zero)."""
    rng = np.random.default_rng(seed)
    w = np.zeros(d)
    actions = np.zeros((T, d))
    for t in range(T):
        s = np.zeros(d); s[rng.integers(d)] = rng.choice([-1, 1])
        what = w + s
        reward = float(losses[t] @ what)
        l_hat = d * s * reward                    # loss estimate (1-point bandit)
        w = w - eta * l_hat                       # OGD update
        actions[t] = what
    return actions


def static_regret(actions, losses, u):
    """R_T(u) = sum_t <l_t, u - a_t>."""
    return float(np.sum(losses * (u - actions).T).sum(axis=0) if False else
                 np.sum([losses[t] @ (u - actions[t]) for t in range(len(actions))]))


def run_experiment(d, T, seed=0, n_seeds=10):
    """Run PABLO on adversarial linear bandits; return avg static regret and sqrt(dT) benchmark."""
    rng = np.random.default_rng(seed)
    total_regret = 0.0
    u = rng.standard_normal(d); u /= np.linalg.norm(u) * 2   # comparator
    for s in range(n_seeds):
        losses = rng.standard_normal((T, d)) * 0.5            # adversarial losses
        actions = pablo(losses, d, T, eta=0.01, seed=s * 100 + seed)
        total_regret += static_regret(actions, losses, u)
    return total_regret / n_seeds


def dynamic_regret(actions, losses, u_path):
    """Dynamic regret: sum_t <l_t, u_t - a_t> where u_path is the comparator sequence."""
    T = len(actions)
    return float(np.sum([losses[t] @ (u_path[t] - actions[t]) for t in range(T)]))
