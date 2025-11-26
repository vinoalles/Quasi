#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 19:09:19 2025

@author: vinodhkumargunasekaran
estimate_qsd.py

Estimate quasi-stationary composition m and decay rate α from
simulated promotion data and the conditional transition matrix Q0_cond.
"""

import os
import numpy as np
import pandas as pd
from numpy.linalg import eig


ACTIVE_STATES = ["TPR", "Feature", "Display", "F+D"]
END_STATE = "End"


def estimate_stationary_distribution(Q0_cond: np.ndarray) -> np.ndarray:
    """
    Compute left eigenvector m such that m Q0_cond = m, sum(m) = 1.

    Uses eigen-decomposition of Q0_cond^T.

    Parameters
    ----------
    Q0_cond : ndarray (4x4)

    Returns
    -------
    m : ndarray (4,)
    """
    vals, vecs = eig(Q0_cond.T)
    # Find eigenvalue closest to 1
    idx = np.argmin(np.abs(vals - 1.0))
    v = np.real(vecs[:, idx])
    v = np.maximum(v, 0.0)
    if v.sum() == 0:
        raise ValueError("Stationary distribution eigenvector sums to zero.")
    m = v / v.sum()
    return m


def estimate_alpha_from_lifetimes(df: pd.DataFrame) -> float:
    """
    Estimate decay rate α from empirical lifetimes.

    We define T as the number of weeks until the series first hits End.
    Then α_hat = 1 / E[T] as a simple exponential approximation.

    Parameters
    ----------
    df : DataFrame with columns ['series_id', 'week', 'promo_state']

    Returns
    -------
    alpha_hat : float
    """
    lifetimes = []
    for series_id, sub in df.sort_values(["series_id", "week"]).groupby("series_id"):
        states = sub["promo_state"].tolist()
        # time to first End
        T = len(states)
        for i, s in enumerate(states):
            if s == END_STATE:
                T = i + 1  # weeks counted from 1
                break
        lifetimes.append(T)

    lifetimes = np.array(lifetimes, dtype=float)
    mean_T = lifetimes.mean()
    alpha_hat = 1.0 / mean_T
    return alpha_hat


def main():
    data_path = os.path.join("data", "promo_data.csv")
    Q0_path = os.path.join("outputs", "Q0_cond.npy")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Expected {data_path}. Run simulate_promo_data.py first."
        )
    if not os.path.exists(Q0_path):
        raise FileNotFoundError(
            f"Expected {Q0_path}. Run compute_transitions.py first."
        )

    df = pd.read_csv(data_path)
    Q0_cond = np.load(Q0_path)

    m = estimate_stationary_distribution(Q0_cond)
    alpha_hat = estimate_alpha_from_lifetimes(df)

    print("[estimate_qsd] Quasi-stationary composition m:")
    print(pd.Series(m, index=ACTIVE_STATES))

    print(f"\n[estimate_qsd] Estimated decay rate alpha: {alpha_hat:.3f}")
    print(f"[estimate_qsd] Expected lifetime 1/alpha: {1.0/alpha_hat:.2f} weeks")

    os.makedirs("outputs", exist_ok=True)
    np.save(os.path.join("outputs", "m_qsd.npy"), m)
    with open(os.path.join("outputs", "alpha.txt"), "w") as f:
        f.write(f"{alpha_hat:.6f}\n")


if __name__ == "__main__":
    main()