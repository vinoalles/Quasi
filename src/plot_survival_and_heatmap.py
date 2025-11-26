#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 19:09:19 2025

@author: vinodhkumargunasekaran
plot_survival_and_heatmap.py

Create:
- Survival curve with exponential fit S(t) = exp(-alpha t).
- Heatmap of Q0_cond transition probabilities.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ACTIVE_STATES = ["TPR", "Feature", "Display", "F+D"]
END_STATE = "End"


def compute_empirical_survival(df: pd.DataFrame):
    """
    Compute empirical survival S(t) = P(T > t) where T is time to End.
    """
    lifetimes = []
    for series_id, sub in df.sort_values(["series_id", "week"]).groupby("series_id"):
        states = sub["promo_state"].tolist()
        T = len(states)
        for i, s in enumerate(states):
            if s == END_STATE:
                T = i + 1
                break
        lifetimes.append(T)

    lifetimes = np.array(lifetimes, dtype=int)
    max_t = lifetimes.max()
    times = np.arange(0, max_t + 1)
    S = np.array([np.mean(lifetimes > t) for t in times])
    return times, S, lifetimes


def main():
    data_path = os.path.join("data", "promo_data.csv")
    Q0_path = os.path.join("outputs", "Q0_cond.npy")
    alpha_path = os.path.join("outputs", "alpha.txt")

    if not os.path.exists(data_path):
        raise FileNotFoundError("Run simulate_promo_data.py first.")
    if not os.path.exists(Q0_path):
        raise FileNotFoundError("Run compute_transitions.py first.")
    if not os.path.exists(alpha_path):
        raise FileNotFoundError("Run estimate_qsd.py first.")

    df = pd.read_csv(data_path)
    Q0_cond = np.load(Q0_path)
    with open(alpha_path, "r") as f:
        alpha_hat = float(f.read().strip())

    os.makedirs("outputs", exist_ok=True)

    # --- Survival curve ------------------------------------------------------
    times, S_emp, lifetimes = compute_empirical_survival(df)
    S_fit = np.exp(-alpha_hat * times)

    plt.figure(figsize=(6, 4))
    plt.step(times, S_emp, where="post", label="Empirical survival")
    plt.plot(times, S_fit, linestyle="--", label=rf"Exponential fit $S(t)=e^{{-\alpha t}}$, $\alpha={alpha_hat:.2f}$")
    plt.xlabel("Weeks since promotion start")
    plt.ylabel("Survival probability S(t)")
    plt.title("Promotion survival curve and exponential fit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "survival_curve.png"), dpi=300)
    plt.close()

    # --- Heatmap of Q0_cond --------------------------------------------------
    plt.figure(figsize=(5, 4))
    df_Q0 = pd.DataFrame(Q0_cond, index=ACTIVE_STATES, columns=ACTIVE_STATES)
    sns.heatmap(df_Q0, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Transition probability heatmap (active states)")
    plt.xlabel("To state")
    plt.ylabel("From state")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "transition_heatmap.png"), dpi=300)
    plt.close()

    print("[plot_survival_and_heatmap] Saved survival_curve.png and transition_heatmap.png in outputs/")


if __name__ == "__main__":
    main()