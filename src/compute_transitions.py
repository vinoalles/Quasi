#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 19:09:19 2025

@author: vinodhkumargunasekaran
compute_transitions.py

Compute empirical transition counts and transition matrices among
active promotion states (TPR, Feature, Display, F+D) from the
simulated dataset.
"""

import os
import numpy as np
import pandas as pd


ACTIVE_STATES = ["TPR", "Feature", "Display", "F+D"]
END_STATE = "End"


def compute_transition_matrices(df: pd.DataFrame):
    """
    Compute transition counts and matrices among active states.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: ['series_id', 'week', 'promo_state'].

    Returns
    -------
    counts : ndarray shape (4, 4)
        Counts of transitions i -> j for i,j in ACTIVE_STATES.
    to_end : ndarray shape (4,)
        Counts of transitions i -> End for i in ACTIVE_STATES.
    Q0_cond : ndarray shape (4, 4)
        Row-stochastic matrix conditional on not transitioning to End.
    """
    state_to_idx = {s: i for i, s in enumerate(ACTIVE_STATES)}

    counts = np.zeros((4, 4), dtype=float)
    to_end = np.zeros(4, dtype=float)

    df_sorted = df.sort_values(["series_id", "week"])

    for series_id, sub in df_sorted.groupby("series_id"):
        states = sub["promo_state"].tolist()
        for s_from, s_to in zip(states[:-1], states[1:]):
            if s_from not in ACTIVE_STATES:
                continue  # ignore transitions from End or unknown

            i = state_to_idx[s_from]

            if s_to == END_STATE:
                to_end[i] += 1
            elif s_to in ACTIVE_STATES:
                j = state_to_idx[s_to]
                counts[i, j] += 1
            else:
                # unexpected state; skip
                continue

    # Conditional transition matrix (given we did NOT go to End)
    row_sums = counts.sum(axis=1, keepdims=True)
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        Q0_cond = np.divide(counts, row_sums, where=row_sums != 0.0)
    # For any inactive row (never observed), default to self-loop
    for i in range(Q0_cond.shape[0]):
        if not np.any(np.isfinite(Q0_cond[i])):
            Q0_cond[i, :] = 0.0
            Q0_cond[i, i] = 1.0

    return counts, to_end, Q0_cond


def main():
    data_path = os.path.join("data", "promo_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Expected {data_path}. Run simulate_promo_data.py first."
        )

    df = pd.read_csv(data_path)

    counts, to_end, Q0_cond = compute_transition_matrices(df)

    print("[compute_transitions] Transition counts among active states:")
    print(pd.DataFrame(counts, index=ACTIVE_STATES, columns=ACTIVE_STATES))

    print("\n[compute_transitions] Transitions to End from each active state:")
    print(pd.Series(to_end, index=ACTIVE_STATES))

    print("\n[compute_transitions] Conditional transition matrix Q0 (rows sum to 1):")
    print(pd.DataFrame(Q0_cond, index=ACTIVE_STATES, columns=ACTIVE_STATES))

    os.makedirs("outputs", exist_ok=True)
    np.save(os.path.join("outputs", "Q0_cond.npy"), Q0_cond)
    np.save(os.path.join("outputs", "trans_counts.npy"), counts)
    np.save(os.path.join("outputs", "trans_to_end.npy"), to_end)
    print("\nSaved Q0_cond and counts to outputs/ as .npy files.")


if __name__ == "__main__":
    main()