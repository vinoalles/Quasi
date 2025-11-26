#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 19:09:19 2025

@author: vinodhkumargunasekaran
"""


#!/usr/bin/env python
"""
simulate_promo_data.py

Generate a 104-week simulated UPC–retailer promotional dataset with
states:
  - TPR
  - Feature
  - Display
  - F+D (Feature + Display)
  - End (absorbing)

and a simple sales model with uplift by state.
"""

import os
import numpy as np
import pandas as pd


def simulate_one_series(
    n_weeks: int,
    trans_matrix: np.ndarray,
    states: list[str],
    start_state: str = "Feature",
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Simulate a single promotion time series.

    Parameters
    ----------
    n_weeks : int
        Number of weeks to simulate.
    trans_matrix : ndarray (n_states x n_states)
        Row-stochastic transition matrix for all states including End.
    states : list[str]
        List of state names in the same order as trans_matrix rows/cols.
    start_state : str
        Initial state (e.g., 'Feature').
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    DataFrame with columns: ['week', 'promo_state', 'sales_units'].
    """
    if seed is not None:
        np.random.seed(seed)

    n_states = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}

    current_state_idx = state_to_idx[start_state]

    base_sales = 200.0
    uplift = {
        "TPR": 40.0,
        "Feature": 70.0,
        "Display": 60.0,
        "F+D": 140.0,
        "End": 0.0,
    }
    sigma = 20.0  # sales noise

    records = []
    for week in range(1, n_weeks + 1):
        state = states[current_state_idx]
        mean_sales = base_sales + uplift[state]
        sales = np.random.normal(loc=mean_sales, scale=sigma)
        sales = max(0.0, sales)

        records.append(
            {
                "week": week,
                "promo_state": state,
                "sales_units": round(sales, 1),
            }
        )

        if state == "End":
            current_state_idx = state_to_idx["End"]
        else:
            probs = trans_matrix[current_state_idx]
            current_state_idx = np.random.choice(n_states, p=probs)

    return pd.DataFrame.from_records(records)


def main():
    # States in the Markov chain
    states = ["TPR", "Feature", "Display", "F+D", "End"]

    # Transition matrix including absorbing End state.
    # Rows: from, columns: to.
    # This is aligned with Table "Example weekly transition probabilities".
    P = np.array(
        [
            [0.55, 0.10, 0.10, 0.10, 0.15],  # from TPR
            [0.15, 0.40, 0.10, 0.15, 0.20],  # from Feature
            [0.10, 0.15, 0.45, 0.10, 0.20],  # from Display
            [0.10, 0.10, 0.20, 0.50, 0.10],  # from F+D
            [0.00, 0.00, 0.00, 0.00, 1.00],  # from End (absorbing)
        ]
    )

    n_weeks = 104
    n_series = 100  # number of independent UPC–retailer series

    all_records: list[pd.DataFrame] = []
    for series_id in range(1, n_series + 1):
        df_series = simulate_one_series(
            n_weeks=n_weeks,
            trans_matrix=P,
            states=states,
            start_state="Feature",
            seed=series_id,  # per-series seed for reproducibility
        )
        df_series.insert(0, "series_id", series_id)
        all_records.append(df_series)

    df = pd.concat(all_records, ignore_index=True)

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "promo_data.csv")
    df.to_csv(out_path, index=False)

    print(f"[simulate_promo_data] Wrote simulated data to {out_path}")
    print(df.head(10))


if __name__ == "__main__":
    main()