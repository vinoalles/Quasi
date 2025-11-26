#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 19:09:19 2025

@author: vinodhkumargunasekaran
plot_decision_matrix.py

Plot a decision matrix for promotion duration and decay rate α,
mirroring the conceptual figure and table in the paper.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Example categories & alphas (similar to Table tab:sct2/tab:manager)
    categories = ["Beverages", "Snacks", "Personal Care", "Household", "Dairy"]
    alpha = np.array([0.14, 0.22, 0.18, 0.26, 0.12])
    life = 1.0 / alpha

    # Regions: Sustain, Rotate, Terminate, Redesign
    # 0.10–0.14, 0.15–0.20, 0.21–0.28, >0.28
    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(6, 4))
    # Region bands in α
    plt.axvspan(0.10, 0.14, alpha=0.1, label="Sustain")
    plt.axvspan(0.15, 0.20, alpha=0.1, color="tab:orange", label="Rotate")
    plt.axvspan(0.21, 0.28, alpha=0.1, color="tab:red", label="Terminate")

    # Scatter points
    for cat, a, L in zip(categories, alpha, life):
        plt.scatter(a, L)
        plt.text(a, L + 0.1, cat, ha="center", fontsize=8)

    plt.xlabel(r"Decay rate $\alpha$")
    plt.ylabel(r"Expected lifetime $1/\alpha$ (weeks)")
    plt.title("Decision matrix for promotion duration vs. decay rate")
    plt.xlim(0.08, 0.30)
    plt.ylim(0, max(life) + 2)

    handles, labels = plt.gca().get_legend_handles_labels()
    # Deduplicate labels
    uniq = dict(zip(labels, handles))
    plt.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "decision_matrix.png"), dpi=300)
    plt.close()

    print("[plot_decision_matrix] Saved decision_matrix.png in outputs/")


if __name__ == "__main__":
    main()