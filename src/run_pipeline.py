#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 19:09:19 2025

@author: vinodhkumargunasekaran
run_pipeline.py

Convenience script to run the full quasi-stationary analysis:

1. Compute transitions.
2. Estimate quasi-stationary composition m and decay rate α.
3. Plot survival curve + heatmap.
4. Plot decision matrix.
"""

import subprocess
import sys


def run(cmd: list[str]):
    print(f"\n=== Running: {' '.join(cmd)} ===")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    run([sys.executable, "src/compute_transitions.py"])
    run([sys.executable, "src/estimate_qsd.py"])
    run([sys.executable, "src/plot_survival_and_heatmap.py"])
    run([sys.executable, "src/plot_decision_matrix.py"])
    print("\nPipeline completed. Check the outputs/ folder for figures and .npy files.")


if __name__ == "__main__":
    main()