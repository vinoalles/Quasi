# Quasi-Stationary Dynamics of CPG Promotions

This repository contains the simulated dataset and Python code used in the paper:

> **Quasi-Stationary Promotion Modeling: Measuring the Lifespan and Effectiveness of Marketing Promotions**

The code reproduces:
- The simulated 104-week promotion dataset (`promo_data.csv`)
- Empirical transition matrices across promotion states
- Quasi-stationary composition `m`
- Survival curves and fitted exponential decay
- Transition heatmaps
- Decision matrix for promotion duration versus decay rate

## Repository layout

- `data/`
  - `promo_data.csv` — 104-week simulated UPC–retailer-level promotion history with sales.
- `src/`
  - `simulate_promo_data.py` — generates the simulated data.
  - `compute_transitions.py` — computes empirical transition matrices.
  - `estimate_qsd.py` — estimates quasi-stationary composition `m` and decay rate `α`.
  - `plot_survival_and_heatmap.py` — survival curve (Fig. 7-style) and transition heatmap (Fig. 8-style).
  - `plot_decision_matrix.py` — decision matrix by `α` region (Fig. 9-style).
  - `run_pipeline.py` — convenience script to run everything end-to-end.
- `requirements.txt` — Python dependencies.

## Quick start

```bash
git clone https://github.com/vinoalles/Quasi.git
cd Quasi

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
python src/simulate_promo_data.py
python src/run_pipeline.py
