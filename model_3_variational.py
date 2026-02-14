"""
Model 3: Variational Inference (ADVI) — Fast Bayesian Approximation
=====================================================================
Framework: outcome[n] ~ Categorical(θ[n])
Link:      θ[n] = softmax(X[n]·β + α)
Priors:    β ~ Normal(0, 1),  α ~ Normal(0, 2)
Inference: Automatic Differentiation Variational Inference (ADVI)

Same Bayesian model as Model 2, but approximates the posterior using
optimization instead of MCMC sampling. Much faster (minutes vs hours).

Requires: ipl_categorical.stan (Stan model file)
Requires: pip install cmdstanpy
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from cmdstanpy import CmdStanModel
import json

# ── Config ──────────────────────────────────────────────────────────────
DATA_FILE = "ipl_training_data_v3.csv"
STAN_FILE = "ipl_categorical.stan"

# ── Target ──────────────────────────────────────────────────────────────
def get_target(row):
    if row['is_wicket'] == 1:
        return 'W'
    runs = row['runs_batter']
    if runs in [0, 1, 2, 3, 4, 6]:
        return str(runs)
    return '1'

# ── Feature List ────────────────────────────────────────────────────────
FEATURE_COLS = [
    'current_score', 'wickets_lost', 'balls_remaining',
    'required_run_rate', 'innings',
    'batter_form_runs_last_3', 'batter_form_sr_last_3', 'batter_form_bp_last_3',
    'bowler_form_economy_last_3', 'bowler_form_wickets_last_3',
    'venue_avg_runs_last5',
    'skill_hard_hitting', 'skill_consistency', 'skill_audacity',
    'skill_clutch', 'skill_chase_iq', 'skill_pace_killer', 'skill_spin_killer',
]

OUTCOME_CATEGORIES = ['W', '0', '1', '2', '3', '4', '6']
OUTCOME_MAP = {cat: i + 1 for i, cat in enumerate(OUTCOME_CATEGORIES)}

def main():
    # 1. Load & Prepare Data
    print("Loading data...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df['outcome'] = df.apply(get_target, axis=1)
    df = df.fillna(0)

    X = df[FEATURE_COLS].values
    y = df['outcome'].map(OUTCOME_MAP).values.astype(int)

    # 2. Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    N, P = X_scaled.shape
    K = len(OUTCOME_CATEGORIES)

    print(f"Data: N={N}, P={P}, K={K}")

    # 3. Prepare Stan Data
    stan_data = {
        'N': N,
        'K': K,
        'P': P,
        'X': X_scaled.tolist(),
        'y': y.tolist()
    }

    # 4. Compile & Fit using Variational Inference
    print("Compiling Stan model...")
    model = CmdStanModel(stan_file=STAN_FILE)

    print("Running Variational Inference (ADVI)...")
    fit = model.variational(
        data=stan_data,
        seed=42,
        output_samples=2000,    # Number of approximate posterior samples
        algorithm='meanfield'   # Faster; use 'fullrank' for better approximation
    )

    # 5. Results
    print("\nVariational Posterior Summary:")
    print(fit.summary())

    # Save
    fit.save_csvfiles(dir='vi_output')
    print("VI output saved to vi_output/")

if __name__ == "__main__":
    main()
