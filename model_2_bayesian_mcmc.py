"""
Model 2: Full Bayesian MCMC (via CmdStanPy)
=============================================
Framework: outcome[n] ~ Categorical(θ[n])
Link:      θ[n] = softmax(X[n]·β + α)
Priors:    β ~ Normal(0, 1),  α ~ Normal(0, 2)
Inference: NUTS (Hamiltonian Monte Carlo) via Stan

Requires: ipl_categorical.stan (Stan model file)
Requires: pip install cmdstanpy

NOTE: This is computationally intensive on full data (~278K rows).
      Estimated runtime: several hours. Consider Model 4 (subsample) instead.
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

# ── Outcome Encoding (Stan uses 1-indexed integers) ────────────────────
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

    # Save data as JSON (can also be loaded by R)
    with open('stan_data_full.json', 'w') as f:
        json.dump(stan_data, f)
    print("Data saved to stan_data_full.json")

    # 4. Compile & Fit Stan Model (MCMC)
    print("Compiling Stan model...")
    model = CmdStanModel(stan_file=STAN_FILE)

    print("Running MCMC (this will take a long time on full data)...")
    fit = model.sample(
        data=stan_data,
        chains=4,
        iter_warmup=500,
        iter_sampling=1000,
        seed=42,
        show_console=True
    )

    # 5. Results
    print("\nPosterior Summary (beta coefficients):")
    print(fit.summary())

    # Save
    fit.save_csvfiles(dir='mcmc_output_full')
    print("MCMC output saved to mcmc_output_full/")

if __name__ == "__main__":
    main()
