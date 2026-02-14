"""
Model 4: Full Bayesian MCMC on Subsample
=========================================
Framework: outcome[n] ~ Categorical(θ[n])
Link:      θ[n] = softmax(X[n]·β + α)
Priors:    β ~ Normal(0, 1),  α ~ Normal(0, 2)
Inference: NUTS (Hamiltonian Monte Carlo) via Stan

Same model as Model 2, but trained on a representative subsample
of ~15,000 observations for computational feasibility.
Estimated runtime: ~15-30 minutes.

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
SUBSAMPLE_SIZE = 15000

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
    # 1. Load & Subsample
    print("Loading data...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df['outcome'] = df.apply(get_target, axis=1)
    df = df.fillna(0)

    print(f"Full dataset: {len(df)} rows")
    print(f"Subsampling to {SUBSAMPLE_SIZE} rows...")
    df_sub = df.sample(n=SUBSAMPLE_SIZE, random_state=42)

    # Verify subsample preserves outcome distribution
    print(f"\nOutcome distribution (subsample):")
    print(df_sub['outcome'].value_counts(normalize=True))

    X = df_sub[FEATURE_COLS].values
    y = df_sub['outcome'].map(OUTCOME_MAP).values.astype(int)

    # 2. Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    N, P = X_scaled.shape
    K = len(OUTCOME_CATEGORIES)

    print(f"\nData: N={N}, P={P}, K={K}")

    # 3. Prepare Stan Data
    stan_data = {
        'N': N,
        'K': K,
        'P': P,
        'X': X_scaled.tolist(),
        'y': y.tolist()
    }

    with open('stan_data_subsample.json', 'w') as f:
        json.dump(stan_data, f)
    print("Data saved to stan_data_subsample.json")

    # 4. Compile & Fit Stan Model (MCMC on subsample)
    print("\nCompiling Stan model...")
    model = CmdStanModel(stan_file=STAN_FILE)

    print("Running MCMC on subsample (estimated: 15-30 min)...")
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
    fit.save_csvfiles(dir='mcmc_output_subsample')
    print("MCMC output saved to mcmc_output_subsample/")

if __name__ == "__main__":
    main()
