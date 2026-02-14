"""
Model 1: Frequentist Multinomial Logistic Regression (MLE)
===========================================================
Framework: outcome ~ Categorical(θ₁...θ₇)
Link:      θ = softmax(Xβ)
Inference: Maximum Likelihood Estimation (no priors)

Outcomes: W, 0, 1, 2, 3, 4, 6  (7 categories)
Data:     ~278,000 ball-by-ball IPL observations
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, classification_report
import joblib

# ── Config ──────────────────────────────────────────────────────────────
DATA_FILE = "ipl_training_data_v3.csv"

# ── Target: Map each ball to one of 7 outcomes ─────────────────────────
def get_target(row):
    if row['is_wicket'] == 1:
        return 'W'
    runs = row['runs_batter']
    if runs in [0, 1, 2, 3, 4, 6]:
        return str(runs)
    return '1'

# ── Feature List ────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Match Context
    'current_score', 'wickets_lost', 'balls_remaining',
    'required_run_rate', 'innings',
    # Batter Recent Form (rolling last 3 matches)
    'batter_form_runs_last_3', 'batter_form_sr_last_3', 'batter_form_bp_last_3',
    # Bowler Recent Form (rolling last ~3 matches)
    'bowler_form_economy_last_3', 'bowler_form_wickets_last_3',
    # Venue
    'venue_avg_runs_last5',
    # Career Skills
    'skill_hard_hitting', 'skill_consistency', 'skill_audacity',
    'skill_clutch', 'skill_chase_iq', 'skill_pace_killer', 'skill_spin_killer',
]

def main():
    # 1. Load & Prepare Data
    print("Loading data...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df['outcome'] = df.apply(get_target, axis=1)
    df = df.fillna(0)

    print(f"Outcome distribution:\n{df['outcome'].value_counts(normalize=True)}\n")

    X = df[FEATURE_COLS]
    y = df['outcome']

    # 2. Scale Features (critical for logistic regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 4. Fit Multinomial Logistic Regression
    #    This maximizes: Σ log P(y_n | X_n, β)
    #    where P(y_n = k | X_n) = softmax(X_n · β_k)
    print(f"Training on {len(FEATURE_COLS)} features...")
    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Log Loss: {log_loss(y_test, y_prob):.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    # 6. Save
    joblib.dump(model, 'model_1_frequentist.pkl')
    joblib.dump(scaler, 'scaler_1_frequentist.pkl')
    print("Model saved.")

    # 7. Feature Importance (coefficients for class '6')
    try:
        idx = list(model.classes_).index('6')
        coefs = model.coef_[idx]
        imp = pd.DataFrame({'Feature': FEATURE_COLS, 'Coef': coefs})
        print(f"\nTop 5 predictors for sixes:")
        print(imp.sort_values('Coef', ascending=False).head(5).to_string(index=False))
    except ValueError:
        pass

if __name__ == "__main__":
    main()
