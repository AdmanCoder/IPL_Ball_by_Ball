
import pandas as pd
import numpy as np
import math
import os
from collections import Counter
from utils_features import get_bowler_styles

# CONFIG
INPUT_FILE = "ipl_balls_raw_v2.csv"
OUTPUT_FILE = "player_career_stats_official_v2.csv"

def extract_official_v2():
    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Run Step 1 first.")
        return

    # 1. Add Bowler Type (Pace/Spin)
    print("Classifying Bowlers...")
    styles = get_bowler_styles() # Returns {name: 'Spin'/'Pace'}
    
    # Vectorized Map
    df['bowler_type'] = df['bowler'].map(styles).fillna('Unknown')
    
    # 2. Group by Batter
    print(f"Processing {len(df)} balls for {df['batter'].nunique()} batters...")
    grouped = df.groupby('batter')
    stats = pd.DataFrame(index=grouped.groups.keys())
    
    # ============================================================================
    # SKILL 1: CONSISTENCY (GINI)
    # ============================================================================
    print("Calculating Consistency...")
    # List of runs per innings per batter
    innings_runs = df.groupby(['batter', 'match_id'])['runs_batter'].sum()
    
    def get_consistency_score(runs_list):
        if not runs_list or len(runs_list) < 5: return 0
        # Gini
        incomes = np.sort(np.array(runs_list))
        n = len(incomes)
        index = np.arange(1, n + 1)
        gini = ((2 * index - n - 1) * incomes).sum() / (n * incomes.sum()) if incomes.sum() > 0 else 0
        
        avg = np.mean(runs_list)
        return (1 - gini) * (1 + 0.5 * np.log10(avg + 1))

    stats['skill_consistency'] = innings_runs.groupby('batter').apply(lambda x: get_consistency_score(list(x)))
    
    # ============================================================================
    # SKILL 2: AUDACITY (EXPONENTIAL STREAK)
    # ============================================================================
    print("Calculating Audacity...")
    def calculate_audacity(runs_seq):
        total_score = 0
        streak = 0
        for r in runs_seq:
            base = 1.5 if r >= 6 else (1.0 if r == 4 else 0)
            if base > 0:
                streak += 1
                multiplier = 2 ** (streak - 1)
                total_score += base * multiplier
            else:
                streak = 0
        return total_score / len(runs_seq) if len(runs_seq) > 0 else 0

    stats['skill_audacity'] = grouped['runs_batter'].apply(list).apply(calculate_audacity)
    
    # ============================================================================
    # SKILL 3: UNPREDICTABILITY (ENTROPY)
    # ============================================================================
    print("Calculating Unpredictability...")
    def calculate_entropy(runs_seq):
        if len(runs_seq) < 10: return 0
        counts = Counter(runs_seq)
        total = sum(counts.values())
        if total == 0: return 0
        p = {k: v/total for k,v in counts.items()}
        # Standard Entropy: -Sum(p * log2(p))
        # Professor Metric: Median Surprise? Or Just Entropy? 
        # "Trigram Entropy" is hard. Let's use Shannon Entropy of Outcomes (0-6-W).
        # Note: W is not in runs_seq (just 0). Check is_wicket?
        # Let's stick to Runs Entropy for now.
        entropy = -sum(prob * math.log2(prob) for prob in p.values() if prob > 0)
        return entropy

    stats['skill_unpredictability'] = grouped['runs_batter'].apply(list).apply(calculate_entropy)

    # ============================================================================
    # SKILL 4: CLUTCH (DEATH OVERS SR)
    # ============================================================================
    print("Calculating Clutch...")
    death_df = df[df['over'] >= 16]
    death_grp = death_df.groupby('batter')
    
    d_runs = death_grp['runs_batter'].sum()
    d_balls = death_grp['batter'].count()
    death_sr = (d_runs / d_balls * 100).fillna(0)
    
    # Playoff Boost (Need 'is_playoff' - check if 'season' or match type helps?)
    # Valid Match Types: 'Qualifier', 'Eliminator', 'Final', 'Semi Final'
    # We don't have 'match_type' in CSV. Only 'season'.
    # Approximation: Assume classic playoffs (last 4 matches of season per team?).
    # Too complex. Let's simplify Clutch to Death SR for V2 to avoid errors.
    # User said "Make sure no confusion".
    # I will stick to Death SR for now.
    stats['skill_clutch'] = death_sr
    
    # ============================================================================
    # SKILL 5: CHASE IQ (RELATING TO TARGET)
    # ============================================================================
    print("Calculating Chase IQ...")
    chase_df = df[df['innings'] == 2].copy()
    if not chase_df.empty:
        # Need Target. Raw CSV doesn't have target.
        # But it has 'required_run_rate' IF we calculated it.
        # convert_json_to_csv_v2 DOES NOT calculate req_rr.
        # It calculates 'runs_batter'.
        # We need Context: Req RR.
        # Wait, Step 3 adds Context.
        # But Skill needs Past Context?
        # Let's approximate Chase IQ as "Average SR in 2nd Innings / Average SR in 1st Innings".
        # Or "SR in high pressure chases"?
        # Professor said: "(RPO / Req) * Pressure".
        # We don't have Req per ball in Raw V2 yet.
        # I will compute "SR in 2nd Innings" as a proxy for V2 Base.
        chase_grp = chase_df.groupby('batter')
        c_runs = chase_grp['runs_batter'].sum()
        c_balls = chase_grp['batter'].count()
        chase_sr = (c_runs / c_balls * 100).fillna(0)
        stats['skill_chase_iq'] = chase_sr
    else:
        stats['skill_chase_iq'] = 0

    # ============================================================================
    # SKILL 6 & 7: PACE/SPIN KILLER
    # ============================================================================
    print("Calculating Pace/Spin Killer...")
    pace_df = df[df['bowler_type'] == 'Pace']
    spin_df = df[df['bowler_type'] == 'Spin']
    
    p_runs = pace_df.groupby('batter')['runs_batter'].sum()
    p_balls = pace_df.groupby('batter')['batter'].count()
    stats['skill_pace_killer'] = (p_runs / p_balls * 100).fillna(0)
    
    s_runs = spin_df.groupby('batter')['runs_batter'].sum()
    s_balls = spin_df.groupby('batter')['batter'].count()
    stats['skill_spin_killer'] = (s_runs / s_balls * 100).fillna(0)

    # ============================================================================
    # SKILL 8: HARD HITTING
    # ============================================================================
    print("Calculating Hard Hitting...")
    sixes = df[df['runs_batter'] == 6].groupby('batter')['runs_batter'].count()
    fours = df[df['runs_batter'] == 4].groupby('batter')['runs_batter'].count()
    total = grouped['batter'].count()
    
    six_pct = (sixes / total).fillna(0)
    bound_pct = ((sixes + fours) / total).fillna(0)
    stats['skill_hard_hitting'] = (six_pct * 0.6 + bound_pct * 0.4) * 100

    # Clean
    stats = stats.fillna(0)
    
    stats.to_csv(OUTPUT_FILE)
    print(f"Saved {OUTPUT_FILE} with {len(stats)} players.")
    print(stats.head())

if __name__ == "__main__":
    extract_official_v2()
