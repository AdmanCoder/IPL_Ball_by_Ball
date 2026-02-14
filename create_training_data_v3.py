import pandas as pd
import numpy as np
from utils_features import get_bowler_styles

# CONFIG
RAW_FILE = "ipl_balls_raw_v2.csv"
SKILLS_FILE = "player_career_stats_official_v2.csv"
BATTER_FORM_FILE = "batter_form_features_v3.csv"
BOWLER_FORM_FILE = "bowler_features_v2.csv" # Actually v2 file content was overwritten with v3 logic
VENUE_FILE = "venue_features_v3.csv"
OUTPUT_FILE = "ipl_training_data_v3.csv"

def create_training_data():
    print("Loading Data...")
    balls = pd.read_csv(RAW_FILE)
    
    # 1. Calculate Match Targets & Required Run Rate (Same as V2)
    print("Calculating Targets and Context...")
    inn1 = balls[balls['innings'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
    inn1.rename(columns={'total_runs': 'target_runs'}, inplace=True)
    inn1['target_runs'] += 1
    balls = pd.merge(balls, inn1, on='match_id', how='left')
    
    balls['balls_remaining'] = 120 - (balls['over'] * 6 + balls['ball'])
    balls['balls_remaining'] = balls['balls_remaining'].clip(lower=1)
    
    balls['required_run_rate'] = np.where(
        balls['innings'] == 2,
        (balls['target_runs'] - balls['current_score']) / balls['balls_remaining'] * 6,
        0
    )
    
    # 2. Previous Ball Boundary (Context)
    balls.sort_values(['match_id', 'innings', 'over', 'ball'], inplace=True)
    balls['prev_runs'] = balls.groupby(['match_id', 'batter'])['runs_batter'].shift(1).fillna(0)
    balls['prev_ball_boundary'] = np.where(balls['prev_runs'] >= 4, 1, 0)
    
    # 3. Bowler Type
    styles = get_bowler_styles()
    balls['bowler_type'] = balls['bowler'].map(styles).fillna('Unknown')
    balls['is_spin'] = balls['bowler_type'].apply(lambda x: 1 if 'Spin' in x else 0)
    
    # 4. Join Static Skills (V2)
    print("Joining Static Skills...")
    skills = pd.read_csv(SKILLS_FILE)
    if 'player' in skills.columns: skills.rename(columns={'player': 'batter'}, inplace=True)
    elif skills.index.name == 'player': 
        skills.reset_index(inplace=True)
        skills.rename(columns={'player': 'batter'}, inplace=True)
    elif 'batter' not in skills.columns:
        skills.rename(columns={skills.columns[0]: 'batter'}, inplace=True)
        
    balls = pd.merge(balls, skills, on='batter', how='left')
    
    # 5. Join Batter Form (V3 NEW)
    print("Joining Batter Form (Recent 3 Matches)...")
    try:
        bat_form = pd.read_csv(BATTER_FORM_FILE)
        balls = pd.merge(balls, bat_form, on=['match_id', 'batter'], how='left')
    except FileNotFoundError:
        print("Warning: Batter Form file not found.")

    # 6. Join Bowler Features (V2/V3 Refined)
    print("Joining Bowler Features (Recent 3 Matches)...")
    try:
        bowl_feats = pd.read_csv(BOWLER_FORM_FILE)
        balls = pd.merge(balls, bowl_feats, on=['match_id', 'bowler'], how='left')
    except FileNotFoundError:
        print("Warning: Bowler Features file not found.")
        
    # 7. Join Venue Features (V3 NEW)
    print("Joining Venue Features...")
    try:
        venue_feats = pd.read_csv(VENUE_FILE)
        # Venue file has match_id, venue. 
        # Balls has match_id, venue.
        # Merge on match_id? Yes, venue is redundant but good for check.
        # Actually venue_feats is UNIQUE per match_id.
        balls = pd.merge(balls, venue_feats, on=['match_id', 'venue'], how='left')
    except FileNotFoundError:
        print("Warning: Venue Features file not found.")

    # 8. Fill NA (New Players / Bowlers / Venues)
    balls.fillna(0, inplace=True)
    
    # 9. Save
    print(f"Saving {len(balls)} rows to {OUTPUT_FILE}...")
    balls.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    create_training_data()
