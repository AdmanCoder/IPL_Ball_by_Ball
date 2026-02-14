
import json
import os
import pandas as pd

def get_bowler_styles(base_dir=None):
    """
    Returns a dictionary {player_name: 'Spin' | 'Pace' | 'Unknown'}
    based on player_attributes.json and name fallback.
    """
    if base_dir is None:
        # Assume current working directory
        base_dir = os.getcwd()
        
    json_path = os.path.join(base_dir, "player_attributes.json")
    
    styles = {}
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            for name, details in data.items():
                style_str = str(details.get('bowling_style', '')).lower()
                
                if pd.isna(details.get('bowling_style')):
                    style_str = "nan"
                
                if 'spin' in style_str or 'break' in style_str or 'googly' in style_str or 'chinaman' in style_str or 'orthodox' in style_str:
                    styles[name] = 'Spin'
                elif 'fast' in style_str or 'medium' in style_str or 'seam' in style_str:
                    styles[name] = 'Pace'
                else:
                    # Fallback to name match if attribute is missing
                    styles[name] = classify_bowler_from_name(name)
                    
        except Exception as e:
            print(f"Warning: Could not read player_attributes.json: {e}")
            
    return styles

def classify_bowler_from_name(name):
    """
    Fallback classification based on common names if attributes are missing.
    """
    name_lower = name.lower()
    
    # Extensive Lists (merged from extract_features_v2 and refined)
    spin_indicators = [
        'chahal', 'ashwin', 'jadeja', 'kuldeep', 'rashid', 'bishnoi', 'axar', 'sundar', 
        'moeen', 'santner', 'tahir', 'narine', 'harbhajan', 'murali', 'swann', 'zampa', 
        'shamsi', 'hasaranga', 'shakib', 'mehidy', 'gowtham', 'krunal', 'varun', 
        'chawla', 'ojha', 'noor ahmad', 'shreyas gopal', 'markande', 'livingstone', 'maxwell',
        'badree', 'mishra', 'piyush', 'kumble', 'karn sharma', 'shahbaz ahmed'
    ]
    
    pace_indicators = [
        'bumrah', 'shami', 'siraj', 'archer', 'starc', 'boult', 'rabada', 'nortje', 
        'cummins', 'hazlewood', 'wood', 'ferguson', 'lockie', 'steyn', 'johnson', 
        'lee', 'nehra', 'zaheer', 'umesh', 'ishant', 'bhuvi', 'deepak chahar', 'shardul', 
        'hardik', 'stokes', 'curran', 'woakes', 'arshdeep', 'avesh', 'harshal', 'natarajan', 
        'mohit', 'dinda', 'morkel', 'morris', 'cottrell', 'malinga', 'pathirana', 'deshpande', 
        'khaleel', 'mukesh', 'southee', 'topley', 'behrendorff', 'jansen'
    ]
    
    for s in spin_indicators:
        if s in name_lower: return 'Spin'
    for p in pace_indicators:
        if p in name_lower: return 'Pace'
        
    return 'Unknown'
