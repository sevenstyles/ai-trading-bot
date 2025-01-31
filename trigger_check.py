import pandas as pd
from datetime import datetime
import glob
import os

def check_triggers():
    # Get latest CSV
    latest_csv = max(glob.glob('screener-results/setups_*.csv'), key=os.path.getctime)
    df = pd.read_csv(latest_csv)
    
    new_triggers = df[df['entry_triggered'] == True]
    
    # Compare with previous state
    try:
        prev_df = pd.read_csv('last_state.csv')
        merged = new_triggers.merge(prev_df, how='left', indicator=True)
        new_triggers = merged[merged['_merge'] == 'left_only']
    except:
        pass  # First run
    
    # Save current state
    df.to_csv('last_state.csv', index=False)
    
    return new_triggers

if __name__ == "__main__":
    triggers = check_triggers()
    if not triggers.empty:
        # Send alerts/execute trades
        print("New triggers:", triggers)
        # Add your execution logic here 