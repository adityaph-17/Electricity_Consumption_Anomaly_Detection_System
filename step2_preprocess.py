"""
Step 2: Clean / Preprocess Data
===============================
Cleans the raw electricity DataFrame:
1. Converts timestamps into datetime and removes unparseable dates.
2. Removes duplicate timestamp rows.
3. Removes negative consumption values (which are physical errors).
4. Fills any missing/blank consumption values using linear interpolation.
5. Sorts the data chronologically.
"""

import pandas as pd
from typing import Tuple, Dict, Any


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw electricity records.
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'timestamp' and 'consumption_kwh'.
        
    Returns:
        Tuple[pd.DataFrame, dict]: Cleaned DataFrame and summary of actions taken.
    """
    clean_df = df.copy()
    initial_count = len(clean_df)
    
    # 1. Parse timestamps and drop invalid dates
    clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], errors="coerce")
    invalid_dates = int(clean_df["timestamp"].isna().sum())
    clean_df = clean_df.dropna(subset=["timestamp"])
    
    # 2. Remove duplicate dates
    duplicates_count = int(clean_df.duplicated(subset=["timestamp"]).sum())
    clean_df = clean_df.drop_duplicates(subset=["timestamp"], keep="first")
    
    # 3. Handle negative consumption values (drop invalid negative numbers)
    negative_mask = clean_df["consumption_kwh"] < 0
    negatives_count = int(negative_mask.sum())
    clean_df = clean_df[~negative_mask]
    
    # 4. Handle missing/blank consumption numbers (interpolate smoothly)
    missing_count = int(clean_df["consumption_kwh"].isna().sum())
    clean_df["consumption_kwh"] = clean_df["consumption_kwh"].interpolate(method="linear").bfill().ffill()
    
    # 5. Sort chronologically
    clean_df = clean_df.sort_values(by="timestamp").reset_index(drop=True)
    
    # Convert timestamp back to string format YYYY-MM-DD for JSON/display
    clean_df["timestamp"] = clean_df["timestamp"].dt.strftime("%Y-%m-%d")
    
    summary = {
        "initial_rows": initial_count,
        "invalid_dates_dropped": invalid_dates,
        "duplicates_removed": duplicates_count,
        "negatives_removed": negatives_count,
        "missing_interpolated": missing_count,
        "final_rows": len(clean_df)
    }
    
    return clean_df, summary


if __name__ == "__main__":
    from step1_upload import load_csv
    sample_file = "sample_data/electricity.csv"
    raw_data = load_csv(sample_file)
    cleaned, clean_summary = clean_data(raw_data)
    
    print("=== Step 2: Clean Output ===")
    print("Cleaning actions:", clean_summary)
    print("First 3 cleaned rows:\n", cleaned.head(3))
