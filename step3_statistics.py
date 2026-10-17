"""
Step 3: Statistics / Baseline Analysis
======================================
Calculates summary statistics for electricity consumption:
- Average (Mean)
- Median (Middle value)
- Spread (Standard Deviation)
- Minimum & Maximum consumption
- Quartiles (Q1, Q3) and Interquartile Range (IQR)
- 7-day rolling moving average to show normal baseline trend
"""

import pandas as pd
from typing import Tuple, Dict, Any


def calculate_statistics(
    df: pd.DataFrame,
    rolling_window: int = 7
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Computes summary statistics and adds a rolling moving average trend.
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'consumption_kwh'.
        rolling_window (int): Number of days for smoothed trend line (default 7).
        
    Returns:
        Tuple[dict, pd.DataFrame]: Dictionary of statistical metrics and enriched DataFrame.
    """
    enriched_df = df.copy()
    series = enriched_df["consumption_kwh"]
    
    # 1. Basic descriptive metrics
    count = int(len(series))
    mean_val = float(series.mean())
    median_val = float(series.median())
    std_val = float(series.std(ddof=1)) if count > 1 else 0.0
    min_val = float(series.min())
    max_val = float(series.max())
    
    # 2. Quartiles and IQR
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr_val = float(q3 - q1)
    
    # 3. Add 7-day rolling moving average to see weekly trend
    enriched_df["rolling_mean"] = (
        series.rolling(window=rolling_window, min_periods=1)
        .mean()
        .round(2)
    )
    
    stats = {
        "count": count,
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "std_dev": round(std_val, 2),
        "min": round(min_val, 2),
        "max": round(max_val, 2),
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr_val, 2),
        "rolling_window": rolling_window
    }
    
    return stats, enriched_df


if __name__ == "__main__":
    from step1_upload import load_csv
    from step2_preprocess import clean_data
    
    raw = load_csv("sample_data/electricity.csv")
    cleaned, _ = clean_data(raw)
    stats, enriched = calculate_statistics(cleaned)
    
    print("=== Step 3: Statistics Output ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("First 3 rows with rolling_mean:\n", enriched[["timestamp", "consumption_kwh", "rolling_mean"]].head(3))
