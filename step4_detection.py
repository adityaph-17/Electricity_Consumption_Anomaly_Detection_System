"""
Step 4: Anomaly Detection Module
=================================
Runs three simple, complementary detection checks on electricity consumption:
1. Z-Score Check: Flags days sitting far above or below the average (|Z| > z_limit).
2. IQR Check: Flags days lying outside Q1 - k*IQR or Q3 + k*IQR.
3. Isolation Forest: Machine learning tree model on 'consumption_kwh' only.

Only 3 configurable settings:
- z_limit (default 3.0)
- iqr_multiplier (default 1.5)
- contamination (default 0.05)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Tuple, Dict, Any


def detect_anomalies(
    df: pd.DataFrame,
    z_limit: float = 3.0,
    iqr_multiplier: float = 1.5,
    contamination: float = 0.05
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies Z-score, IQR, and Isolation Forest detection methods on 'consumption_kwh'.
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'consumption_kwh'.
        z_limit (float): Standard deviation cutoff for Z-score (default 3.0).
        iqr_multiplier (float): Multiplier for IQR fences (default 1.5).
        contamination (float): Expected share of anomalies for Isolation Forest (default 0.05).
        
    Returns:
        Tuple[pd.DataFrame, dict]: Enriched DataFrame with anomaly flag columns and summary counts.
    """
    result_df = df.copy()
    series = result_df["consumption_kwh"]
    
    # 1. Z-Score Method
    mean_val = series.mean()
    std_val = series.std(ddof=1) if len(series) > 1 else 1.0
    if std_val == 0:
        std_val = 1e-9
    
    z_scores = (series - mean_val) / std_val
    result_df["z_score"] = z_scores.round(2)
    result_df["anomaly_zscore"] = (z_scores.abs() > z_limit).astype(int)
    
    # 2. IQR (Interquartile Range) Method
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (iqr_multiplier * iqr)
    upper_bound = q3 + (iqr_multiplier * iqr)
    
    result_df["anomaly_iqr"] = (
        (series < lower_bound) | (series > upper_bound)
    ).astype(int)
    
    # 3. Isolation Forest Method (on 'consumption_kwh' column only)
    # Contamination is the expected fraction of outliers in the data
    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )
    # IsolationForest outputs -1 for outliers, 1 for inliers
    preds = model.fit_predict(result_df[["consumption_kwh"]])
    result_df["anomaly_iforest"] = (preds == -1).astype(int)
    
    summary = {
        "zscore_count": int(result_df["anomaly_zscore"].sum()),
        "iqr_count": int(result_df["anomaly_iqr"].sum()),
        "iforest_count": int(result_df["anomaly_iforest"].sum()),
        "z_limit": z_limit,
        "iqr_multiplier": iqr_multiplier,
        "contamination": contamination
    }
    
    return result_df, summary


if __name__ == "__main__":
    from step1_upload import load_csv
    from step2_preprocess import clean_data
    from step3_statistics import calculate_statistics
    
    raw = load_csv("sample_data/electricity.csv")
    cleaned, _ = clean_data(raw)
    _, enriched = calculate_statistics(cleaned)
    detected, counts = detect_anomalies(enriched)
    
    print("=== Step 4: Detection Output ===")
    print("Anomalies found by each check:", counts)
    print("Flagged days summary (first 5 flagged):")
    flagged = detected[(detected["anomaly_zscore"] == 1) | (detected["anomaly_iqr"] == 1) | (detected["anomaly_iforest"] == 1)]
    print(flagged[["timestamp", "consumption_kwh", "z_score", "anomaly_zscore", "anomaly_iqr", "anomaly_iforest"]].head(5))
