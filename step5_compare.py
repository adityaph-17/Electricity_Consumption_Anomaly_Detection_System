"""
Step 5: Compare Detection Methods
=================================
Compares the three methods and counts agreement:
- Counts votes for each day (0 to 3 methods).
- Flags a day as a consensus anomaly if at least 2 methods agree.
  (This prevents any single method from triggering a false alarm).
"""

import pandas as pd
from typing import Tuple, Dict, Any


def compare_methods(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Counts how many detection methods flagged each day and determines consensus.
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'anomaly_zscore', 'anomaly_iqr', and 'anomaly_iforest'.
        
    Returns:
        Tuple[pd.DataFrame, dict]: DataFrame with 'votes' and 'anomaly_consensus' columns,
                                  plus comparison summary metrics.
    """
    result_df = df.copy()
    
    # Sum votes across the three detection checks (0, 1, 2, or 3)
    method_columns = ["anomaly_zscore", "anomaly_iqr", "anomaly_iforest"]
    result_df["votes"] = result_df[method_columns].sum(axis=1)
    
    # Flag consensus: at least 2 methods agreed that this day was unusual
    result_df["anomaly_consensus"] = (result_df["votes"] >= 2).astype(int)
    
    # Simple summary breakdown
    three_votes = int((result_df["votes"] == 3).sum())
    two_votes = int((result_df["votes"] == 2).sum())
    one_vote = int((result_df["votes"] == 1).sum())
    consensus_count = int(result_df["anomaly_consensus"].sum())
    
    summary = {
        "consensus_count": consensus_count,
        "three_methods_agreed": three_votes,
        "two_methods_agreed": two_votes,
        "one_method_only": one_vote,
        "total_days": len(result_df)
    }
    
    return result_df, summary


if __name__ == "__main__":
    from step1_upload import load_csv
    from step2_preprocess import clean_data
    from step3_statistics import calculate_statistics
    from step4_detection import detect_anomalies
    
    raw = load_csv("sample_data/electricity.csv")
    cleaned, _ = clean_data(raw)
    _, enriched = calculate_statistics(cleaned)
    detected, _ = detect_anomalies(enriched)
    compared, summary = compare_methods(detected)
    
    print("=== Step 5: Compare Output ===")
    print("Method agreement summary:", summary)
    print("Consensus anomalies (votes >= 2):")
    flagged = compared[compared["anomaly_consensus"] == 1]
    print(flagged[["timestamp", "consumption_kwh", "votes", "anomaly_consensus"]])
