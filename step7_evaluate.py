"""
Step 7: Accuracy / Evaluation Module
====================================
Evaluates detection performance against ground-truth labels (if present).
Calculates:
- Precision: Out of all days flagged as unusual, how many were truly unusual?
- Recall: Out of all truly unusual days, how many did we catch?
- F1-Score: Harmonic balance between precision and recall.
- Accuracy: Overall percentage of correctly classified days.
- Confusion Matrix: True Positives (TP), False Positives (FP),
                    False Negatives (FN), True Negatives (TN).
"""

import pandas as pd
from typing import Dict, Any, Optional


def evaluate_accuracy(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Compares detection flags against the ground-truth 'label' column.
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'label' and 'anomaly_consensus'.
        
    Returns:
        dict or None: Accuracy metrics dictionary, or None if no 'label' column exists.
    """
    if "label" not in df.columns:
        return None
        
    y_true = df["label"].astype(int)
    y_pred = df["anomaly_consensus"].astype(int)
    
    # Calculate confusion matrix components
    tp = int(((y_true == 1) & (y_pred == 1)).sum())  # True Positives: correctly flagged
    fp = int(((y_true == 0) & (y_pred == 1)).sum())  # False Positives: false alarm
    fn = int(((y_true == 1) & (y_pred == 0)).sum())  # False Negatives: missed anomaly
    tn = int(((y_true == 0) & (y_pred == 0)).sum())  # True Negatives: correctly normal
    
    total = len(df)
    
    # Metrics formulas
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    return {
        "has_label": True,
        "total_anomalies_in_data": int(y_true.sum()),
        "correctly_caught_tp": tp,
        "false_alarms_fp": fp,
        "missed_anomalies_fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "accuracy": round(accuracy * 100, 1)
    }


if __name__ == "__main__":
    from step1_upload import load_csv
    from step2_preprocess import clean_data
    from step3_statistics import calculate_statistics
    from step4_detection import detect_anomalies
    from step5_compare import compare_methods
    
    raw = load_csv("sample_data/electricity.csv")
    cleaned, _ = clean_data(raw)
    _, enriched = calculate_statistics(cleaned)
    detected, _ = detect_anomalies(enriched)
    compared, _ = compare_methods(detected)
    metrics = evaluate_accuracy(compared)
    
    print("=== Step 7: Accuracy Output ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
