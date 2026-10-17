"""
Step 8: Export / Download Module
================================
Saves the final analyzed table (with rolling averages, flags, and votes)
to a downloadable CSV file.
"""

import os
import pandas as pd


def export_to_csv(df: pd.DataFrame, output_path: str = "result.csv") -> str:
    """
    Exports the analyzed DataFrame to a CSV file.
    
    Parameters:
        df (pd.DataFrame): Final analyzed DataFrame.
        output_path (str): File path where CSV will be saved.
        
    Returns:
        str: Absolute path of the exported CSV file.
    """
    # Ensure directory exists if path has one
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    df.to_csv(output_path, index=False)
    return os.path.abspath(output_path)


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
    
    out = export_to_csv(compared, "results/test_result.csv")
    print("=== Step 8: Download Output ===")
    print("File saved to:", out)
    print("File size:", os.path.getsize(out), "bytes")
