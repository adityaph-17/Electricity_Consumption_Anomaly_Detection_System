"""
Step 1: Upload / Load CSV Data
==============================
Reads the CSV file, identifies the date and consumption columns,
and returns a clean pandas DataFrame with standardized column names:
- 'timestamp'
- 'consumption_kwh'
- 'label' (optional, if present in file)
"""

import pandas as pd


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Reads an electricity CSV file and standardizes column names.
    
    Parameters:
        file_path (str): Path to the uploaded CSV file.
        
    Returns:
        pd.DataFrame: DataFrame with standardized columns ['timestamp', 'consumption_kwh']
                      and optionally ['label'].
    """
    # Read the raw CSV file
    df = pd.read_csv(file_path)
    
    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    # Find the date/timestamp column (case-insensitive)
    cols_lower = {col.strip().lower(): col for col in df.columns}
    
    date_col = None
    for candidate in ["timestamp", "date", "datetime", "time", "reading_date"]:
        if candidate in cols_lower:
            date_col = cols_lower[candidate]
            break
    if not date_col:
        # Fall back to the very first column
        date_col = df.columns[0]
        
    # Find the electricity consumption column
    cons_col = None
    for candidate in ["consumption_kwh", "consumption", "kwh", "usage", "units", "power", "reading"]:
        if candidate in cols_lower:
            cons_col = cols_lower[candidate]
            break
    if not cons_col:
        # Fall back to the second column
        if len(df.columns) > 1:
            cons_col = df.columns[1]
        else:
            raise ValueError("Could not find a consumption/kWh column in CSV.")
            
    # Find optional ground-truth label column
    label_col = None
    for candidate in ["label", "is_anomaly", "anomaly", "actual"]:
        if candidate in cols_lower:
            label_col = cols_lower[candidate]
            break

    # Build standardized DataFrame
    standard_df = pd.DataFrame()
    standard_df["timestamp"] = df[date_col].astype(str)
    standard_df["consumption_kwh"] = pd.to_numeric(df[cons_col], errors="coerce")
    
    if label_col:
        standard_df["label"] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
        
    return standard_df


if __name__ == "__main__":
    # Test Step 1 directly on the sample file
    sample_file = "sample_data/electricity.csv"
    data = load_csv(sample_file)
    print("=== Step 1: Load Output ===")
    print(f"Total rows loaded: {len(data)}")
    print("Columns:", list(data.columns))
    print("First 3 rows:\n", data.head(3))
