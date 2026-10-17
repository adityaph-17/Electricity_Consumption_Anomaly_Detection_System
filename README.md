# Electricity Consumption Anomaly Detection

A clean, easy-to-understand web application built with **Flask, Pandas, NumPy, and Scikit-Learn** to detect abnormal daily electricity consumption (spikes and drops) using three complementary methods:
1. **Z-Score**: Flags values sitting far from the average.
2. **IQR (Interquartile Range)**: Flags values outside the middle 50% routine range.
3. **Isolation Forest**: Machine learning tree model on `consumption_kwh` to isolate unusual readings.

---

## 📁 Project Structure

```text
mini_project/
├── app.py                  # Flask web server & pipeline orchestrator
├── step1_upload.py         # Step 1: Read and load CSV
├── step2_preprocess.py     # Step 2: Clean duplicates, missing, and negative values
├── step3_statistics.py     # Step 3: Calculate average, median, std dev, rolling mean
├── step4_detection.py      # Step 4: Run Z-score, IQR, and Isolation Forest
├── step5_compare.py        # Step 5: Count agreement votes (consensus)
├── step7_evaluate.py       # Step 7: Accuracy, precision, recall, F1 (if labels exist)
├── step8_export.py         # Step 8: Save result.csv
├── templates/
│   └── index.html          # Clean single-page dashboard
├── static/
│   ├── style.css           # Simple stylesheet (no complex bloat)
│   └── script.js           # Lightweight client logic & Chart.js graph
├── sample_data/
│   ├── electricity.csv             # Realistic sample with test labels
│   └── electricity_unlabelled.csv  # Unlabelled sample
└── requirements.txt        # Dependencies (Flask, pandas, numpy, scikit-learn)
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Application
```bash
python app.py
```

### 3. Open in Your Browser
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## ⚙️ The 3 User Settings

1. **Z-Score Limit (`z_limit`)**: Cutoff for standard deviations from the mean (default: `3.0`).
2. **IQR Multiplier (`iqr_multiplier`)**: Multiplier for quartile fences (default: `1.5`).
3. **Expected Share of Anomalies (`contamination`)**: Outlier fraction for Isolation Forest (default: `0.05` = 5%).

---

## 🔍 The Flow (Example)

Say your file has these days: `30, 32, 31, 29, 95, 30 kWh`.

| Step | What happens | Example |
|---|---|---|
| **1. Upload** | Read the CSV | 6 rows loaded |
| **2. Clean** | Remove duplicates, blanks, and negative values | Nothing to fix |
| **3. Statistics** | Work out the average and the spread | Average is about 41 |
| **4. Detect** | Z-score, IQR, and Isolation Forest each flag far-off days | All three flag `95` |
| **5. Compare** | Count how many methods flagged each day | `95` gets 3 votes |
| **6. Show** | Draw the graph and the table, with flagged days highlighted | `95` is in red |
| **7. Accuracy** | Only if the file has a label column, compare flags with the labels | Precision, recall, F1 |
| **8. Download** | Save the table as a CSV | `result.csv` |
