"""
Electricity Consumption Anomaly Detection — Flask Application
==============================================================
A simple, clean web application that connects the 8 steps:
1. Upload: Read CSV file
2. Clean: Remove duplicates, blanks, and negative numbers
3. Statistics: Calculate average and spread
4. Detect: Z-score, IQR, and Isolation Forest
5. Compare: Count votes (methods agreed)
6. Show: Send graph and table data to index.html
7. Accuracy: Calculate precision, recall, F1 (if labels exist)
8. Download: Export results to result.csv

Strictly operates ONLY on user-uploaded data — no dummy or hardcoded data.
"""

import os
from flask import Flask, render_template, request, jsonify, send_file

# Import the 8 simple steps
from step1_upload import load_csv
from step2_preprocess import clean_data
from step3_statistics import calculate_statistics
from step4_detection import detect_anomalies
from step5_compare import compare_methods
from step7_evaluate import evaluate_accuracy
from step8_export import export_to_csv

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["RESULT_FILE"] = os.path.join(os.path.dirname(__file__), "results", "result.csv")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(os.path.dirname(app.config["RESULT_FILE"]), exist_ok=True)


@app.route("/")
def index():
    """Renders the dashboard page."""
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def run_pipeline():
    """
    Runs the complete 8-step pipeline strictly on user-uploaded CSV data.
    If no file is provided and no previous file was uploaded, returns a clear error.
    
    3 Settings:
    - z_limit (Z-score limit, default 3.0)
    - iqr_multiplier (IQR multiplier, default 1.5)
    - contamination (Isolation Forest expected outlier share, default 0.05)
    """
    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], "uploaded.csv")
        original_filename = "uploaded.csv"

        # Check if user sent a new file
        if "file" in request.files and request.files["file"].filename != "":
            uploaded_file = request.files["file"]
            original_filename = uploaded_file.filename
            
            if not original_filename.lower().endswith(".csv"):
                return jsonify({
                    "success": False,
                    "error": "Invalid file format. Please upload a valid .csv file."
                }), 400
                
            uploaded_file.save(file_path)
            
        elif not os.path.exists(file_path):
            return jsonify({
                "success": False,
                "error": "No file uploaded. Please select and upload an electricity CSV file."
            }), 400

        # Extract the 3 user settings
        z_limit = float(request.form.get("z_limit", 3.0))
        iqr_multiplier = float(request.form.get("iqr_multiplier", 1.5))
        contamination = float(request.form.get("contamination", 0.05))

        # --- STEP 1: Upload / Load CSV ---
        raw_df = load_csv(file_path)

        # --- STEP 2: Clean Data ---
        clean_df, clean_info = clean_data(raw_df)

        if len(clean_df) == 0:
            return jsonify({
                "success": False,
                "error": "The uploaded file contains no valid electricity records after cleaning."
            }), 400

        # --- STEP 3: Statistics ---
        stats, enriched_df = calculate_statistics(clean_df)

        # --- STEP 4: Detection (Z-score, IQR, Isolation Forest) ---
        detected_df, detect_info = detect_anomalies(
            enriched_df,
            z_limit=z_limit,
            iqr_multiplier=iqr_multiplier,
            contamination=contamination
        )

        # --- STEP 5: Compare Methods (Votes & Consensus) ---
        final_df, compare_info = compare_methods(detected_df)

        # --- STEP 7: Accuracy Evaluation (if labels present) ---
        accuracy_metrics = evaluate_accuracy(final_df)

        # --- STEP 8: Export Result to CSV ---
        export_to_csv(final_df, app.config["RESULT_FILE"])

        # Format records for chart and table
        records = []
        for _, row in final_df.iterrows():
            records.append({
                "date": str(row["timestamp"]),
                "kwh": float(row["consumption_kwh"]),
                "rolling_mean": float(row["rolling_mean"]),
                "z_score": float(row["z_score"]),
                "z_flag": int(row["anomaly_zscore"]),
                "iqr_flag": int(row["anomaly_iqr"]),
                "iforest_flag": int(row["anomaly_iforest"]),
                "votes": int(row["votes"]),
                "flagged": int(row["anomaly_consensus"]),
                "actual": int(row["label"]) if "label" in row else None
            })

        return jsonify({
            "success": True,
            "filename": original_filename,
            "clean_info": clean_info,
            "stats": stats,
            "detect_info": detect_info,
            "compare_info": compare_info,
            "accuracy": accuracy_metrics,
            "rows": records
        })

    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/download")
def download_result():
    """Step 8: Download the generated result.csv from uploaded data."""
    result_path = app.config["RESULT_FILE"]
    if not os.path.exists(result_path):
        return "No results generated yet. Please upload a file and run analysis first.", 404
    return send_file(result_path, as_attachment=True, download_name="result.csv")


@app.route("/api/sample")
def download_sample():
    """Allows downloading a template CSV format."""
    sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "electricity.csv")
    return send_file(sample_path, as_attachment=True, download_name="electricity_template.csv")


if __name__ == "__main__":
    app.run(debug=True)
