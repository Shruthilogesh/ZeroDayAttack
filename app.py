import sys
import os
from flask import Flask, render_template, request, jsonify

# -------------------------------
# FIX PYTHON PATH (VERY IMPORTANT)
# -------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# now Python can see ml/
from ml.detect import detect

# -------------------------------
# FLASK APP SETUP
# -------------------------------
app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

# -------------------------------
# HOME PAGE
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------------
# DETECTION API
# -------------------------------
@app.route("/detect")
def run_detection():
    try:
        threshold = float(request.args.get("threshold", 0.6))

        result_df = detect(threshold)

        return jsonify({
            "results": result_df.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------
# START SERVER
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
