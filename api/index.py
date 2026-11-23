import os
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

# Determine absolute paths (supports both local running and Vercel serverless deployment)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

def resolve_path(filename):
    """Finds file either in the api/ directory or in project root."""
    api_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(api_path):
        return api_path
    root_path = os.path.join(PROJECT_ROOT, filename)
    if os.path.exists(root_path):
        return root_path
    return api_path

MODEL_PATH = resolve_path('model.joblib')
OPTIONS_PATH = resolve_path('options.json')
PUBLIC_DIR = os.path.join(PROJECT_ROOT, 'public')

# Initialize Flask app
app = Flask(__name__, static_folder=PUBLIC_DIR)

# Load trained pipeline model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# Load valid options
if not os.path.exists(OPTIONS_PATH):
    raise FileNotFoundError(f"Options file not found at {OPTIONS_PATH}")
with open(OPTIONS_PATH, 'r', encoding='utf-8') as f:
    options_data = json.load(f)


def standardize_education(val: str) -> str:
    """Standardizes incoming education string to match model categories."""
    if not isinstance(val, str):
        return str(val)
    val_clean = val.strip()
    val_lower = val_clean.lower()
    if 'bachelor' in val_lower:
        return "Bachelor's"
    elif 'master' in val_lower:
        return "Master's"
    elif 'phd' in val_lower or 'ph.d' in val_lower:
        return "PhD"
    elif 'high school' in val_lower:
        return "High School"
    return val_clean


@app.route('/')
def serve_index():
    """Serves the frontend single-page application."""
    return send_from_directory(PUBLIC_DIR, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serves static files from the public folder."""
    return send_from_directory(PUBLIC_DIR, path)


@app.route('/api/options', methods=['GET'])
def get_options():
    """Returns valid education levels and job titles."""
    return jsonify(options_data), 200


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Accepts JSON: {"education": str, "experience": number, "job_title": str}
    Validates payload and returns predicted salary.
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({
            "error": "Invalid request. Request body must be valid JSON."
        }), 400

    # Check required fields
    required_fields = ['education', 'experience', 'job_title']
    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        return jsonify({
            "error": f"Missing required fields: {', '.join(missing_fields)}"
        }), 400

    education = data.get('education')
    experience = data.get('experience')
    job_title = data.get('job_title')

    # Validate education
    if not isinstance(education, str) or not education.strip():
        return jsonify({
            "error": "Field 'education' must be a non-empty string."
        }), 400

    # Validate job_title
    if not isinstance(job_title, str) or not job_title.strip():
        return jsonify({
            "error": "Field 'job_title' must be a non-empty string."
        }), 400

    # Validate experience (number >= 0)
    # Notice: bool is a subclass of int in Python (isinstance(True, int) == True), so exclude bool explicitly
    if isinstance(experience, bool) or not isinstance(experience, (int, float)):
        return jsonify({
            "error": "Field 'experience' must be a numeric value (integer or float)."
        }), 400

    try:
        exp_val = float(experience)
    except (ValueError, TypeError):
        return jsonify({
            "error": "Field 'experience' could not be parsed as a number."
        }), 400

    if exp_val < 0:
        return jsonify({
            "error": "Field 'experience' must be greater than or equal to 0."
        }), 400

    if exp_val > 60:
        return jsonify({
            "error": "Field 'experience' cannot exceed 60 years."
        }), 400

    # Format input for model pipeline
    norm_education = standardize_education(education)
    norm_job_title = job_title.strip()

    input_df = pd.DataFrame([{
        'Education Level': norm_education,
        'Years of Experience': exp_val,
        'Job Title': norm_job_title
    }])

    try:
        raw_prediction = model.predict(input_df)[0]
        # Predicted salary must never be negative, rounded to 2 decimals
        predicted_salary = max(0.0, round(float(raw_prediction), 2))

        return jsonify({
            "predicted_salary": predicted_salary
        }), 200

    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500


if __name__ == '__main__':
    # Local development server on port 5000
    app.run(host='0.0.0.0', port=5000, debug=False)
