# 💼 Salary Prediction System

A production-ready machine learning system that predicts expected employee compensation based on degree qualification, professional experience, and job title. Designed to empower employees with fair-market compensation transparency and provide HR teams with reliable budgeting estimates.

---

## 📌 Problem Statement

Determining fair compensation in competitive job markets is often opaque, subjective, and prone to demographic bias. Both job seekers and recruiting teams require reliable, data-backed benchmarks reflecting an individual's education, experience, and role. 

This project delivers:
- An end-to-end Machine Learning pipeline utilizing debiased feature selection.
- Multiple evaluated regression models with clear statistical benchmarks ($R^2$ and MAE).
- A lightweight Flask REST API for programmatic predictions.
- A modern, responsive web application for interactive salary estimation.

---

## 📊 Dataset

- **File**: `Salary_Data.csv` (located in the project root)
- **Features in Dataset**: `Age`, `Gender`, `Education Level`, `Job Title`, `Years of Experience`, `Salary`
- **Features Used for Prediction**:
  - `Education Level` (Categorical)
  - `Years of Experience` (Numerical)
  - `Job Title` (Categorical)
- **Target Variable**: `Salary` (Annual salary in USD)
- **Bias Mitigation**: `Gender` is intentionally omitted from model training to prevent algorithmic pay disparity. `Age` is utilized solely during Exploratory Data Analysis (EDA) and removed prior to training to eliminate multicollinearity with professional experience.

---

## ⚙️ Data Preprocessing Pipeline

Data preparation and feature transformation are organized using `scikit-learn` `Pipeline` and `ColumnTransformer`:

1. **Data Cleaning & Normalization**:
   - Dropping missing values (`dropna()`) and duplicate records (`drop_duplicates()`).
   - Normalizing educational qualifications to standard categories (`High School`, `Bachelor's`, `Master's`, `PhD`), unifying historical text variations (such as `"Bachelor's Degree"` and `"Bachelor's"`).
2. **Numeric Feature Scaling**:
   - `StandardScaler` applied to `Years of Experience` to ensure numerical stability and zero-mean, unit-variance properties.
3. **Polynomial Features**:
   - `PolynomialFeatures(degree=2, include_bias=False)` applied to `Years of Experience` to capture diminishing marginal salary returns at higher experience levels.
4. **Categorical Feature Encoding**:
   - `OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=5)` applied to `Education Level` and `Job Title`. This groups low-frequency job titles to avoid high-cardinality dimensionality explosions and gracefully handles unseen categories during production inference without throwing errors.

---

## 🏆 Model Evaluation & Comparison

All models were evaluated on a held-out 20% test set (`random_state=42`) using $R^2$ (variance explained) and MAE (Mean Absolute Error):

| Model | $R^2$ Score | MAE ($) | Status |
| :--- | :---: | :---: | :---: |
| **Polynomial Ridge Regression** | **0.9900** | **$4,382.74** | 🥇 **Best Model (Selected & Serialized)** |
| **Linear Regression** | 0.9895 | $4,447.36$ | Runner Up |
| **Random Forest Regressor** | 0.9793 | $6,347.18$ | 3rd Place |

- **Best Model Selection**: `Polynomial Ridge Regression` achieved the highest $R^2$ (0.9900) and the lowest MAE ($4,382.74).
- **Serialization**: Saved as `model.joblib` using `joblib.dump(compress=3)` (file size: **~2.17 KB**).

---

## 📁 Project Structure

```text
salary-prediction/
│
├── api/
│   └── index.py             # Flask REST API backend & static file server
│
├── public/
│   └── index.html           # Single-file modern frontend (HTML/CSS/JS)
│
├── model.joblib             # Serialized best model pipeline (compress=3)
├── options.json             # Serialized valid Education Levels and Job Titles
├── notebook.ipynb           # End-to-end Jupyter notebook with visualizations
├── Salary_Data.csv          # Raw employee salary dataset
├── requirements.txt         # Pinned production dependencies
├── .gitignore               # Git exclusions for venv, checkpoints, and cache
└── README.md                # Comprehensive project documentation
```

---

## 🚀 How to Run Locally

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Git

### 2. Setup Virtual Environment
```bash
# Navigate to the project root directory
cd salary-prediction

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 3. Launch the Application
Run the Flask server:
```bash
python api/index.py
```

Open your browser and navigate to:
```text
http://localhost:5000
```
The entire application (interactive UI + REST API) is served directly on port `5000`.

---

## 🔌 API Usage (REST Endpoints)

### 1. Health & Options
**Endpoint**: `GET /api/options`

**Example Request**:
```bash
curl -X GET http://localhost:5000/api/options
```

**Response**:
```json
{
  "education_levels": [
    "Bachelor's",
    "High School",
    "Master's",
    "PhD"
  ],
  "job_titles": [
    "Business Analyst",
    "Data Analyst",
    "Data Scientist",
    "Product Manager",
    "Software Engineer"
  ]
}
```

---

### 2. Predict Salary
**Endpoint**: `POST /api/predict`

**Headers**: `Content-Type: application/json`

#### Valid Request Example
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"education": "Bachelor'\''s", "experience": 5.0, "job_title": "Software Engineer"}'
```

**Response (`200 OK`)**:
```json
{
  "predicted_salary": 117498.37
}
```

#### Invalid Request Example (Negative Experience)
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"education": "Master'\''s", "experience": -2.0, "job_title": "Data Scientist"}'
```

**Response (`400 Bad Request`)**:
```json
{
  "error": "Field 'experience' must be greater than or equal to 0."
}
```

---

## ⚠️ Limitations

1. **Geographic Variance**: Compensation is heavily dependent on location, cost of living, and remote work tiers (e.g. San Francisco vs. London vs. Bangalore).
2. **Equity & Stock Compensation**: The current dataset estimates base monetary salary; total compensation (RSUs, performance bonuses, equity vesting) is not captured.
3. **Company Size / Stage**: Early-stage startups often compensate with equity, whereas enterprise corporations offer higher cash bases.

---

## 🔮 Future Improvements

1. **Industry Sector & Domain**: Add features for specific business verticals (e.g., FinTech, Healthcare, E-Commerce, Gaming).
2. **Confidence Intervals & Ranges**: Implement Quantile Regression or prediction intervals to output a fair salary range (`$110k - $125k`) rather than a single point estimate.
3. **Cost-of-Living Adjustment (COLA)**: Integrate city/country location multipliers to provide localized market rates.
