# ⚡ Electricity Consumption Prediction System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg)](https://scikit-learn.org)
[![IEEE 830](https://img.shields.io/badge/Specification-IEEE%20830-green.svg)](https://standards.ieee.org)

An end-to-end Machine Learning Regression System and interactive Streamlit Web Application for forecasting building electricity consumption (`kWh`), designed and implemented in strict compliance with the **IEEE 830 Software Requirements Specification (SRS v1.0)**.

---

## 🌟 Key Features

- **Dataset Ingestion & Validation (`src/data_loader.py`)**: Strict schema validation for all 27 attributes across 12,000 hourly/daily records (Residential, Commercial, Industrial).
- **Modular Preprocessing (`src/preprocessor.py`)**: Scikit-Learn `ColumnTransformer` (median imputation, one-hot encoding, standard scaling) with 70/15/15 stratified train/val/test split.
- **Multi-Model Benchmarking (`src/model_trainer.py`)**: Rigorous evaluation of 7 candidate regressors:
  - Linear Regression, Ridge, Lasso
  - Random Forest Regressor
  - Gradient Boosting Regressor, XGBoost, LightGBM
  - Full metrics suite: $R^2$, MAE, RMSE, MAPE.
- **Lag Feature Transparency (SRS Section 9.3)**: Side-by-side study evaluating models with vs. without autocorrelation lag features (`previous_day_consumption_kWh`, `avg_weekly_consumption_kWh`).
- **Production Inference Engine (`src/predictor.py`)**: Single-record real-time prediction with confidence intervals, tariff cost calculations, and building benchmarks; plus bulk CSV batch inferencing.
- **Interactive Streamlit Web Application (`app.py`)**: 6 rich tabs (Overview, Data Explorer / EDA, Model Benchmarks, Single Prediction, Batch Predictions, IEEE Documentation).
- **Containerized & Deployable**: Pre-configured `Dockerfile`, `.streamlit/config.toml`, `requirements.txt`, and `runtime.txt`.

---

## 📊 Model Performance Summary

| Model | Test $R^2$ | Test MAE (kWh) | Test RMSE (kWh) | Test MAPE (%) | IEEE 830 Acceptance Met |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **0.9991** | **2.75** | **4.09** | **2.68%** | **PASS** ($R^2 \ge 0.85, \text{MAPE} \le 15\%$) |
| **XGBoost** | 0.9991 | 2.76 | 4.11 | 2.67% | **PASS** |
| **LightGBM** | 0.9991 | 2.79 | 4.17 | 2.71% | **PASS** |
| **Linear Regression** | 0.9990 | 3.07 | 4.20 | 3.85% | **PASS** |
| **Ridge Regression** | 0.9990 | 3.07 | 4.20 | 3.82% | **PASS** |
| **Random Forest** | 0.9988 | 3.05 | 4.62 | 2.85% | **PASS** |
| **Lasso Regression** | 0.9989 | 3.11 | 4.49 | 3.51% | **PASS** |

---

## 📁 Repository Structure

```
Electicity_consumption_prediction/
├── electricity_consumption_dataset.csv  # 12,000-record dataset
├── config/
│   └── config.yaml                     # Externalized paths, hyperparams, thresholds
├── src/
│   ├── __init__.py
│   ├── config.py                       # Configuration loader & constants
│   ├── data_loader.py                  # Ingestion & schema validation (FR-1)
│   ├── preprocessor.py                 # Cleaning, encoding, scaling & stratified splitting (FR-2, FR-3)
│   ├── model_trainer.py                # Multi-model training, evaluation & metrics logging (FR-4, FR-5, FR-9, FR-10)
│   └── predictor.py                    # Inference engine for single & batch predictions (FR-6, FR-7)
├── artifacts/
│   ├── models/                         # Serialized model & pipeline artifacts (.joblib)
│   └── metrics/                        # Model comparison logs (JSON, CSV)
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py                # Automated unit test suite
├── .streamlit/
│   └── config.toml                     # Streamlit custom theme & server config
├── app.py                              # Modern interactive Streamlit web application (FR-8, FR-9, FR-6, FR-7, FR-12)
├── Dockerfile                          # Containerization for deployment (Section 8)
├── requirements.txt                    # Pinned production dependencies
├── runtime.txt                         # Documented Python runtime (python-3.12)
└── README.md                           # Documentation & user manual
```

---

## 🚀 Quick Start & Installation

### 1. Clone & Setup Environment
```bash
# Clone the repository
git clone <repository_url>
cd Electicity_consumption_prediction

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train & Benchmark Models
```bash
python -m src.model_trainer
```

### 3. Run Automated Tests
```bash
python -m unittest discover tests
```

### 4. Launch Streamlit Web Application
```bash
streamlit run app.py
```
The application will be accessible at `http://localhost:8501`.

---

## 🐳 Docker Deployment

To build and run the application inside a container:

```bash
docker build -t electricity-consumption-predictor .
docker run -p 8501:8501 electricity-consumption-predictor
```

---

## ☁️ Cloud Deployment Options (SRS Section 8)

1. **Streamlit Community Cloud**:
   - Push repository to GitHub.
   - Connect repository in [share.streamlit.io](https://share.streamlit.io).
   - Set entrypoint to `app.py`.
2. **Hugging Face Spaces**:
   - Create a new Space with Streamlit SDK.
   - Push repository contents.
3. **AWS / GCP / Render**:
   - Deploy using the provided `Dockerfile`.
