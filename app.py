"""
Electricity Consumption Prediction System - Streamlit Web Application.
Compliant with IEEE 830 SRS Specification v1.0 (FR-1 through FR-12).
"""

import os
import json
import io
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.config import (
    DEFAULT_DATASET_PATH,
    MODELS_DIR,
    METRICS_DIR,
    SCHEMA_COLUMNS,
    CATEGORICAL_CHOICES,
    NUMERIC_RANGES,
    TARGET_COLUMN,
    DROP_COLUMNS
)
from src.data_loader import load_dataset, validate_dataset, get_dataset_summary
from src.preprocessor import get_feature_lists
from src.model_trainer import train_and_evaluate_pipeline, run_lag_impact_comparison
from src.predictor import Predictor

# --- Page Configuration ---
st.set_page_config(
    page_title="Electricity Consumption Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling & Theme Tokens ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .badge-primary {
        background-color: #2563EB;
        color: #FFFFFF;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 8px;
    }
    
    .badge-success {
        background-color: #059669;
        color: #FFFFFF;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1E293B, #111827);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.5);
    }
    
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #38BDF8;
        line-height: 1.2;
    }
    
    .metric-lbl {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    
    .pred-highlight-box {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.15) 0%, rgba(14, 165, 233, 0.1) 100%);
        border: 1px solid rgba(56, 189, 248, 0.4);
        border-radius: 16px;
        padding: 28px;
        margin: 20px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0F172A;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 500;
        color: #94A3B8;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
        border: 1px solid rgba(59, 130, 246, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Data and Model Resource Loaders ---
@st.cache_data
def get_cached_dataset() -> pd.DataFrame:
    """Load and cache the historical electricity dataset."""
    return load_dataset(DEFAULT_DATASET_PATH)


@st.cache_resource
def get_cached_predictor() -> Predictor:
    """Load and cache the production inference predictor."""
    # Ensure artifacts exist
    if not (MODELS_DIR / "best_model_pipeline.joblib").exists():
        train_and_evaluate_pipeline(save_artifacts=True)
    return Predictor()


@st.cache_data
def get_cached_metrics(with_lags: bool = True) -> dict:
    """Load cached model comparison metrics."""
    metrics_file = METRICS_DIR / ("model_comparison_metrics.json" if with_lags else "model_comparison_no_lags.json")
    if not metrics_file.exists():
        train_and_evaluate_pipeline(include_lags=with_lags, save_artifacts=True)
    with open(metrics_file, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def get_cached_lag_impact() -> dict:
    """Load lag feature impact comparison."""
    impact_file = METRICS_DIR / "lag_features_impact.json"
    if not impact_file.exists():
        run_lag_impact_comparison()
    with open(impact_file, "r", encoding="utf-8") as f:
        return json.load(f)


# --- Initialize Resources ---
df_raw = get_cached_dataset()
predictor = get_cached_predictor()

# --- Sidebar ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?auto=format&fit=crop&w=600&q=80", use_container_width=True)
    st.markdown("### ⚡ **EnergySense ML Engine**")
    st.markdown("<span class='badge-primary'>IEEE 830 Spec</span> <span class='badge-success'>v1.0 Ready</span>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("#### 🎯 **System Highlights**")
    st.markdown("""
    - **Dataset**: 12,000 hourly/daily records
    - **Target**: Electricity Consumption (`kWh`)
    - **Best Model**: `Gradient Boosting` ($R^2 > 0.999$)
    - **Mean Absolute Error**: ~2.75 kWh
    - **Deploy Tier**: Streamlit Cloud / Docker Ready
    """)
    st.markdown("---")
    
    if st.button("🔄 Retrain Models & Refresh Artifacts"):
        with st.spinner("Retraining candidate regression models..."):
            train_and_evaluate_pipeline(include_lags=True, save_artifacts=True)
            train_and_evaluate_pipeline(include_lags=False, save_artifacts=True)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Artifacts retrained and persisted successfully!")
            st.rerun()

# --- App Header ---
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; color: #FFFFFF; font-size: 2.2rem; font-weight: 800;">⚡ Electricity Consumption Prediction System</h1>
            <p style="margin: 6px 0 0 0; color: #94A3B8; font-size: 1.05rem;">
                End-to-end Machine Learning Pipeline, Exploratory Data Analytics, Model Benchmarking & Live Serving
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Navigation Tabs ---
tab_home, tab_eda, tab_models, tab_single, tab_batch, tab_docs = st.tabs([
    "🏠 Overview",
    "📊 Data Explorer (EDA)",
    "⚙️ Model Benchmarks",
    "⚡ Single Prediction",
    "📁 Batch Predictions",
    "📖 IEEE Documentation"
])

# ==============================================================================
# TAB 1: OVERVIEW / HOME
# ==============================================================================
with tab_home:
    st.markdown("### 🌟 Project Perspective & Executive Summary")
    st.markdown("""
    The **Electricity Consumption Prediction System** delivers high-accuracy regression modeling to forecast hourly and daily electricity demand (kWh) across **Residential**, **Commercial**, and **Industrial** buildings.
    Built in strict compliance with the **IEEE 830 Software Requirements Specification**, this application provides full pipeline transparency, interactive dataset exploration, model auditing, and live multi-record inferencing.
    """)
    
    # KPI Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">12,000</div>
            <div class="metric-lbl">Total Dataset Records</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">27</div>
            <div class="metric-lbl">Engineered Features</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">0.9991</div>
            <div class="metric-lbl">Production R² Score</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">2.68%</div>
            <div class="metric-lbl">Test MAPE Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("#### 🔄 **System Architecture & Data Flow**")
        st.markdown("""
        ```mermaid
        graph LR
            A[Historical CSV Dataset] --> B[Data Validation & Preprocessing]
            B --> C[70/15/15 Stratified Split]
            C --> D[Candidate Regression Models]
            D --> E[Evaluation: MAE, RMSE, MAPE, R²]
            E --> F[Serialized Artifact: best_model.joblib]
            F --> G[Interactive Streamlit Serving UI]
            G --> H[Single & Batch Inferences]
        ```
        """)
        
        st.markdown("#### 💡 **Core Capabilities**")
        st.markdown("""
        - **Comprehensive EDA**: Multivariate correlation heatmaps, weather vs load regressions, and peak-hour profiles.
        - **Algorithmic Benchmarking**: Comparison across Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, and LightGBM.
        - **Lag Feature Transparency**: Distinct evaluations with and without near-leakage lag variables.
        - **Real-Time Predictions**: Instant single-record predictions with tariff cost calculations and building-type benchmark comparisons.
        - **Batch Processing**: Scalable CSV upload with instant validation, bulk inference, and exportable reports.
        """)
        
    with col_right:
        st.markdown("#### 🏢 **Dataset Building Distribution**")
        btype_counts = df_raw["building_type"].value_counts().reset_index()
        btype_counts.columns = ["Building Type", "Count"]
        fig_pie = px.pie(
            btype_counts,
            values="Count",
            names="Building Type",
            color="Building Type",
            color_discrete_map={"Residential": "#3B82F6", "Commercial": "#10B981", "Industrial": "#F59E0B"},
            hole=0.45
        )
        fig_pie.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC")
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ==============================================================================
# TAB 2: DATA EXPLORER (EDA)
# ==============================================================================
with tab_eda:
    st.markdown("### 📊 Exploratory Data Analysis & Feature Analytics")
    st.markdown("Interactively filter the historical electricity consumption dataset to discover demand patterns, weather correlations, and occupancy trends.")
    
    # Interactive Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sel_btype = st.multiselect(
            "Filter by Building Type:",
            options=CATEGORICAL_CHOICES["building_type"],
            default=CATEGORICAL_CHOICES["building_type"]
        )
    with col_f2:
        sel_season = st.multiselect(
            "Filter by Season:",
            options=CATEGORICAL_CHOICES["season"],
            default=CATEGORICAL_CHOICES["season"]
        )
    with col_f3:
        hour_range = st.slider("Filter by Hour of Day:", min_value=0, max_value=23, value=(0, 23))
        
    filtered_df = df_raw[
        (df_raw["building_type"].isin(sel_btype)) &
        (df_raw["season"].isin(sel_season)) &
        (df_raw["hour"] >= hour_range[0]) &
        (df_raw["hour"] <= hour_range[1])
    ]
    
    st.markdown(f"**Showing `{len(filtered_df):,}` filtered records** (out of `{len(df_raw):,}` total)")
    
    # EDA Chart Grid
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("##### 📈 Electricity Consumption Distribution (kWh)")
        fig_hist = px.histogram(
            filtered_df,
            x="energy_consumption_kWh",
            color="building_type",
            nbins=50,
            marginal="box",
            color_discrete_map={"Residential": "#3B82F6", "Commercial": "#10B981", "Industrial": "#F59E0B"},
            labels={"energy_consumption_kWh": "Electricity Consumption (kWh)"}
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with c2:
        st.markdown("##### 🌡️ Temperature vs. Electricity Consumption")
        fig_scatter = px.scatter(
            filtered_df.sample(min(1500, len(filtered_df)), random_state=42),
            x="temperature_C",
            y="energy_consumption_kWh",
            color="building_type",
            trendline="ols",
            color_discrete_map={"Residential": "#3B82F6", "Commercial": "#10B981", "Industrial": "#F59E0B"},
            labels={"temperature_C": "Ambient Temperature (°C)", "energy_consumption_kWh": "Consumption (kWh)"}
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("##### 🕒 Hourly Consumption Profiles")
        hourly_avg = filtered_df.groupby(["hour", "building_type"])["energy_consumption_kWh"].mean().reset_index()
        fig_hourly = px.line(
            hourly_avg,
            x="hour",
            y="energy_consumption_kWh",
            color="building_type",
            markers=True,
            color_discrete_map={"Residential": "#3B82F6", "Commercial": "#10B981", "Industrial": "#F59E0B"},
            labels={"hour": "Hour of Day (0-23)", "energy_consumption_kWh": "Mean Consumption (kWh)"}
        )
        fig_hourly.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
        
    with c4:
        st.markdown("##### 📐 Square Footage vs. Consumption")
        fig_sqft = px.scatter(
            filtered_df.sample(min(1500, len(filtered_df)), random_state=42),
            x="square_footage",
            y="energy_consumption_kWh",
            color="building_type",
            color_discrete_map={"Residential": "#3B82F6", "Commercial": "#10B981", "Industrial": "#F59E0B"},
            labels={"square_footage": "Square Footage (sq. ft.)", "energy_consumption_kWh": "Consumption (kWh)"}
        )
        fig_sqft.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_sqft, use_container_width=True)
        
    # Correlation Heatmap
    st.markdown("##### 🔥 Numeric Feature Correlation Heatmap")
    num_cols_corr = [
        "temperature_C", "humidity_percent", "square_footage", "num_occupants",
        "num_appliances", "ac_usage_hours", "heating_usage_hours", "lighting_hours",
        "previous_day_consumption_kWh", "avg_weekly_consumption_kWh", "energy_consumption_kWh"
    ]
    corr_matrix = filtered_df[num_cols_corr].corr().round(2)
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(color="Correlation")
    )
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC"),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ==============================================================================
# TAB 3: MODEL BENCHMARKS & TRAINING
# ==============================================================================
with tab_models:
    st.markdown("### ⚙️ Regression Model Benchmarking & Acceptance Criteria")
    st.markdown("""
    In accordance with **SRS Section 9**, candidate models were trained on a 70% split, tuned on 15% validation data, and rigorously scored on a 15% held-out test split.
    """)
    
    # Toggle for Lag Features
    col_t1, col_t2 = st.columns([2, 3])
    with col_t1:
        use_lag_toggle = st.radio(
            "Feature Engineering Mode:",
            options=["Include Lag Features (Standard)", "Exclude Lag Features (No Autocorrelation)"],
            index=0
        )
    include_lags = (use_lag_toggle == "Include Lag Features (Standard)")
    
    metrics_data = get_cached_metrics(with_lags=include_lags)
    
    # Acceptance Criteria Callout
    st.info(f"**IEEE 830 Acceptance Criteria**: Production Target requires $R^2 \\ge 0.85$ and $\\text{{MAPE}} \\le 15.0\\%$. Best model selected: **{metrics_data['best_model_name']}**.")
    
    # Metrics Table
    records = []
    for m_name, m_info in metrics_data["models"].items():
        records.append({
            "Model Name": m_name,
            "Train R²": f"{m_info['train']['r2']:.4f}",
            "Val R²": f"{m_info['val']['r2']:.4f}",
            "Test R²": f"{m_info['test']['r2']:.4f}",
            "Test MAE (kWh)": f"{m_info['test']['mae']:.2f}",
            "Test RMSE (kWh)": f"{m_info['test']['rmse']:.2f}",
            "Test MAPE (%)": f"{m_info['test']['mape']:.2f}%",
            "Status": "✅ PASS" if m_info["meets_acceptance"] else "❌ FAIL"
        })
    df_metrics = pd.DataFrame(records)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    
    # Comparison Charts
    col_mc1, col_mc2 = st.columns(2)
    with col_mc1:
        st.markdown("##### 📊 Test R² Score Comparison")
        fig_r2 = px.bar(
            df_metrics,
            x="Model Name",
            y=[float(x) for x in df_metrics["Test R²"]],
            color="Model Name",
            text=[f"{float(x):.4f}" for x in df_metrics["Test R²"]],
            labels={"y": "Test R² Score", "x": "Model"}
        )
        fig_r2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_r2, use_container_width=True)
        
    with col_mc2:
        st.markdown("##### 📉 Test MAE Error (Lower is Better)")
        fig_mae = px.bar(
            df_metrics,
            x="Model Name",
            y=[float(x) for x in df_metrics["Test MAE (kWh)"]],
            color="Model Name",
            text=[f"{float(x):.2f}" for x in df_metrics["Test MAE (kWh)"]],
            labels={"y": "Mean Absolute Error (kWh)", "x": "Model"}
        )
        fig_mae.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_mae, use_container_width=True)
        
    # Feature Importance of Best Model
    st.markdown(f"##### 🎯 Feature Importance Ranking ({metrics_data['best_model_name']})")
    best_feat_imp = metrics_data["models"][metrics_data["best_model_name"]]["feature_importances"]
    if best_feat_imp:
        df_feat = pd.DataFrame(best_feat_imp)
        fig_imp = px.bar(
            df_feat,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Blues",
            labels={"importance": "Relative Importance", "feature": "Engineered Feature"}
        )
        fig_imp.update_layout(
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        
    # Lag Feature Impact Analysis
    st.markdown("##### 🔍 Lag Features Impact Study (SRS Section 9.3)")
    lag_impact = get_cached_lag_impact()
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 12px; padding: 18px;">
            <h4 style="color: #38BDF8; margin-top:0;">⚡ With Lag Features</h4>
            <p><b>Best Algorithm:</b> {lag_impact['with_lags']['best_model']}</p>
            <p><b>Test R²:</b> {lag_impact['with_lags']['metrics']['r2']:.4f}</p>
            <p><b>Test MAE:</b> {lag_impact['with_lags']['metrics']['mae']:.2f} kWh</p>
            <p><b>Test MAPE:</b> {lag_impact['with_lags']['metrics']['mape']:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    with col_l2:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 18px;">
            <h4 style="color: #34D399; margin-top:0;">🏛️ Without Lag Features</h4>
            <p><b>Best Algorithm:</b> {lag_impact['without_lags']['best_model']}</p>
            <p><b>Test R²:</b> {lag_impact['without_lags']['metrics']['r2']:.4f}</p>
            <p><b>Test MAE:</b> {lag_impact['without_lags']['metrics']['mae']:.2f} kWh</p>
            <p><b>Test MAPE:</b> {lag_impact['without_lags']['metrics']['mape']:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 4: SINGLE RECORD PREDICTION
# ==============================================================================
with tab_single:
    st.markdown("### ⚡ Real-Time Single Record Prediction")
    st.markdown("Specify environmental, structural, and operational parameters to forecast instantaneous electricity demand (kWh) and estimated utility cost.")
    
    with st.form("single_prediction_form"):
        st.markdown("#### 1. 🏢 Building Characteristics")
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        with col_b1:
            in_btype = st.selectbox("Building Type", CATEGORICAL_CHOICES["building_type"], index=0)
            in_sqft = st.number_input("Square Footage (sq. ft.)", min_value=100.0, max_value=60000.0, value=1850.0, step=50.0)
        with col_b2:
            in_age = st.slider("Building Age (years)", min_value=0, max_value=60, value=12)
            in_occupants = st.number_input("Number of Occupants", min_value=1, max_value=150, value=4)
        with col_b3:
            in_appliances = st.number_input("Major Appliances Count", min_value=1, max_value=500, value=14)
            in_insulation = st.select_slider("Insulation Quality (1=Poor, 5=Excellent)", options=[1, 2, 3, 4, 5], value=3)
        with col_b4:
            in_energy_star = st.select_slider("Energy Star Rating (1-5)", options=[1, 2, 3, 4, 5], value=3)
            in_solar = st.selectbox("Rooftop Solar Installed?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
            
        st.markdown("#### 2. ⛅ Temporal & Weather Conditions")
        col_w1, col_w2, col_w3, col_w4 = st.columns(4)
        with col_w1:
            in_season = st.selectbox("Season", CATEGORICAL_CHOICES["season"], index=2)
            in_month = st.slider("Month (1-12)", min_value=1, max_value=12, value=7)
        with col_w2:
            in_hour = st.slider("Hour of Day (0-23)", min_value=0, max_value=23, value=14)
            in_dow = st.selectbox("Day of Week", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x], index=2)
        with col_w3:
            in_is_weekend = 1 if in_dow >= 5 else 0
            in_holiday = st.selectbox("Public Holiday?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
            in_temp = st.number_input("Ambient Temperature (°C)", min_value=-10.0, max_value=50.0, value=31.5, step=0.5)
        with col_w4:
            in_humidity = st.slider("Humidity (%)", min_value=10.0, max_value=100.0, value=60.0)
            in_wind = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=8.5, step=0.5)
            in_precip = st.number_input("Precipitation (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)

        st.markdown("#### 3. 🔌 Climate & Appliance Usage & Tariff")
        col_u1, col_u2, col_u3, col_u4 = st.columns(4)
        with col_u1:
            in_ac_hours = st.slider("AC Usage (hours)", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
        with col_u2:
            in_heat_hours = st.slider("Heating Usage (hours)", min_value=0.0, max_value=24.0, value=0.0, step=0.5)
        with col_u3:
            in_light_hours = st.slider("Lighting (hours)", min_value=0.0, max_value=24.0, value=6.5, step=0.5)
        with col_u4:
            in_price = st.number_input("Tariff Rate ($ / kWh)", min_value=1.0, max_value=25.0, value=8.25, step=0.25)
            
        st.markdown("#### 4. 📈 Historical Baseline / Lag Parameters")
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            in_prev_day = st.number_input("Previous Day Consumption (kWh)", min_value=0.0, max_value=1000.0, value=48.5, step=1.0)
        with col_l2:
            in_avg_weekly = st.number_input("Average Weekly Consumption (kWh)", min_value=0.0, max_value=1000.0, value=50.2, step=1.0)
            
        submit_btn = st.form_submit_button("⚡ Generate Electricity Prediction", use_container_width=True)
        
    if submit_btn:
        input_payload = {
            "hour": in_hour,
            "day_of_week": in_dow,
            "is_weekend": in_is_weekend,
            "month": in_month,
            "season": in_season,
            "is_holiday": in_holiday,
            "temperature_C": in_temp,
            "humidity_percent": in_humidity,
            "wind_speed_kmph": in_wind,
            "precipitation_mm": in_precip,
            "building_type": in_btype,
            "square_footage": in_sqft,
            "building_age_years": in_age,
            "num_occupants": in_occupants,
            "num_appliances": in_appliances,
            "insulation_quality": in_insulation,
            "energy_star_rating": in_energy_star,
            "solar_panel_installed": in_solar,
            "electricity_price_per_kWh": in_price,
            "ac_usage_hours": in_ac_hours,
            "heating_usage_hours": in_heat_hours,
            "lighting_hours": in_light_hours,
            "previous_day_consumption_kWh": in_prev_day,
            "avg_weekly_consumption_kWh": in_avg_weekly
        }
        
        try:
            pred_res = predictor.predict_single(input_payload)
            
            st.markdown(f"""
            <div class="pred-highlight-box">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <span style="color: #94A3B8; font-size: 0.95rem; font-weight: 600; text-transform: uppercase;">Predicted Electricity Consumption</span>
                        <div style="font-size: 3.2rem; font-weight: 900; color: #38BDF8; margin: 4px 0;">
                            {pred_res['predicted_kWh']:.2f} <span style="font-size: 1.6rem; color: #94A3B8;">kWh</span>
                        </div>
                        <div style="color: #F8FAFC; font-size: 1.05rem;">
                            Estimated Cost: <b>${pred_res['estimated_cost']:.2f}</b> (@ ${pred_res['tariff_rate']}/kWh)
                        </div>
                    </div>
                    <div style="text-align: right; background: rgba(15, 23, 42, 0.6); padding: 16px 24px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);">
                        <div style="color: #94A3B8; font-size: 0.85rem;">95% Confidence Interval</div>
                        <div style="color: #34D399; font-size: 1.2rem; font-weight: 700;">
                            {pred_res['confidence_interval_kWh'][0]:.2f} – {pred_res['confidence_interval_kWh'][1]:.2f} kWh
                        </div>
                        <div style="margin-top: 8px; color: #94A3B8; font-size: 0.85rem;">
                            Vs {pred_res['building_type']} Avg ({pred_res['benchmark_mean_kWh']:.1f} kWh):
                            <b style="color: {'#F87171' if pred_res['delta_from_benchmark_pct'] > 0 else '#34D399'};">
                                {pred_res['delta_from_benchmark_pct']:+.1f}%
                            </b>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error executing prediction: {str(e)}")

# ==============================================================================
# TAB 5: BATCH PREDICTION
# ==============================================================================
with tab_batch:
    st.markdown("### 📁 Batch Prediction & Bulk Processing")
    st.markdown("Upload a CSV file containing multiple building records to compute predictions at scale and export results.")
    
    # Download sample template
    sample_df = df_raw.drop(columns=[TARGET_COLUMN]).head(10)
    csv_buffer = io.StringIO()
    sample_df.to_csv(csv_buffer, index=False)
    
    col_btn, _ = st.columns([2, 4])
    with col_btn:
        st.download_button(
            label="📥 Download Sample Batch CSV Template",
            data=csv_buffer.getvalue(),
            file_name="electricity_batch_template.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    uploaded_file = st.file_uploader("Upload Batch CSV Dataset", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded file with `{len(batch_df):,}` rows and `{len(batch_df.columns)}` columns.")
            
            st.markdown("##### 🔍 Uploaded Data Preview")
            st.dataframe(batch_df.head(5), use_container_width=True)
            
            if st.button("🚀 Process Batch Predictions", type="primary", use_container_width=True):
                with st.spinner("Computing inferences across batch records..."):
                    results_df, batch_summary = predictor.predict_batch(batch_df)
                    
                    st.markdown("---")
                    st.markdown("#### 📊 Batch Processing Results Summary")
                    
                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    with col_s1:
                        st.metric("Total Records Processed", f"{batch_summary['total_records']:,}")
                    with col_s2:
                        st.metric("Average Consumption", f"{batch_summary['mean_predicted_kWh']:.2f} kWh")
                    with col_s3:
                        st.metric("Total Batch Energy", f"{batch_summary['total_predicted_kWh']:,.1f} kWh")
                    with col_s4:
                        st.metric("Peak Demand", f"{batch_summary['max_predicted_kWh']:.2f} kWh")
                        
                    # Batch prediction distribution chart
                    fig_batch = px.histogram(
                        results_df,
                        x="predicted_energy_consumption_kWh",
                        color="building_type" if "building_type" in results_df.columns else None,
                        title="Predicted Consumption Distribution across Batch Records",
                        labels={"predicted_energy_consumption_kWh": "Predicted Energy Consumption (kWh)"}
                    )
                    fig_batch.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#F8FAFC")
                    )
                    st.plotly_chart(fig_batch, use_container_width=True)
                    
                    st.markdown("##### 📋 Augmented Predictions Table")
                    st.dataframe(results_df.head(20), use_container_width=True)
                    
                    # Download predicted results
                    out_csv = io.StringIO()
                    results_df.to_csv(out_csv, index=False)
                    st.download_button(
                        label="💾 Download Predictions as CSV",
                        data=out_csv.getvalue(),
                        file_name="electricity_predictions_output.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    
        except Exception as e:
            st.error(f"Error processing batch upload: {str(e)}")

# ==============================================================================
# TAB 6: DOCUMENTATION & SRS REFERENCE
# ==============================================================================
with tab_docs:
    st.markdown("### 📖 IEEE 830 System Specifications & Data Dictionary")
    
    st.markdown("""
    #### 1. System Scope & Objective
    The **Electricity Consumption Prediction System** provides automated regression estimation for residential, commercial, and industrial buildings to facilitate grid management, dynamic peak shaving, and facility energy audits.
    
    #### 2. Functional Requirements Coverage
    - **FR-1 Ingestion**: Validated 27-column schema loading with automated data quality checks.
    - **FR-2 Preprocessing**: Scikit-Learn `ColumnTransformer` with median imputation, one-hot encoding, and standard scaling.
    - **FR-3 Feature Selection**: Stratified 70/15/15 train/val/test splits with configurable lag-feature inclusion.
    - **FR-4 & FR-5 Model Benchmarking**: Evaluation of Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, and LightGBM across MAE, RMSE, MAPE, and $R^2$.
    - **FR-6 & FR-7 Serving Engine**: Interactive real-time single-record inference and scalable batch CSV processing.
    - **FR-8 & FR-9 Visualization**: Plotly EDA dashboards, model metric comparisons, and feature importance analysis.
    - **FR-10 Model Persistence**: Joblib serialized pipelines with versioned artifact metadata.
    - **FR-12 Deployment**: Containerized with Docker and ready for Streamlit Community Cloud.
    """)
    
    st.markdown("#### 3. Data Dictionary (27 Schema Attributes)")
    data_dict_records = [
        {"Column": "record_id", "Type": "Integer", "Range / Unit": "1 – 12,000", "Description": "Unique row identifier"},
        {"Column": "date", "Type": "Date", "Range / Unit": "2023-01-01 to 2023-12-31", "Description": "Calendar date of reading"},
        {"Column": "hour", "Type": "Integer", "Range / Unit": "0 – 23", "Description": "Hour of day"},
        {"Column": "day_of_week", "Type": "Integer", "Range / Unit": "0 – 6", "Description": "Day of week (Monday=0)"},
        {"Column": "is_weekend", "Type": "Binary", "Range / Unit": "0 or 1", "Description": "1 if Saturday/Sunday"},
        {"Column": "month", "Type": "Integer", "Range / Unit": "1 – 12", "Description": "Calendar month"},
        {"Column": "season", "Type": "Categorical", "Range / Unit": "Winter/Spring/Summer/Autumn", "Description": "Meteorological season"},
        {"Column": "is_holiday", "Type": "Binary", "Range / Unit": "0 or 1", "Description": "1 if public holiday"},
        {"Column": "temperature_C", "Type": "Float", "Range / Unit": "-20.0 to 60.0 °C", "Description": "Ambient temperature"},
        {"Column": "humidity_percent", "Type": "Float", "Range / Unit": "10 – 100 %", "Description": "Relative humidity"},
        {"Column": "wind_speed_kmph", "Type": "Float", "Range / Unit": "km/h", "Description": "Wind speed"},
        {"Column": "precipitation_mm", "Type": "Float", "Range / Unit": "mm", "Description": "Rainfall"},
        {"Column": "building_type", "Type": "Categorical", "Range / Unit": "Residential/Commercial/Industrial", "Description": "Property category"},
        {"Column": "square_footage", "Type": "Float", "Range / Unit": "sq. ft.", "Description": "Built-up area"},
        {"Column": "building_age_years", "Type": "Integer", "Range / Unit": "0 – 60 yrs", "Description": "Age of building"},
        {"Column": "num_occupants", "Type": "Integer", "Range / Unit": "≥ 1", "Description": "Occupancy count"},
        {"Column": "num_appliances", "Type": "Integer", "Range / Unit": "≥ 2", "Description": "Major appliances count"},
        {"Column": "insulation_quality", "Type": "Integer", "Range / Unit": "1 – 5", "Description": "1=poor, 5=excellent"},
        {"Column": "energy_star_rating", "Type": "Integer", "Range / Unit": "1 – 5", "Description": "Energy-efficiency rating"},
        {"Column": "solar_panel_installed", "Type": "Binary", "Range / Unit": "0 or 1", "Description": "Rooftop solar presence"},
        {"Column": "electricity_price_per_kWh", "Type": "Float", "Range / Unit": "$/kWh", "Description": "Tariff rate"},
        {"Column": "ac_usage_hours", "Type": "Float", "Range / Unit": "0 – 24 hrs", "Description": "AC runtime"},
        {"Column": "heating_usage_hours", "Type": "Float", "Range / Unit": "0 – 24 hrs", "Description": "Heating runtime"},
        {"Column": "lighting_hours", "Type": "Float", "Range / Unit": "0 – 24 hrs", "Description": "Lighting runtime"},
        {"Column": "previous_day_consumption_kWh", "Type": "Float", "Range / Unit": "kWh", "Description": "Prior day consumption lag"},
        {"Column": "avg_weekly_consumption_kWh", "Type": "Float", "Range / Unit": "kWh", "Description": "Rolling weekly average lag"},
        {"Column": "energy_consumption_kWh", "Type": "Float", "Range / Unit": "kWh", "Description": "TARGET — total electricity consumed"}
    ]
    st.dataframe(pd.DataFrame(data_dict_records), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("© 2026 Electricity Consumption Prediction System. IEEE 830 Compliant Release v1.0.")
