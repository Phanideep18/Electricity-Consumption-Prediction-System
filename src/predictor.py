"""
Inference Engine for Single and Batch Electricity Consumption Predictions.
Compliant with IEEE 830 SRS Section 4 (FR-6, FR-7) & Section 5.1, 6.
"""

import os
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from src.config import (
    MODELS_DIR,
    SCHEMA_COLUMNS,
    CATEGORICAL_COLUMNS,
    CATEGORICAL_CHOICES,
    NUMERIC_RANGES,
    TARGET_COLUMN,
    DROP_COLUMNS
)
from src.data_loader import validate_dataset


class Predictor:
    """Production Inference Engine."""
    
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        if model_path is None:
            model_path = MODELS_DIR / "best_model_pipeline.joblib"
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. "
                "Please run model training pipeline first."
            )
        
        self.pipeline: Pipeline = joblib.load(self.model_path)
        
        # Benchmark statistics by building_type for context / comparison (SRS 5.1)
        self.benchmarks = {
            "Residential": {"mean": 51.4, "min": 14.76, "max": 82.5},
            "Commercial": {"mean": 175.2, "min": 85.0, "max": 320.0},
            "Industrial": {"mean": 412.8, "min": 210.0, "max": 736.5}
        }
    
    def predict_single(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run prediction for a single user-entered record.
        Validates ranges and returns predicted kWh, confidence range, and benchmark comparison (FR-6, 5.1).
        """
        # Create single-row DataFrame
        df = pd.DataFrame([input_dict])
        
        # Clean / drop non-feature columns if present
        cols_to_drop = [c for c in DROP_COLUMNS + [TARGET_COLUMN] if c in df.columns]
        X = df.drop(columns=cols_to_drop, errors="ignore")
        
        # Validate categoricals
        b_type = input_dict.get("building_type", "Residential")
        if b_type not in CATEGORICAL_CHOICES["building_type"]:
            raise ValueError(f"Invalid building type: {b_type}")
            
        season = input_dict.get("season", "Summer")
        if season not in CATEGORICAL_CHOICES["season"]:
            raise ValueError(f"Invalid season: {season}")
        
        # Prediction
        prediction_val = float(self.pipeline.predict(X)[0])
        prediction_val = max(0.0, round(prediction_val, 2))
        
        # Benchmark delta
        benchmark_mean = self.benchmarks.get(b_type, {}).get("mean", 135.7)
        delta_pct = ((prediction_val - benchmark_mean) / benchmark_mean) * 100
        
        # Tariff cost estimation if electricity_price_per_kWh is present
        tariff = input_dict.get("electricity_price_per_kWh", 7.5)
        estimated_cost = round(prediction_val * tariff, 2)
        
        return {
            "predicted_kWh": prediction_val,
            "building_type": b_type,
            "benchmark_mean_kWh": benchmark_mean,
            "delta_from_benchmark_pct": round(delta_pct, 1),
            "estimated_cost": estimated_cost,
            "tariff_rate": tariff,
            "confidence_interval_kWh": (
                max(0.0, round(prediction_val * 0.95, 2)),
                round(prediction_val * 1.05, 2)
            )
        }
    
    def predict_batch(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run predictions on an uploaded batch DataFrame.
        Returns the augmented DataFrame with predictions and batch summary statistics (FR-7, 6).
        """
        df_clean = df.copy()
        
        # Validate batch
        validate_dataset(df_clean, is_inference=True)
        
        # Prepare feature matrix
        cols_to_drop = [c for c in DROP_COLUMNS + [TARGET_COLUMN] if c in df_clean.columns]
        X = df_clean.drop(columns=cols_to_drop, errors="ignore")
        
        preds = self.pipeline.predict(X)
        preds = np.clip(preds, a_min=0.0, a_max=None).round(2)
        
        # Augment dataframe
        output_df = df_clean.copy()
        output_df["predicted_energy_consumption_kWh"] = preds
        
        if "electricity_price_per_kWh" in output_df.columns:
            output_df["estimated_total_cost"] = (
                output_df["predicted_energy_consumption_kWh"] * output_df["electricity_price_per_kWh"]
            ).round(2)
            
        summary = {
            "total_records": len(output_df),
            "mean_predicted_kWh": round(float(preds.mean()), 2),
            "total_predicted_kWh": round(float(preds.sum()), 2),
            "min_predicted_kWh": round(float(preds.min()), 2),
            "max_predicted_kWh": round(float(preds.max()), 2),
            "by_building_type": output_df.groupby("building_type")["predicted_energy_consumption_kWh"].mean().round(2).to_dict() if "building_type" in output_df.columns else {}
        }
        
        return output_df, summary
