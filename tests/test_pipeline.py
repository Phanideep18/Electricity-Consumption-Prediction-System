"""
Unit and Integration Tests for Electricity Consumption Prediction System.
Tests data loader, validation rules, preprocessor, model training, and predictor inference.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import DEFAULT_DATASET_PATH, MODELS_DIR, TARGET_COLUMN
from src.data_loader import load_dataset, validate_dataset, DataValidationError, get_dataset_summary
from src.preprocessor import get_feature_lists, create_preprocessor, split_data
from src.predictor import Predictor


class TestElectricityPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset(DEFAULT_DATASET_PATH)
        cls.predictor = Predictor()

    def test_01_dataset_loading_and_shape(self):
        """Verify dataset loads and meets volume requirement (>=10,000 rows)."""
        self.assertIsInstance(self.df, pd.DataFrame)
        self.assertGreaterEqual(len(self.df), 10000)
        self.assertEqual(len(self.df), 12000)

    def test_02_dataset_validation(self):
        """Verify dataset passes schema and categorical checks."""
        is_valid, warnings = validate_dataset(self.df)
        self.assertTrue(is_valid)

    def test_03_validation_catches_invalid_categories(self):
        """Verify validator raises error on invalid building_type."""
        corrupted_df = self.df.head(10).copy()
        corrupted_df.loc[0, "building_type"] = "InvalidType"
        with self.assertRaises(DataValidationError):
            validate_dataset(corrupted_df)

    def test_04_preprocessor_and_split(self):
        """Verify 70/15/15 stratified split and preprocessor transformation."""
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            self.df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        self.assertAlmostEqual(len(X_train), 8400, delta=5)
        self.assertAlmostEqual(len(X_val), 1800, delta=5)
        self.assertEqual(len(X_test), 1800)
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), 12000)

        _, num_cols, cat_cols = get_feature_lists(self.df, include_lags=True)
        preprocessor = create_preprocessor(num_cols, cat_cols)
        X_train_trans = preprocessor.fit_transform(X_train)
        
        self.assertEqual(X_train_trans.shape[0], len(X_train))
        self.assertFalse(np.isnan(X_train_trans).any())

    def test_05_single_prediction_inference(self):
        """Verify single record prediction returns realistic kWh, bounds, and cost."""
        sample_input = {
            "hour": 14,
            "day_of_week": 2,
            "is_weekend": 0,
            "month": 7,
            "season": "Summer",
            "is_holiday": 0,
            "temperature_C": 32.5,
            "humidity_percent": 65.0,
            "wind_speed_kmph": 12.0,
            "precipitation_mm": 0.0,
            "building_type": "Residential",
            "square_footage": 1850.0,
            "building_age_years": 12,
            "num_occupants": 4,
            "num_appliances": 15,
            "insulation_quality": 4,
            "energy_star_rating": 4,
            "solar_panel_installed": 1,
            "electricity_price_per_kWh": 8.50,
            "ac_usage_hours": 6.5,
            "heating_usage_hours": 0.0,
            "lighting_hours": 5.0,
            "previous_day_consumption_kWh": 45.2,
            "avg_weekly_consumption_kWh": 47.8
        }
        res = self.predictor.predict_single(sample_input)
        self.assertIn("predicted_kWh", res)
        self.assertGreater(res["predicted_kWh"], 0.0)
        self.assertIn("estimated_cost", res)
        self.assertIn("confidence_interval_kWh", res)
        self.assertEqual(res["building_type"], "Residential")

    def test_06_batch_prediction_inference(self):
        """Verify batch prediction processes DataFrame and generates output with summary."""
        test_batch = self.df.drop(columns=[TARGET_COLUMN]).head(25)
        output_df, summary = self.predictor.predict_batch(test_batch)
        
        self.assertEqual(len(output_df), 25)
        self.assertIn("predicted_energy_consumption_kWh", output_df.columns)
        self.assertIn("estimated_total_cost", output_df.columns)
        self.assertEqual(summary["total_records"], 25)
        self.assertGreater(summary["mean_predicted_kWh"], 0.0)


if __name__ == "__main__":
    unittest.main()
