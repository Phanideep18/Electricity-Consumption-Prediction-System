"""
Configuration and Schema Definitions for Electricity Consumption Prediction System.
Compliant with IEEE 830 SRS Section 3.3 & Section 4 (FR-11).
"""

import os
from pathlib import Path
from typing import Dict, Any, List

BASE_DIR = Path(__file__).resolve().parent.parent

# Feature Schema Definitions per SRS Section 3.3
SCHEMA_COLUMNS = [
    "record_id",
    "date",
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "season",
    "is_holiday",
    "temperature_C",
    "humidity_percent",
    "wind_speed_kmph",
    "precipitation_mm",
    "building_type",
    "square_footage",
    "building_age_years",
    "num_occupants",
    "num_appliances",
    "insulation_quality",
    "energy_star_rating",
    "solar_panel_installed",
    "electricity_price_per_kWh",
    "ac_usage_hours",
    "heating_usage_hours",
    "lighting_hours",
    "previous_day_consumption_kWh",
    "avg_weekly_consumption_kWh",
    "energy_consumption_kWh"
]

CATEGORICAL_COLUMNS = ["building_type", "season"]

CATEGORICAL_CHOICES = {
    "building_type": ["Residential", "Commercial", "Industrial"],
    "season": ["Winter", "Spring", "Summer", "Autumn"]
}

NUMERIC_RANGES = {
    "hour": (0, 23),
    "day_of_week": (0, 6),
    "is_weekend": (0, 1),
    "month": (1, 12),
    "is_holiday": (0, 1),
    "temperature_C": (-20.0, 60.0),
    "humidity_percent": (0.0, 100.0),
    "wind_speed_kmph": (0.0, 150.0),
    "precipitation_mm": (0.0, 300.0),
    "square_footage": (50.0, 200000.0),
    "building_age_years": (0, 150),
    "num_occupants": (1, 1000),
    "num_appliances": (1, 2000),
    "insulation_quality": (1, 5),
    "energy_star_rating": (1, 5),
    "solar_panel_installed": (0, 1),
    "electricity_price_per_kWh": (0.01, 100.0),
    "ac_usage_hours": (0.0, 24.0),
    "heating_usage_hours": (0.0, 24.0),
    "lighting_hours": (0.0, 24.0),
    "previous_day_consumption_kWh": (0.0, 5000.0),
    "avg_weekly_consumption_kWh": (0.0, 5000.0),
    "energy_consumption_kWh": (0.0, 10000.0)
}

TARGET_COLUMN = "energy_consumption_kWh"
DROP_COLUMNS = ["record_id", "date"]
LAG_FEATURES = ["previous_day_consumption_kWh", "avg_weekly_consumption_kWh"]

DEFAULT_DATASET_PATH = BASE_DIR / "electricity_consumption_dataset.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
