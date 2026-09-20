"""
Data Ingestion and Validation Module.
Compliant with IEEE 830 SRS Section 3.4, 3.8 & Section 4 (FR-1).
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.config import (
    SCHEMA_COLUMNS,
    CATEGORICAL_COLUMNS,
    CATEGORICAL_CHOICES,
    NUMERIC_RANGES,
    TARGET_COLUMN,
    DEFAULT_DATASET_PATH
)


class DataValidationError(Exception):
    """Raised when dataset fails validation checks."""
    pass


def load_dataset(file_path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load dataset from CSV or Parquet file.
    Supports growing datasets and format variations (FR-1, 3.8).
    """
    if file_path is None:
        file_path = DEFAULT_DATASET_PATH
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")
    
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    
    return df


def validate_dataset(df: pd.DataFrame, is_inference: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate dataframe against schema, data quality, and integrity rules (FR-1, 3.4).
    Returns (is_valid, list_of_warning_messages).
    """
    warnings = []
    
    # 1. Check required columns
    expected_cols = [c for c in SCHEMA_COLUMNS if not (is_inference and c in [TARGET_COLUMN, "record_id"])]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")
    
    # 2. Check for duplicate record_ids (if present)
    if "record_id" in df.columns:
        dup_count = df["record_id"].duplicated().sum()
        if dup_count > 0:
            raise DataValidationError(f"Found {dup_count} duplicate 'record_id' entries in dataset.")
    
    # 3. Check for missing values
    null_counts = df[expected_cols].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        warnings.append(f"Missing values detected in: {cols_with_nulls.to_dict()}")
    
    # 4. Check Categoricals
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            valid_choices = CATEGORICAL_CHOICES[col]
            invalid_entries = df[~df[col].isin(valid_choices)][col].unique()
            if len(invalid_entries) > 0:
                raise DataValidationError(
                    f"Invalid categorical values in '{col}': {invalid_entries.tolist()}. "
                    f"Expected one of: {valid_choices}"
                )
    
    # 5. Check Target column if training
    if not is_inference and TARGET_COLUMN in df.columns:
        if (df[TARGET_COLUMN] <= 0).any():
            warnings.append("Target variable 'energy_consumption_kWh' contains zero or negative values.")
    
    # 6. Check Numeric Ranges
    for col, (min_val, max_val) in NUMERIC_RANGES.items():
        if col in df.columns and col in expected_cols:
            out_of_bounds = df[(df[col] < min_val) | (df[col] > max_val)]
            if not out_of_bounds.empty:
                warnings.append(
                    f"Column '{col}' has {len(out_of_bounds)} values outside expected range [{min_val}, {max_val}]."
                )
    
    return True, warnings


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Return high-level summary statistics of the dataset."""
    return {
        "num_records": len(df),
        "num_columns": len(df.columns),
        "building_type_counts": df["building_type"].value_counts().to_dict() if "building_type" in df.columns else {},
        "season_counts": df["season"].value_counts().to_dict() if "season" in df.columns else {},
        "mean_consumption_kWh": float(df[TARGET_COLUMN].mean()) if TARGET_COLUMN in df.columns else None,
        "std_consumption_kWh": float(df[TARGET_COLUMN].std()) if TARGET_COLUMN in df.columns else None,
        "min_consumption_kWh": float(df[TARGET_COLUMN].min()) if TARGET_COLUMN in df.columns else None,
        "max_consumption_kWh": float(df[TARGET_COLUMN].max()) if TARGET_COLUMN in df.columns else None,
    }
