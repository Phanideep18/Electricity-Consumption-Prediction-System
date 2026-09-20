"""
Data Preprocessing and Feature Pipeline Module.
Compliant with IEEE 830 SRS Section 3.7 & Section 4 (FR-2, FR-3).
"""

from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.config import (
    CATEGORICAL_COLUMNS,
    TARGET_COLUMN,
    DROP_COLUMNS,
    LAG_FEATURES
)


def get_feature_lists(
    df: pd.DataFrame,
    include_lags: bool = True
) -> Tuple[List[str], List[str], List[str]]:
    """
    Separate features into numeric and categorical lists, respecting lag feature toggle.
    """
    drop_cols = set(DROP_COLUMNS)
    if TARGET_COLUMN in df.columns:
        drop_cols.add(TARGET_COLUMN)
    
    if not include_lags:
        for lag_col in LAG_FEATURES:
            drop_cols.add(lag_col)
    
    feature_cols = [c for c in df.columns if c not in drop_cols]
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    
    return feature_cols, num_cols, cat_cols


def create_preprocessor(
    num_cols: List[str],
    cat_cols: List[str]
) -> ColumnTransformer:
    """
    Create a robust Scikit-Learn ColumnTransformer for scaling numeric features
    and one-hot encoding categorical features with missing value imputation.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ],
        remainder="drop"
    )
    
    return preprocessor


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    stratify_col: str = "building_type",
    include_lags: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split dataset into 70% Train, 15% Validation, and 15% Test sets,
    stratified by building_type (FR-2, SRS Section 3.7).
    """
    feature_cols, _, _ = get_feature_lists(df, include_lags=include_lags)
    
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()
    
    stratify = df[stratify_col] if stratify_col in df.columns else None
    
    # First split: Train+Val (85%) and Test (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        random_state=random_state,
        stratify=stratify
    )
    
    # Second split: Train (70/85) and Val (15/85)
    val_relative_size = val_ratio / (train_ratio + val_ratio)
    stratify_train_val = X_train_val[stratify_col] if stratify_col in X_train_val.columns else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=stratify_train_val
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
