"""Data preprocessing pipeline module for Student Performance Prediction System."""

from typing import List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def build_preprocessor(
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Build a Scikit-learn ColumnTransformer for preprocessing student data.

    Numerical pipeline:
        - Median imputation for missing numeric values.
        - Standard scaling for zero mean and unit variance.

    Categorical pipeline:
        - Most frequent imputation for missing categories.
        - One-hot encoding with unknown categories ignored.

    Args:
        numerical_cols: List of numerical column names. Defaults to NUMERICAL_FEATURES.
        categorical_cols: List of categorical column names. Defaults to CATEGORICAL_FEATURES.

    Returns:
        ColumnTransformer: An unfitted ColumnTransformer pipeline.
    """
    if numerical_cols is None:
        numerical_cols = list(NUMERICAL_FEATURES)
    if categorical_cols is None:
        categorical_cols = list(CATEGORICAL_FEATURES)

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor

