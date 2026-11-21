"""Unit tests for src.preprocessing."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.data_loader import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.preprocessing import build_preprocessor


def test_build_preprocessor_structure():
    """Verify ColumnTransformer transformers and parameters."""
    preprocessor = build_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)

    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names


def test_preprocessor_handles_missing_values():
    """Verify that SimpleImputer fills missing numerical and categorical values."""
    df_with_nan = pd.DataFrame({
        "studytime": [2.0, np.nan, 4.0],
        "failures": [0.0, np.nan, 2.0],
        "absences": [4.0, 8.0, np.nan],
        "traveltime": [1.0, np.nan, 2.0],
        "schoolsup": ["yes", None, "no"],
        "famsup": ["no", "yes", np.nan],
        "paid": ["yes", "no", None],
        "activities": ["no", "yes", "no"],
        "higher": ["yes", "yes", "no"],
        "internet": ["yes", "no", "yes"],
    })

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(df_with_nan)

    # Output must be dense float array without any NaNs
    assert not np.isnan(transformed).any()
    assert transformed.shape[0] == 3


def test_preprocessor_handles_unseen_categories():
    """Verify OneHotEncoder handles unknown categories gracefully via handle_unknown='ignore'."""
    df_train = pd.DataFrame({
        "studytime": [1, 2],
        "failures": [0, 0],
        "absences": [2, 4],
        "traveltime": [1, 1],
        "schoolsup": ["yes", "no"],
        "famsup": ["yes", "no"],
        "paid": ["yes", "no"],
        "activities": ["yes", "no"],
        "higher": ["yes", "no"],
        "internet": ["yes", "no"],
    })

    df_test_unknown = pd.DataFrame({
        "studytime": [3],
        "failures": [1],
        "absences": [5],
        "traveltime": [2],
        "schoolsup": ["unknown_value"],
        "famsup": ["other"],
        "paid": ["maybe"],
        "activities": ["yes"],
        "higher": ["yes"],
        "internet": ["no"],
    })

    preprocessor = build_preprocessor()
    preprocessor.fit(df_train)
    transformed = preprocessor.transform(df_test_unknown)

    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()

