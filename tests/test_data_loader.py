"""Unit tests for src.data_loader."""

from pathlib import Path
import pandas as pd
import pytest

from src.data_loader import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PASS_GRADE_THRESHOLD,
    TARGET_REGRESSION,
    check_duplicates,
    get_dataset_summary,
    get_missing_values_summary,
    load_data,
    split_features_and_targets,
    validate_columns,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "student-mat.csv"


def test_load_data_success():
    """Verify that student-mat.csv loads correctly with semicolon delimiter."""
    df = load_data(DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == 395
    assert len(df.columns) == 33
    assert TARGET_REGRESSION in df.columns


def test_load_data_file_not_found():
    """Verify FileNotFoundError is raised when file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_data("data/non_existent_file.csv")


def test_validate_columns_missing():
    """Verify ValueError is raised if any required predictor or target column is missing."""
    df_missing = pd.DataFrame({"studytime": [1, 2], "failures": [0, 1]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_columns(df_missing)


def test_check_duplicates():
    """Verify duplicate detection utility."""
    df = pd.DataFrame({
        "a": [1, 1, 2],
        "b": [3, 3, 4],
    })
    count, dupes = check_duplicates(df)
    assert count == 1
    assert len(dupes) == 1


def test_get_missing_values_summary():
    """Verify missing values computation."""
    df = pd.DataFrame({
        "col1": [1, None, 3],
        "col2": ["a", "b", "c"],
    })
    summary = get_missing_values_summary(df)
    assert "Column" in summary.columns
    assert "Missing Count" in summary.columns
    assert summary.loc[summary["Column"] == "col1", "Missing Count"].iloc[0] == 1
    assert summary.loc[summary["Column"] == "col2", "Missing Count"].iloc[0] == 0


def test_split_features_and_targets_data_leakage():
    """CRITICAL: Verify data leakage prevention.

    G1, G2, and G3 must NOT be in model input features X.
    Target regression must be G3, and classification must be binary (G3 >= 10).
    """
    df = load_data(DATA_PATH)
    X, y_reg, y_class = split_features_and_targets(df)

    # Verify input feature dimensions and columns
    assert X.shape[1] == len(ALL_FEATURES)
    assert set(X.columns) == set(ALL_FEATURES)

    # Leakage check: G1 and G2 must be completely absent from input features
    assert "G1" not in X.columns
    assert "G2" not in X.columns
    assert "G3" not in X.columns

    # Verify targets
    assert len(y_reg) == len(df)
    assert len(y_class) == len(df)
    assert set(y_class.unique()).issubset({0, 1})

    # Check threshold logic
    for grade, status in zip(y_reg.iloc[:20], y_class.iloc[:20]):
        if grade >= PASS_GRADE_THRESHOLD:
            assert status == 1
        else:
            assert status == 0


def test_get_dataset_summary():
    """Verify dataset summary metadata."""
    df = load_data(DATA_PATH)
    summary = get_dataset_summary(df)
    assert summary["total_students"] == 395
    assert summary["duplicate_records"] == 0
    assert summary["pass_count"] + summary["fail_count"] == 395
    assert 0.0 <= summary["pass_percentage"] <= 100.0

