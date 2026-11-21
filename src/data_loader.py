"""Data loading and validation module for Student Performance Prediction System."""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd

# Model Feature Definitions
NUMERICAL_FEATURES: List[str] = [
    "studytime",
    "failures",
    "absences",
    "traveltime",
]

CATEGORICAL_FEATURES: List[str] = [
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "higher",
    "internet",
]

ALL_FEATURES: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_REGRESSION: str = "G3"
PASS_GRADE_THRESHOLD: int = 10


def load_data(filepath: str | Path = "data/student-mat.csv") -> pd.DataFrame:
    """Load the student performance dataset from a semicolon-delimited CSV file.

    Args:
        filepath: Path to the CSV dataset file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If loading fails or data is empty.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")

    try:
        df = pd.read_csv(path, sep=";")
    except Exception as e:
        raise ValueError(f"Failed to read dataset from {path}: {str(e)}") from e

    if df.empty:
        raise ValueError(f"Dataset at {path} is empty.")

    # Standardize column names (strip whitespace)
    df.columns = [col.strip() for col in df.columns]

    # Validate that required features are present
    validate_columns(df)

    # Ensure target variable is numeric and drop any rows with missing target
    df[TARGET_REGRESSION] = pd.to_numeric(df[TARGET_REGRESSION], errors="coerce")
    if df[TARGET_REGRESSION].isnull().any():
        df = df.dropna(subset=[TARGET_REGRESSION]).copy()

    return df


def validate_columns(df: pd.DataFrame) -> bool:
    """Validate that the dataset contains all expected training features and target.

    Args:
        df: DataFrame to validate.

    Returns:
        bool: True if valid.

    Raises:
        ValueError: If any required feature or target is missing.
    """
    required_cols = set(ALL_FEATURES + [TARGET_REGRESSION])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(list(missing))}"
        )
    return True


def check_duplicates(df: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
    """Identify and count duplicate records in the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple[int, pd.DataFrame]: (count of duplicate rows, DataFrame of duplicate rows)
    """
    dup_mask = df.duplicated()
    dup_count = int(dup_mask.sum())
    dup_rows = df[dup_mask].copy()
    return dup_count, dup_rows


def get_missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute missing values count and percentage per column.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Summary table with columns ['Column', 'Missing Count', 'Missing Percentage (%)'].
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    summary = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing_count.values,
        "Missing Percentage (%)": missing_pct.round(2).values,
    })
    return summary


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate high-level metadata and descriptive statistics for the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        Dict[str, Any]: Dictionary containing dataset metrics and statistics.
    """
    dup_count, _ = check_duplicates(df)
    missing_summary = get_missing_values_summary(df)
    total_missing = int(missing_summary["Missing Count"].sum())

    pass_count = int((df[TARGET_REGRESSION] >= PASS_GRADE_THRESHOLD).sum())
    fail_count = len(df) - pass_count
    pass_pct = (pass_count / len(df)) * 100 if len(df) > 0 else 0.0

    return {
        "total_students": len(df),
        "total_columns": len(df.columns),
        "duplicate_records": dup_count,
        "total_missing_values": total_missing,
        "average_final_grade": float(df[TARGET_REGRESSION].mean()),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_percentage": round(pass_pct, 2),
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "numerical_stats": df.describe(),
    }


def split_features_and_targets(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate the input features from regression and classification targets.

    Prevents data leakage by strictly excluding G1, G2, and unrelated demographic features.

    Args:
        df: Full dataset DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.Series]:
            - X: DataFrame containing only selected predictor features.
            - y_reg: Series of continuous final grades G3 (0-20).
            - y_class: Series of binary classification labels (1: Pass, 0: Fail).
    """
    validate_columns(df)

    # Strictly select only model input features
    X = df[ALL_FEATURES].copy()

    # Regression target: G3 continuous grade
    y_reg = df[TARGET_REGRESSION].copy()

    # Classification target: 1 if G3 >= 10 (Pass), 0 if G3 < 10 (Fail)
    y_class = (y_reg >= PASS_GRADE_THRESHOLD).astype(int)
    y_class.name = "Pass_Status"

    return X, y_reg, y_class

