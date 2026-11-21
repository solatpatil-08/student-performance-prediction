"""Model training and evaluation module for Student Performance Prediction System."""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_loader import split_features_and_targets
from src.preprocessing import build_preprocessor


def compute_rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error compatible across scikit-learn versions."""
    try:
        from sklearn.metrics import root_mean_squared_error
        return float(root_mean_squared_error(y_true, y_pred))
    except ImportError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_regression_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, Pipeline], pd.DataFrame, str, Pipeline]:
    """Train and evaluate Linear Regression and Random Forest Regressor.

    Identifies the best model using the lowest RMSE.

    Args:
        X_train: Training features.
        y_train: Training continuous targets (G3).
        X_test: Test features.
        y_test: Test continuous targets (G3).

    Returns:
        Tuple[Dict[str, Pipeline], pd.DataFrame, str, Pipeline]:
            - Dictionary of trained pipelines
            - Metrics DataFrame (Model, MAE, RMSE, R2 Score)
            - Name of the best model
            - Best model pipeline
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
    }

    trained_pipelines: Dict[str, Pipeline] = {}
    metrics_records = []

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("regressor", model),
            ]
        )
        # Train strictly on training split to prevent data leakage
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = compute_rmse(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics_records.append({
            "Model": name,
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "R2 Score": round(float(r2), 4),
        })
        trained_pipelines[name] = pipeline

    metrics_df = pd.DataFrame(metrics_records)

    # Lowest RMSE is the best model
    best_row = metrics_df.sort_values(by="RMSE", ascending=True).iloc[0]
    best_name = best_row["Model"]
    best_pipeline = trained_pipelines[best_name]

    return trained_pipelines, metrics_df, best_name, best_pipeline


def train_classification_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, Pipeline], pd.DataFrame, Dict[str, np.ndarray], str, Pipeline]:
    """Train and evaluate Logistic Regression and Random Forest Classifier.

    Identifies the best model using the highest F1 Score.

    Args:
        X_train: Training features.
        y_train: Training binary labels (1: Pass, 0: Fail).
        X_test: Test features.
        y_test: Test binary labels.

    Returns:
        Tuple[Dict[str, Pipeline], pd.DataFrame, Dict[str, np.ndarray], str, Pipeline]:
            - Dictionary of trained pipelines
            - Metrics DataFrame (Model, Accuracy, Precision, Recall, F1 Score)
            - Dictionary of confusion matrices
            - Name of the best model
            - Best model pipeline
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
    }

    trained_pipelines: Dict[str, Pipeline] = {}
    metrics_records = []
    confusion_matrices: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", model),
            ]
        )
        # Train strictly on training split to prevent data leakage
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        metrics_records.append({
            "Model": name,
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1 Score": round(float(f1), 4),
        })
        trained_pipelines[name] = pipeline
        confusion_matrices[name] = cm

    metrics_df = pd.DataFrame(metrics_records)

    # Highest F1 Score is the best model
    best_row = metrics_df.sort_values(by="F1 Score", ascending=False).iloc[0]
    best_name = best_row["Model"]
    best_pipeline = trained_pipelines[best_name]

    return trained_pipelines, metrics_df, confusion_matrices, best_name, best_pipeline


def train_and_evaluate_all(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Orchestrate end-to-end dataset splitting, model training, and evaluation.

    Args:
        df: Raw student dataset DataFrame.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random state for reproducibility.

    Returns:
        Dict[str, Any]: Consolidated dictionary of trained models, metrics, and test data.

    Raises:
        ValueError: If the dataset is too small for cross-validation splitting.
    """
    if len(df) < 10:
        raise ValueError(
            f"Dataset has too few records to train models (minimum 10 required, got {len(df)})."
        )

    X, y_reg, y_class = split_features_and_targets(df)

    # Use stratification when classes are adequately populated (minimum 2 samples per class)
    stratify_target = None
    if y_class.nunique() > 1 and int(y_class.value_counts().min()) >= 2:
        stratify_target = y_class

    (
        X_train,
        X_test,
        y_reg_train,
        y_reg_test,
        y_class_train,
        y_class_test,
    ) = train_test_split(
        X,
        y_reg,
        y_class,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    reg_pipelines, reg_metrics, best_reg_name, best_reg_pipe = (
        train_regression_models(X_train, y_reg_train, X_test, y_reg_test)
    )

    clf_pipelines, clf_metrics, confusion_mats, best_clf_name, best_clf_pipe = (
        train_classification_models(
            X_train, y_class_train, X_test, y_class_test
        )
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_reg_train": y_reg_train,
        "y_reg_test": y_reg_test,
        "y_class_train": y_class_train,
        "y_class_test": y_class_test,
        "regression_models": reg_pipelines,
        "regression_metrics": reg_metrics,
        "best_regression_name": best_reg_name,
        "best_regression_pipeline": best_reg_pipe,
        "classification_models": clf_pipelines,
        "classification_metrics": clf_metrics,
        "confusion_matrices": confusion_mats,
        "best_classification_name": best_clf_name,
        "best_classification_pipeline": best_clf_pipe,
    }

