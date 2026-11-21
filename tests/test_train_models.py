"""Unit tests for src.train_models."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data_loader import load_data
from src.train_models import (
    compute_rmse,
    train_and_evaluate_all,
    train_classification_models,
    train_regression_models,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "student-mat.csv"


def test_compute_rmse():
    """Verify RMSE calculation accuracy."""
    y_true = np.array([10.0, 12.0, 14.0])
    y_pred = np.array([10.0, 10.0, 10.0])
    # squared diffs: 0, 4, 16 -> mean = 20 / 3 = 6.6667 -> sqrt = 2.58198
    rmse = compute_rmse(y_true, y_pred)
    assert np.isclose(rmse, np.sqrt(20.0 / 3.0), atol=1e-4)


def test_train_and_evaluate_all_pipeline():
    """Verify full end-to-end training, metrics calculation, and model selection."""
    df = load_data(DATA_PATH)
    results = train_and_evaluate_all(df)

    # Check regression results
    reg_models = results["regression_models"]
    assert "Linear Regression" in reg_models
    assert "Random Forest Regressor" in reg_models
    assert isinstance(results["best_regression_pipeline"], Pipeline)

    reg_metrics = results["regression_metrics"]
    assert set(reg_metrics.columns) == {"Model", "MAE", "RMSE", "R2 Score"}
    assert len(reg_metrics) == 2

    # Verify best regression model has lowest RMSE
    best_reg_name = results["best_regression_name"]
    min_rmse_model = reg_metrics.sort_values(by="RMSE", ascending=True).iloc[0]["Model"]
    assert best_reg_name == min_rmse_model

    # Check classification results
    clf_models = results["classification_models"]
    assert "Logistic Regression" in clf_models
    assert "Random Forest Classifier" in clf_models
    assert isinstance(results["best_classification_pipeline"], Pipeline)

    clf_metrics = results["classification_metrics"]
    assert set(clf_metrics.columns) == {
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    }
    assert len(clf_metrics) == 2

    # Verify best classification model has highest F1 Score
    best_clf_name = results["best_classification_name"]
    max_f1_model = clf_metrics.sort_values(by="F1 Score", ascending=False).iloc[0]["Model"]
    assert best_clf_name == max_f1_model

    # Check confusion matrices
    cms = results["confusion_matrices"]
    for name, cm in cms.items():
        assert cm.shape == (2, 2)
        assert cm.sum() == len(results["y_class_test"])


def test_train_and_evaluate_too_few_records():
    """Verify ValueError is raised when dataset has fewer than 10 records."""
    df_small = pd.DataFrame({
        "studytime": [1, 2],
        "failures": [0, 0],
        "absences": [2, 4],
        "traveltime": [1, 1],
        "schoolsup": ["no", "no"],
        "famsup": ["yes", "yes"],
        "paid": ["no", "no"],
        "activities": ["no", "yes"],
        "higher": ["yes", "yes"],
        "internet": ["yes", "yes"],
        "G3": [12, 14],
    })
    with pytest.raises(ValueError, match="too few records"):
        train_and_evaluate_all(df_small)

