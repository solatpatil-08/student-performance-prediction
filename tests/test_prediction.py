"""Unit tests for src.prediction."""

from pathlib import Path
import pandas as pd
import pytest

from src.data_loader import load_data
from src.prediction import format_student_input, predict_student
from src.train_models import train_and_evaluate_all

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "student-mat.csv"


@pytest.fixture(scope="module")
def trained_models():
    """Module-level fixture to train models once for inference tests."""
    df = load_data(DATA_PATH)
    return train_and_evaluate_all(df)


@pytest.fixture
def valid_input():
    """Sample valid student input dictionary."""
    return {
        "studytime": 2,
        "failures": 0,
        "absences": 4,
        "traveltime": 1,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "higher": "yes",
        "internet": "yes",
    }


def test_format_student_input_valid(valid_input):
    """Verify format_student_input returns clean DataFrame with expected columns."""
    df = format_student_input(valid_input)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df["studytime"].iloc[0] == 2
    assert df["schoolsup"].iloc[0] == "no"


def test_format_student_input_missing_keys(valid_input):
    """Verify ValueError when required keys are missing."""
    del valid_input["studytime"]
    with pytest.raises(ValueError, match="Missing required input features"):
        format_student_input(valid_input)


def test_format_student_input_invalid_ranges(valid_input):
    """Verify ValueError on negative absences or invalid ranges."""
    bad_absences = valid_input.copy()
    bad_absences["absences"] = -1
    with pytest.raises(ValueError, match="absences must be between"):
        format_student_input(bad_absences)

    bad_studytime = valid_input.copy()
    bad_studytime["studytime"] = 10
    with pytest.raises(ValueError, match="studytime must be between"):
        format_student_input(bad_studytime)

    bad_cat = valid_input.copy()
    bad_cat["schoolsup"] = "invalid_binary"
    with pytest.raises(ValueError, match="must be 'yes' or 'no'"):
        format_student_input(bad_cat)


def test_predict_student_output_structure(trained_models, valid_input):
    """Verify inference results contain all required output fields and clamped values."""
    reg_pipe = trained_models["best_regression_pipeline"]
    clf_pipe = trained_models["best_classification_pipeline"]

    result = predict_student(
        reg_pipe,
        clf_pipe,
        valid_input,
        trained_models["best_regression_name"],
        trained_models["best_classification_name"],
    )

    assert "predicted_grade" in result
    assert "pass_status" in result
    assert "class_label" in result
    assert "pass_probability" in result
    assert "reg_model_name" in result
    assert "clf_model_name" in result
    assert "disclaimer" in result

    # Validate grade scale clamping [0, 20]
    assert 0.0 <= result["predicted_grade"] <= 20.0
    assert result["pass_status"] in {"Pass", "Fail"}
    assert result["class_label"] in {0, 1}
    if result["pass_probability"] is not None:
        assert 0.0 <= result["pass_probability"] <= 100.0


def test_predict_student_with_dataframe(trained_models, valid_input):
    """Verify predict_student accepts DataFrame input."""
    reg_pipe = trained_models["best_regression_pipeline"]
    clf_pipe = trained_models["best_classification_pipeline"]
    df_input = format_student_input(valid_input)

    result = predict_student(reg_pipe, clf_pipe, df_input)
    assert 0.0 <= result["predicted_grade"] <= 20.0

