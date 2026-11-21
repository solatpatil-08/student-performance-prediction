"""Single student prediction inference module for Student Performance Prediction System."""

from typing import Any, Dict, Optional
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data_loader import ALL_FEATURES

DISCLAIMER_TEXT = (
    "Disclaimer: Machine learning predictions represent statistical estimates "
    "derived from historical training patterns. They are not deterministic or "
    "guaranteed academic outcomes. True student success depends on diverse "
    "pedagogical, emotional, contextual, and individual growth factors."
)


def format_student_input(data: Dict[str, Any]) -> pd.DataFrame:
    """Format and validate a dictionary of student features into a single-row DataFrame.

    Args:
        data: Dictionary of input features.

    Returns:
        pd.DataFrame: Formatted 1-row DataFrame ready for pipeline inference.

    Raises:
        ValueError: If any expected feature is missing or has invalid values.
    """
    if not isinstance(data, dict):
        raise ValueError("Student input data must be provided as a dictionary.")

    missing = [feat for feat in ALL_FEATURES if feat not in data]
    if missing:
        raise ValueError(f"Missing required input features for prediction: {missing}")

    # Validate and cast numerical inputs
    try:
        studytime = int(data["studytime"])
        if studytime < 1 or studytime > 4:
            raise ValueError(f"studytime must be between 1 and 4, got: {studytime}")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid value for 'studytime': {str(e)}") from e

    try:
        failures = int(data["failures"])
        if failures < 0 or failures > 4:
            raise ValueError(f"failures must be between 0 and 4, got: {failures}")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid value for 'failures': {str(e)}") from e

    try:
        absences = int(data["absences"])
        if absences < 0 or absences > 100:
            raise ValueError(f"absences must be between 0 and 100, got: {absences}")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid value for 'absences': {str(e)}") from e

    try:
        traveltime = int(data["traveltime"])
        if traveltime < 1 or traveltime > 4:
            raise ValueError(f"traveltime must be between 1 and 4, got: {traveltime}")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid value for 'traveltime': {str(e)}") from e

    # Validate and clean categorical inputs
    valid_binary = {"yes", "no"}
    cat_cleaned = {}
    for cat_feat in ["schoolsup", "famsup", "paid", "activities", "higher", "internet"]:
        val = str(data[cat_feat]).strip().lower()
        if val not in valid_binary:
            raise ValueError(
                f"Invalid value for '{cat_feat}': must be 'yes' or 'no', got: '{val}'"
            )
        cat_cleaned[cat_feat] = val

    row = {
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "traveltime": traveltime,
        **cat_cleaned,
    }

    return pd.DataFrame([row], columns=ALL_FEATURES)


def predict_student(
    reg_pipeline: Pipeline,
    clf_pipeline: Pipeline,
    student_data: Dict[str, Any] | pd.DataFrame,
    reg_model_name: Optional[str] = None,
    clf_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate both regression (final grade) and classification (Pass/Fail) predictions.

    Args:
        reg_pipeline: Trained Scikit-learn regression pipeline.
        clf_pipeline: Trained Scikit-learn classification pipeline.
        student_data: Dictionary of input values or single-row DataFrame.
        reg_model_name: Optional name of the regression model used.
        clf_model_name: Optional name of the classification model used.

    Returns:
        Dict[str, Any]: Prediction results, probabilities, model metadata, and educational disclaimer.
    """
    if isinstance(student_data, dict):
        df_input = format_student_input(student_data)
    elif isinstance(student_data, pd.DataFrame):
        missing = [feat for feat in ALL_FEATURES if feat not in student_data.columns]
        if missing:
            raise ValueError(f"DataFrame input missing required columns: {missing}")
        df_input = student_data[ALL_FEATURES].copy()
    else:
        raise ValueError("student_data must be a dictionary or a pandas DataFrame.")

    # Infer model names if not provided
    if not reg_model_name:
        if hasattr(reg_pipeline, "named_steps") and "regressor" in reg_pipeline.named_steps:
            reg_model_name = type(reg_pipeline.named_steps["regressor"]).__name__
        else:
            reg_model_name = "Regression Model"

    if not clf_model_name:
        if hasattr(clf_pipeline, "named_steps") and "classifier" in clf_pipeline.named_steps:
            clf_model_name = type(clf_pipeline.named_steps["classifier"]).__name__
        else:
            clf_model_name = "Classification Model"

    # Regression: Predict grade on 0-20 scale
    raw_grade = float(reg_pipeline.predict(df_input)[0])
    # Clamp between 0.0 and 20.0
    clamped_grade = max(0.0, min(20.0, raw_grade))
    predicted_grade = round(clamped_grade, 2)

    # Classification: Predict pass (1) or fail (0)
    class_pred = int(clf_pipeline.predict(df_input)[0])
    pass_status = "Pass" if class_pred == 1 else "Fail"

    # Prediction confidence / probability if available
    pass_probability = None
    if hasattr(clf_pipeline, "predict_proba"):
        try:
            probabilities = clf_pipeline.predict_proba(df_input)[0]
            # Probability of Class 1 (Pass)
            pass_probability = round(float(probabilities[1]) * 100, 1)
        except Exception:
            pass_probability = None

    return {
        "predicted_grade": predicted_grade,
        "raw_grade": round(raw_grade, 2),
        "class_label": class_pred,
        "pass_status": pass_status,
        "pass_probability": pass_probability,
        "reg_model_name": reg_model_name,
        "clf_model_name": clf_model_name,
        "disclaimer": DISCLAIMER_TEXT,
    }

