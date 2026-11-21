"""Student Performance Prediction System.

A professional, modular Streamlit web application for analyzing student data
and predicting academic performance using machine learning.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

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
)
from src.prediction import predict_student
from src.train_models import train_and_evaluate_all

# Set base project directory for reliable cross-platform path resolution
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student-mat.csv"

# Streamlit Page Configuration
st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-pass {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-fail {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .disclaimer-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 14px;
        border-radius: 6px;
        margin-top: 20px;
        color: #1E40AF;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_cached_dataset(filepath: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load and cache the student dataset."""
    return load_data(filepath)


@st.cache_resource
def get_cached_models(_df: pd.DataFrame) -> dict:
    """Train and cache the machine learning pipelines and evaluation metrics."""
    return train_and_evaluate_all(_df)


def render_header():
    """Render top page header and breadcrumbs."""
    st.markdown(
        '<div class="main-title">🎓 Student Performance Prediction System</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">End-to-end Machine Learning System for predicting secondary student academic outcomes and mitigating risk.</div>',
        unsafe_allow_html=True,
    )
    st.divider()


def render_dashboard(df: pd.DataFrame, summary: dict):
    """Render the executive analytics dashboard."""
    st.subheader("📊 Executive Overview & Key Metrics")

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Students", f"{summary['total_students']}")
    with col2:
        st.metric("Average Final Grade", f"{summary['average_final_grade']:.2f} / 20")
    with col3:
        st.metric("Passing Students", f"{summary['pass_count']}")
    with col4:
        st.metric("Failing Students", f"{summary['fail_count']}")
    with col5:
        st.metric("Overall Pass Rate", f"{summary['pass_percentage']}%")

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📈 Performance Distribution & Factor Analysis")

    # Visualization Row 1
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(
            df[TARGET_REGRESSION],
            bins=21,
            kde=True,
            color="#2563EB",
            edgecolor="white",
            ax=ax,
        )
        ax.axvline(
            PASS_GRADE_THRESHOLD,
            color="#DC2626",
            linestyle="--",
            linewidth=2,
            label=f"Passing Grade Threshold ({PASS_GRADE_THRESHOLD})",
        )
        ax.set_title("Distribution of Final Grades (G3)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Final Grade (0 - 20)")
        ax.set_ylabel("Student Count")
        ax.legend(frameon=True)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        st.pyplot(fig)
        plt.close(fig)

    with chart_col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        status_counts = pd.Series({
            "Passed (G3 ≥ 10)": summary["pass_count"],
            "Failed (G3 < 10)": summary["fail_count"],
        })
        colors = ["#10B981", "#EF4444"]
        ax.pie(
            status_counts.values,
            labels=status_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        ax.set_title("Pass vs. Fail Proportion", fontsize=12, fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)

    # Visualization Row 2
    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        fig, ax = plt.subplots(figsize=(6, 4))
        study_map = {
            1: "< 2 hours",
            2: "2 to 5 hours",
            3: "5 to 10 hours",
            4: "> 10 hours",
        }
        df_plot = df.copy()
        df_plot["Study Time Category"] = df_plot["studytime"].map(study_map)
        order = ["< 2 hours", "2 to 5 hours", "5 to 10 hours", "> 10 hours"]
        sns.barplot(
            data=df_plot,
            x="Study Time Category",
            y=TARGET_REGRESSION,
            order=order,
            hue="Study Time Category",
            palette="Blues_d",
            legend=False,
            ax=ax,
            errorbar=None,
        )
        ax.set_title("Mean Final Grade by Weekly Study Time", fontsize=12, fontweight="bold")
        ax.set_xlabel("Study Time per Week")
        ax.set_ylabel("Average G3 Grade")
        ax.set_ylim(0, 20)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        st.pyplot(fig)
        plt.close(fig)

    with chart_col4:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=df,
            x="absences",
            y=TARGET_REGRESSION,
            hue=(df[TARGET_REGRESSION] >= PASS_GRADE_THRESHOLD).map({True: "Passed", False: "Failed"}),
            palette={"Passed": "#10B981", "Failed": "#EF4444"},
            alpha=0.7,
            s=45,
            ax=ax,
        )
        ax.set_title("School Absences vs. Final Grade", fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of School Absences")
        ax.set_ylabel("Final Grade (G3)")
        ax.set_ylim(0, 20)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(title="Outcome", frameon=True)
        st.pyplot(fig)
        plt.close(fig)


def render_dataset_explorer(df: pd.DataFrame, summary: dict):
    """Render the exploratory data inspection section."""
    st.subheader("🔍 Dataset Inspection & Schema Details")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", summary["total_students"])
    with col2:
        st.metric("Total Features", summary["total_columns"])
    with col3:
        st.metric("Duplicate Rows", summary["duplicate_records"])
    with col4:
        st.metric("Missing Values", summary["total_missing_values"])

    st.markdown("#### 📋 Raw Dataset Preview")
    preview_mode = st.radio(
        "Preview slice:",
        ["First 10 Rows", "Last 10 Rows", "Full Dataset"],
        horizontal=True,
    )
    if preview_mode == "First 10 Rows":
        st.dataframe(df.head(10), use_container_width=True)
    elif preview_mode == "Last 10 Rows":
        st.dataframe(df.tail(10), use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

    st.markdown("#### 🧩 Column Metadata & Data Types")
    missing_summary = get_missing_values_summary(df)
    missing_summary["Data Type"] = [str(df[col].dtype) for col in missing_summary["Column"]]
    missing_summary["Unique Values"] = [int(df[col].nunique()) for col in missing_summary["Column"]]
    st.dataframe(missing_summary, use_container_width=True, hide_index=True)

    st.markdown("#### 🔢 Numerical Features Summary Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)


def render_model_performance(model_results: dict):
    """Render machine learning model benchmarks and confusion matrices."""
    st.subheader("⚙️ Machine Learning Models & Evaluation Benchmarks")

    best_reg = model_results["best_regression_name"]
    best_clf = model_results["best_classification_name"]

    col1, col2 = st.columns(2)
    with col1:
        st.success(
            f"🏆 **Best Regression Model:** {best_reg}\n\n"
            f"Evaluated by lowest RMSE (Root Mean Squared Error)."
        )
    with col2:
        st.success(
            f"🏆 **Best Classification Model:** {best_clf}\n\n"
            f"Evaluated by highest F1 Score."
        )

    st.markdown("---")

    # Regression Section
    st.markdown("### 1. Regression Models (Predicting Final Grade G3: 0–20)")
    reg_df = model_results["regression_metrics"].copy()
    st.dataframe(
        reg_df.style.highlight_min(subset=["RMSE", "MAE"], color="#DEF7EC")
        .highlight_max(subset=["R2 Score"], color="#DEF7EC"),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "**Regression Evaluation Metrics Guide:**\n"
        "- **MAE (Mean Absolute Error):** Average magnitude of prediction errors in grade points.\n"
        "- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors; the primary selection metric.\n"
        "- **R² Score:** Proportion of final grade variance explained by the model features."
    )

    st.markdown("---")

    # Classification Section
    st.markdown("### 2. Classification Models (Predicting Pass vs. Fail: G3 ≥ 10)")
    clf_df = model_results["classification_metrics"].copy()
    st.dataframe(
        clf_df.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1 Score"],
            color="#DEF7EC",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "**Classification Evaluation Metrics Guide:**\n"
        "- **Accuracy:** Overall proportion of correct Pass/Fail classifications.\n"
        "- **Precision:** Accuracy when predicting a student will pass.\n"
        "- **Recall:** Proportion of passing students correctly identified.\n"
        "- **F1 Score:** Harmonic mean of Precision and Recall; the primary selection metric."
    )

    st.markdown("#### 🎯 Confusion Matrices on Holdout Test Split (20% Stratified)")
    cm_col1, cm_col2 = st.columns(2)

    cms = model_results["confusion_matrices"]
    model_names = list(cms.keys())

    for idx, (col, m_name) in enumerate(zip([cm_col1, cm_col2], model_names)):
        with col:
            cm = cms[m_name]
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["Predicted Fail", "Predicted Pass"],
                yticklabels=["Actual Fail", "Actual Pass"],
                ax=ax,
            )
            ax.set_title(f"{m_name}\nConfusion Matrix", fontsize=11, fontweight="bold")
            st.pyplot(fig)
            plt.close(fig)


def render_prediction_form(model_results: dict):
    """Render single-student interactive prediction form and results."""
    st.subheader("🎯 Real-Time Student Performance Prediction")
    st.write(
        "Enter student academic history and behavioral attributes below to generate "
        "model estimates for the continuous final grade and pass/fail classification."
    )

    best_reg_pipe = model_results["best_regression_pipeline"]
    best_clf_pipe = model_results["best_classification_pipeline"]
    best_reg_name = model_results["best_regression_name"]
    best_clf_name = model_results["best_classification_name"]

    with st.form("student_prediction_form"):
        st.markdown("##### 📚 1. Academic & Attendance Factors")
        col1, col2 = st.columns(2)

        with col1:
            studytime_label = st.selectbox(
                "Weekly Study Time:",
                options=[
                    "1: < 2 hours",
                    "2: 2 to 5 hours",
                    "3: 5 to 10 hours",
                    "4: > 10 hours",
                ],
                index=1,
                help="Amount of dedicated self-study time per week.",
            )
            studytime = int(studytime_label.split(":")[0])

            failures = st.number_input(
                "Past Course Failures (0 to 4):",
                min_value=0,
                max_value=4,
                value=0,
                step=1,
                help="Number of past academic class failures.",
            )

        with col2:
            absences = st.slider(
                "School Absences:",
                min_value=0,
                max_value=93,
                value=4,
                help="Cumulative recorded school absences.",
            )

            traveltime_label = st.selectbox(
                "Home-to-School Travel Time:",
                options=[
                    "1: < 15 min.",
                    "2: 15 to 30 min.",
                    "3: 30 min. to 1 hour",
                    "4: > 1 hour",
                ],
                index=0,
                help="Commute duration to school.",
            )
            traveltime = int(traveltime_label.split(":")[0])

        st.markdown("##### 🤝 2. Support System & Behavioral Environment")
        col3, col4, col5 = st.columns(3)

        with col3:
            schoolsup = st.radio(
                "Extra School Support (Remedial):",
                options=["no", "yes"],
                horizontal=True,
            )
            famsup = st.radio(
                "Family Educational Support:",
                options=["yes", "no"],
                horizontal=True,
            )

        with col4:
            paid = st.radio(
                "Paid Extra Subject Classes:",
                options=["no", "yes"],
                horizontal=True,
            )
            activities = st.radio(
                "Extracurricular Activities:",
                options=["yes", "no"],
                horizontal=True,
            )

        with col5:
            higher = st.radio(
                "Wants Higher Education:",
                options=["yes", "no"],
                horizontal=True,
            )
            internet = st.radio(
                "Internet Access at Home:",
                options=["yes", "no"],
                horizontal=True,
            )

        submitted = st.form_submit_button(
            "🚀 Generate Performance Prediction",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        input_data = {
            "studytime": studytime,
            "failures": failures,
            "absences": absences,
            "traveltime": traveltime,
            "schoolsup": schoolsup,
            "famsup": famsup,
            "paid": paid,
            "activities": activities,
            "higher": higher,
            "internet": internet,
        }

        try:
            with st.spinner("Executing model pipelines..."):
                result = predict_student(
                    reg_pipeline=best_reg_pipe,
                    clf_pipeline=best_clf_pipe,
                    student_data=input_data,
                    reg_model_name=best_reg_name,
                    clf_model_name=best_clf_name,
                )
        except ValueError as err:
            st.error(f"❌ Input Validation Error: {str(err)}")
            return

        st.markdown("### 📋 Prediction Results")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Final Grade (G3)</div>
                    <div class="metric-value">{result['predicted_grade']} / 20</div>
                    <p style="margin-top: 8px; color: #475569; font-size: 0.9rem;">
                        Estimated by: <strong>{result['reg_model_name']}</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col2:
            status = result["pass_status"]
            badge_class = "badge-pass" if status == "Pass" else "badge-fail"
            prob_text = (
                f"(Pass Probability: {result['pass_probability']}%)"
                if result["pass_probability"] is not None
                else ""
            )

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Academic Status</div>
                    <div style="margin-top: 6px;">
                        <span class="{badge_class}" style="font-size: 1.4rem;">{status}</span>
                    </div>
                    <p style="margin-top: 8px; color: #475569; font-size: 0.9rem;">
                        Estimated by: <strong>{result['clf_model_name']}</strong> {prob_text}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Academic Guidance & Progress Feedback
        if result["predicted_grade"] >= 16:
            st.success("🌟 **Academic Standing: Excellent** — Student demonstrates strong indicators for high achievement.")
        elif result["predicted_grade"] >= 14:
            st.info("📘 **Academic Standing: Good** — Student is tracking well above average requirements.")
        elif result["predicted_grade"] >= 10:
            st.warning("📙 **Academic Standing: Satisfactory** — Student is in passing range, but proactive study habits could strengthen mastery.")
        else:
            st.error("⚠️ **Academic Standing: At-Risk** — Student shows indicators consistent with academic difficulty. Consider academic intervention, attendance review, or study support.")

        # Required Disclaimer
        st.markdown(
            f"""
            <div class="disclaimer-box">
                <strong>⚠️ Ethical & Educational Note:</strong><br>
                {result['disclaimer']}
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    """Application entrypoint."""
    render_header()

    # Load dataset with error handling
    try:
        df = get_cached_dataset(DATA_PATH)
        summary = get_dataset_summary(df)
    except Exception as e:
        st.error(f"Error loading dataset from `{DATA_PATH}`: {str(e)}")
        st.stop()

    # Train and cache models
    with st.spinner("Initializing and training machine learning pipelines..."):
        try:
            model_results = get_cached_models(df)
        except Exception as e:
            st.error(f"Error during model training pipeline: {str(e)}")
            st.stop()

    # Sidebar Navigation
    st.sidebar.title("🧭 Navigation")
    section = st.sidebar.radio(
        "Select Section:",
        [
            "Dashboard",
            "Dataset Explorer",
            "Model Performance",
            "Student Prediction",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Quick Stats")
    st.sidebar.write(f"• **Dataset Size:** {len(df)} students")
    st.sidebar.write(f"• **Input Features:** {len(ALL_FEATURES)} attributes")
    st.sidebar.write(f"• **Overall Pass Rate:** {summary['pass_percentage']}%")
    st.sidebar.write(f"• **Best Regressor:** {model_results['best_regression_name']}")
    st.sidebar.write(f"• **Best Classifier:** {model_results['best_classification_name']}")

    st.sidebar.markdown("---")
    st.sidebar.caption("Student Performance Prediction System • Scikit-learn & Streamlit")

    # Render selected section
    if section == "Dashboard":
        render_dashboard(df, summary)
    elif section == "Dataset Explorer":
        render_dataset_explorer(df, summary)
    elif section == "Model Performance":
        render_model_performance(model_results)
    elif section == "Student Prediction":
        render_prediction_form(model_results)


if __name__ == "__main__":
    main()

