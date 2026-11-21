# 🎓 Student Performance Prediction System

An end-to-end, production-ready machine learning system and interactive web dashboard designed to analyze secondary school student academic records and predict future academic performance. The system estimates both a **continuous final grade** ($0$ to $20$ scale) and a **binary academic status** (Pass vs. Fail), enabling educators, academic advisors, and counselors to proactively identify at-risk students and implement targeted interventions.

Built with **Python**, **Scikit-learn**, **Pandas**, **NumPy**, and **Streamlit**.

---

## 🎯 Project Objectives

1. **Early Academic Intervention:** Provide educational institutions with an accessible tool to identify students at risk of academic failure before final grading periods.
2. **Leakage-Free Predictive Modeling:** Construct robust regression and classification pipelines that avoid target and feature leakage by strictly excluding collinear mid-term grades ($G1$, $G2$).
3. **Objective Model Benchmarking:** Compare linear models against tree-based ensembles on standardized holdout partitions using reproducible metrics (RMSE, MAE, $R^2$, F1 Score).
4. **Transparent Decision Support:** Provide an intuitive Streamlit interface that highlights model metrics, confusion matrices, prediction probabilities, and explicit educational disclaimers.

---

## 💻 Technology Stack

- **Core Language:** Python 3.10+ (Recommended: Python 3.11)
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Machine Learning & Preprocessing:** Scikit-learn (`Pipeline`, `ColumnTransformer`, `StandardScaler`, `SimpleImputer`, `OneHotEncoder`)
- **Interactive Web Dashboard:** Streamlit
- **Data Visualization:** Matplotlib, Seaborn
- **Automated Testing:** Pytest
- **Version Control & Hosting:** Git, GitHub, Streamlit Community Cloud

---

## 📌 Main Features

- **Executive Analytics Dashboard:**
  - Real-time KPIs: Total student count ($395$), average final grade ($10.42 / 20$), passed count ($265$), failed count ($130$), and overall pass rate ($67.09\%$).
  - Distribution and factor analysis: Final grade ($G3$) histogram with passing threshold line ($10$), Pass vs. Fail proportions, average grade categorized by weekly study time, and absences vs. final grade scatter.

- **Exploratory Dataset Explorer:**
  - Interactive raw dataset preview (First 10 Rows, Last 10 Rows, or Full Dataset).
  - Schema inspection: Column names, data types, unique value counts, and missing value percentages.
  - Comprehensive descriptive statistics (`describe()`) for numerical features.
  - Automated duplicate row detection and reporting.

- **Model Performance & Benchmarks:**
  - Automated comparison between Linear Models and Ensemble Tree algorithms.
  - **Regression Models:** Linear Regression vs. Random Forest Regressor evaluated on MAE, RMSE, and $R^2$ Score.
  - **Classification Models:** Logistic Regression vs. Random Forest Classifier evaluated on Accuracy, Precision, Recall, and F1 Score.
  - Dual holdout test confusion matrices rendered side-by-side with counts and labels.
  - Automated selection of the top-performing models (lowest RMSE for regression, highest F1 Score for classification).

- **Real-Time Student Prediction Interface:**
  - Intuitive input widgets (dropdowns, sliders, radios) for academic habits, commute time, past failures, and support systems.
  - Form validation with strict range and categorical checking.
  - In-memory inference producing continuous grade estimates and binary Pass/Fail classifications with confidence probability.
  - Prominent ethical and academic disclaimer on predictive machine learning limitations.

---

## 🤖 Machine Learning Algorithms & Selection

### 1. Regression (Target: $G3 \in [0, 20]$)
- **Linear Regression:** Baseline parametric regressor optimizing Ordinary Least Squares.
- **Random Forest Regressor:** Non-linear ensemble of 100 decision trees reducing variance and capturing non-linear feature interactions.
- **Decision Criterion:** **Lowest Root Mean Squared Error (RMSE)**. Penalizes large errors heavily, critical for grading thresholds.

### 2. Classification (Target: Pass if $G3 \ge 10$, else Fail)
- **Logistic Regression:** Linear classifier with $L_2$ regularization optimizing log-loss.
- **Random Forest Classifier:** Ensemble classifier of 100 decision trees utilizing Gini impurity splits.
- **Decision Criterion:** **Highest F1 Score**. Maximizes the harmonic balance of Precision and Recall on class distributions.

---

## 🛡️ Data Leakage Prevention & Feature Strategy

In student performance datasets, period grades ($G1$ and $G2$) are recorded during the academic year and exhibit extreme collinearity with the final grade ($G3$). Using $G1$ or $G2$ to predict $G3$ causes severe target leakage and produces artificially inflated accuracy without capturing behavioral drivers.

To ensure ethical, operational validity:
1. **$G1$, $G2$, and $G3$ are strictly excluded** from model input features $X$.
2. **Unrelated demographic features are excluded** to avoid biased decision boundaries.
3. Preprocessing pipelines (imputation, scaling, one-hot encoding) are **fitted strictly on the training partition** ($80\%$) and only transform the test set ($20\%$).

### Selected Predictor Features:

| Feature | Type | Description | Values / Encoding |
| :--- | :--- | :--- | :--- |
| `studytime` | Numerical | Weekly study time | $1$: $<2\text{h}$, $2$: $2-5\text{h}$, $3$: $5-10\text{h}$, $4$: $>10\text{h}$ |
| `failures` | Numerical | Number of past academic class failures | $0$ to $4$ |
| `absences` | Numerical | Number of recorded school absences | $0$ to $93$ |
| `traveltime` | Numerical | Home-to-school travel time | $1$: $<15\text{m}$, $2$: $15-30\text{m}$, $3$: $30-60\text{m}$, $4$: $>60\text{m}$ |
| `schoolsup` | Categorical | Extra educational/remedial support | `yes` / `no` |
| `famsup` | Categorical | Family educational support | `yes` / `no` |
| `paid` | Categorical | Extra paid subject tutoring classes | `yes` / `no` |
| `activities` | Categorical | Extracurricular activities participation | `yes` / `no` |
| `higher` | Categorical | Desire to pursue higher education | `yes` / `no` |
| `internet` | Categorical | Internet access at home | `yes` / `no` |

---

## 📊 Evaluation Metrics Explanation

| Metric | Type | Definition & Purpose |
| :--- | :--- | :--- |
| **MAE** | Regression | **Mean Absolute Error:** Average absolute grade point difference between actual and predicted grades. |
| **RMSE** | Regression | **Root Mean Squared Error:** Square root of mean squared errors. Heavily penalizes outliers. Primary regression selector. |
| **$R^2$ Score** | Regression | **Coefficient of Determination:** Proportion of final grade variance explained by the model features. |
| **Accuracy** | Classification | Overall proportion of correct classifications over all predictions. |
| **Precision** | Classification | Proportion of predicted passing students who actually passed: $\frac{TP}{TP + FP}$. |
| **Recall** | Classification | Proportion of actual passing students correctly identified: $\frac{TP}{TP + FN}$. |
| **F1 Score** | Classification | Harmonic mean of Precision and Recall: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$. Primary classification selector. |

---

## 🏗️ Project Folder Structure

```text
student-performance-prediction/
│
├── app.py                      # Main Streamlit web application & UI
│
├── data/
│   └── student-mat.csv         # Semicolon-delimited student dataset
│
├── src/
│   ├── __init__.py             # Package marker
│   ├── data_loader.py          # Data ingestion, schema validation, and summary utilities
│   ├── preprocessing.py       # Scikit-learn Pipeline and ColumnTransformer definitions
│   ├── train_models.py         # Model training, stratified splitting, evaluation, and selection
│   └── prediction.py           # Real-time inference formatting, grade clamping, and probability scoring
│
├── tests/
│   ├── __init__.py             # Test suite marker
│   ├── test_data_loader.py     # Ingestion and leakage prevention tests
│   ├── test_preprocessing.py   # Transformer, imputer, and encoder tests
│   ├── test_train_models.py    # Training and metrics computation tests
│   └── test_prediction.py      # Real-time inference and validation tests
│
├── requirements.txt            # Project dependencies for deployment
├── README.md                   # Comprehensive project documentation
└── .gitignore                  # Git tracking exclusions
```

---

## 🚀 Local Installation & Execution

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.11 and Python 3.13)
- `pip` package manager
- `git` version control

### 2. Clone the Repository
```bash
git clone https://github.com/solatpatil-08/student-performance-prediction.git
cd student-performance-prediction
```

### 3. Create and Activate a Virtual Environment
```bash
# On Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# On macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run Automated Tests
```bash
pytest -v
```

### 6. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
The application will launch in your default browser at `http://localhost:8501`.

---

## ☁️ Deployment Instructions (Streamlit Community Cloud)

This repository is pre-configured for direct, one-click deployment on **Streamlit Community Cloud**:

1. **Push Changes to GitHub:**
   Ensure the latest code is committed and pushed to your GitHub repository:
   ```bash
   git add .
   git commit -m "chore: prepare repository for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Access Streamlit Community Cloud:**
   - Navigate to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.

3. **Deploy New App:**
   - Click the **"Create app"** (or **"New app"**) button.
   - Select **"I already have an app"**.

4. **Configure App Settings:**
   - **Repository:** `solatpatil-08/student-performance-prediction`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL (optional):** Choose a custom subdomain (e.g., `student-performance-predictor`).

5. **Advanced Settings (Python Version):**
   - Click **"Advanced settings"**.
   - Select **Python 3.11** as the Python version for optimal stability.

6. **Deploy:**
   - Click **"Deploy!"**.
   - Streamlit Cloud will install dependencies from `requirements.txt` and launch the application.

---

## ⚖️ Ethical Considerations & Educational Disclaimer

> **Important Notice:**
> Machine learning predictions in this system represent statistical approximations derived from historical cohort patterns. They are **not deterministic forecasts** and must never be used for disciplinary decisions, academic tracking, or gatekeeping.
>
> Student academic trajectory is influenced by individual motivation, pedagogical support, health, and personal circumstances that cannot be fully captured in structured datasets. These estimates are intended exclusively as advisory signals to help educators deploy supportive tutoring and academic guidance.

---

## 📜 Dataset Citation & License

- **Dataset Source:** P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance*. In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.
- **UCI Machine Learning Repository:** [Student Performance Data Set](https://archive.ics.uci.edu/dataset/320/student+performance)
- Distributed under the open-source **MIT License**.
