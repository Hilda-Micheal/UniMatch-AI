# UniMatch AI

**An intelligent, explainable university recommendation system for graduate applicants.**

UniMatch AI takes a student's academic profile and answers three questions:

1. *Which universities am I eligible for?*
2. *Which of them fit my budget, ranking preference and goals?*
3. *What is my chance of admission, and why?*

The project follows a 10-task computational-intelligence pipeline, from raw data through classical ML, deep learning and Explainable AI (XAI). This repository contains the **data-science stages (Tasks 4-8)**: data preparation, feature engineering, ML models, a deep neural network, and explainability.

---

## Pipeline

| Task | Stage | Notebook | Main outputs |
|------|-------|----------|--------------|
| 4 | Data preparation & EDA | `Task4_Data_Preparation_EDA.ipynb` | Clean dataset, EDA charts, data-quality report |
| 5 | Feature engineering & selection | `Task5_Feature_Engineering.ipynb` | Feature matrices, selected features |
| 6 | Machine learning models | `Task6_ML_Model_Development.ipynb` | Best model + scaler (`.pkl`), evaluation reports |
| 7 | Deep learning (DNN) | `Task7_Deep_Learning_Model_Development.ipynb` | `.keras` / `.h5` models, training history, DNN vs ML comparison |
| 8 | Explainable AI | `Task8_Explainable_AI.ipynb` | SHAP, LIME, counterfactuals, trust scores, fairness report |

Models are built for two settings, **university** and **graduate** (see `outputs/task5_outputs/feature_matrix_*.csv`).

Tasks 1-3 (knowledge base, rule-based expert system, state-space search) and 9-10 (integration/deployment, Responsible-AI evaluation) are described in the final report in `reports/` and are not part of this repository's notebooks.

---

## Repository structure

```
UniMatch-AI/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                  # input datasets (see data/README.md)
│   ├── README.md
│   └── raw/
├── notebooks/             # Tasks 4-8, run in order
├── reports/               # per-task reports (.md)
└── outputs/               # everything the notebooks generate
    ├── task5_outputs/     # feature matrices, selected features
    ├── task6_outputs/     # ML models, scalers, evaluation reports
    ├── task7_outputs/     # DNN models, history, curves, comparisons
    └── task8_outputs/     # SHAP, LIME, counterfactuals, trust, fairness
```

---

## Data

Four datasets are used: graduate-admission applicant records, a larger applicant-outcome dataset, US college admissions statistics, and a 2026 world-rankings table (QS / THE / ARWU).

Sources, columns and placement instructions are in **[`data/README.md`](data/README.md)**. Place the files in `data/raw/` before running Task 4.

---

## Getting started

```bash
git clone https://github.com/Hilda-Micheal/UniMatch-AI.git
cd UniMatch-AI

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook
```

Run the notebooks in `notebooks/` in order, Task 4 to Task 8. Each notebook reads the previous task's output from `outputs/`.

If you only want to explore models and explanations, you can skip Tasks 4-5 and start at Task 6, since the feature matrices are already in `outputs/task5_outputs/`.

---

## Outputs at a glance

**Task 6: ML models.** Logistic Regression, Decision Tree, Random Forest, SVM and XGBoost are compared in `evaluation_report_*.csv`. The best model per setting is saved with its scaler.

**Task 7: Deep learning.** A DNN is trained for each setting. See `hyperparameter_report.csv`, `overfitting_check.csv`, `dnn_vs_ml_comparison.csv` and the plots `accuracy_loss_curves.png` and `dnn_roc_confusion.png`.

**Task 8: Explainable AI.**
- **SHAP**: global summary and per-prediction waterfall plots
- **LIME**: local explanations, including interactive `.html` files
- **Counterfactuals**: what would raise a student's admission chance
- **Trust scores**: confidence in each prediction vs accuracy
- **Fairness**: group-level metrics in `fairness_report.csv` and `fairness_by_group.csv`
- **Permutation importance** and per-row **explainability dashboards**
- Full write-up in `outputs/task8_outputs/xai_report.md`

For the actual metrics, open the report CSVs; they are the source of truth for results.

---

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, TensorFlow/Keras, SHAP, LIME, matplotlib, seaborn, Jupyter.

---

## Reports

Documentation lives in `reports/`: one report each for Tasks 4, 5 and 6, 7 and 8 ; And a final report including Tasks 1-3 (knowledge base, rule-based expert system, state-space search) and 9-10 (integration/deployment, Responsible-AI evaluation)

---

## Limitations

- Models are trained on historical, self-reported and public data, so predictions are **estimates, not guarantees** of admission.
- Applicant datasets can carry selection and reporting bias. The fairness analysis in Task 8 is a first audit, not a certification.
- Rankings change every year; the rankings table is a 2026 snapshot.


