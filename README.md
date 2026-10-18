# UniMatch AI

**An intelligent, explainable university recommendation system for graduate applicants.**

UniMatch AI takes a student's academic profile and answers three questions:

1. *My overall profile rating for admissions*
2. *Which universities am I eligible for?*
3. *What is my chance of admission, and why?*

The project follows a 10-task computational-intelligence pipeline, from raw data through classical ML, deep learning and Explainable AI (XAI). This repository contains the **data-science stages (Tasks 4-8)** and a Streamlit application that integrates their saved outputs into an applicant-facing workflow.

---

## Pipeline

| Task | Stage | Notebook | Main outputs |
|------|-------|----------|--------------|
| 4 | Data preparation & EDA | `Task4_Data_Preparation_EDA.ipynb` | Clean dataset, EDA charts, data-quality report |
| 5 | Feature engineering & selection | `Task5_Feature_Engineering.ipynb` | Feature matrices with a saved train/val/test split, selected features, category encodings |
| 6 | Machine learning models | `Task6_ML_Model_Development.ipynb` | Best model + scaler (`.pkl`), evaluation reports |
| 7 | Deep learning (DNN) | `Task7_Deep_Learning_Model.ipynb` | `.keras` / `.h5` models, training history, DNN vs ML comparison |
| 8 | Explainable AI | `Task8_Explainable_AI.ipynb` | SHAP, LIME, counterfactuals, trust scores, fairness report |
| 9 | Integrated application | `app.py` | Validated inputs, ML/DNN predictions, recommendations, explanations and HTML reports |

Models are built for two settings, **university** and **graduate** (see `outputs/task5_output/feature_matrix_*.csv`).

Tasks 1-3 (knowledge base, rule-based expert system, state-space search) and 9-10 (integration/deployment, Responsible-AI evaluation) are described in the final report in `reports/` and are not part of this repository's notebooks.

---

## Repository structure

```
UniMatch-AI/
├── README.md
├── app.py                 # integrated Streamlit application
├── .streamlit/
│   └── config.toml        # consistent light theme and accessible color palette
├── DEPLOYMENT.md          # local and hosted deployment instructions
├── requirements.txt
├── .gitignore
├── data/                  # input datasets (see data/README.md)
│   ├── README.md
│   └── raw/
├── notebooks/             # Tasks 4-8, run in order
├── reports/               # per-task reports (.md) + model_fixes_and_app_update.md
└── outputs/               # everything the notebooks generate
    ├── task5_output/      # feature matrices, selected features, category_encodings.json
    ├── task6_output/      # ML models, scalers, evaluation reports
    ├── task7_output/      # DNN models, history, curves, comparisons
    └── task8_output/      # SHAP, LIME, counterfactuals, trust, fairness
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

If you only want to explore models and explanations, you can skip Tasks 4-5 and start at Task 6, since the feature matrices are already in `outputs/task5_output/`.

### Run the application

The checked-in Task 5-8 artifacts are used for inference; the notebooks do not need to be rerun:

```bash
python -m streamlit run app.py
```

The app has four tabs:

- **Your profile**: a short, grouped form (grades, test scores, documents, experience, study plans) with
  help text on every field. GRE total is computed from verbal + quantitative, and CGPA on any 4 / 10 / 100
  scale is converted for the models.
- **Your matches**: overall admission outlook, and the best-matching universities labelled Safe / Target /
  Reach with admission chance, match score, profile fit and world rank. Every column is explained; the
  results download as an HTML report or CSV.
- **Why these results**: plain-language explanations of what helps and what holds the applicant back,
  for the overall outlook and for any recommended university, plus explained SHAP, LIME, counterfactual,
  trust-score and fairness outputs from Task 8.
- **Model details**: Task 5-8 evaluation tables and plots with a metric glossary.

Each estimate averages the selected ML model and the DNN; model choice is under *Advanced* in the
sidebar. See [`DEPLOYMENT.md`](DEPLOYMENT.md) for hosted deployment requirements.

---

## Outputs at a glance

**Task 6: ML models.** Logistic Regression, Decision Tree, Random Forest, SVM and XGBoost are compared in `evaluation_report_*.csv`. Task 5 saves a train / validation / test split (by applicant for the university data, so no applicant appears in more than one part); the best model per setting is chosen on validation ROC-AUC and reported on the untouched test rows. All models are saved with their scaler.

**Task 7: Deep learning.** A DNN is trained for each setting. See `hyperparameter_report.csv`, `overfitting_check.csv`, `dnn_vs_ml_comparison.csv` and the plots `accuracy_loss_curves.png` and `dnn_roc_confusion.png`. The DNN is the best university model (test ROC-AUC 0.780 vs 0.760 for the best ML model); for the graduate setting Logistic Regression and the DNN are effectively tied (0.956 vs 0.946).

**Training fixes.** The applicant-level split, the removal of a frequency-encoding leak, one-hot category encoding and validation-based model selection are documented, with before/after results, in [`reports/model_fixes_and_app_update.md`](reports/model_fixes_and_app_update.md).

**Task 8: Explainable AI.**
- **SHAP**: global summary and per-prediction waterfall plots
- **LIME**: local explanations, including interactive `.html` files
- **Counterfactuals**: what would raise a student's admission chance
- **Trust scores**: confidence in each prediction vs accuracy
- **Fairness**: group-level metrics in `fairness_report.csv` and `fairness_by_group.csv`
- **Permutation importance** and per-row **explainability dashboards**
- Full write-up in `reports/xai_report.md`

For the actual metrics, open the report CSVs; they are the source of truth for results.

---

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, TensorFlow/Keras, Streamlit, SHAP, LIME, matplotlib, seaborn, Jupyter.

---

## Reports

Documentation lives in `reports/`: one report each for Tasks 4, 5 and 6, 7 and 8 ; And a final report including Tasks 1-3 (knowledge base, rule-based expert system, state-space search) and 9-10 (integration/deployment, Responsible-AI evaluation)

`reports/model_fixes_and_app_update.md` records the review of model training (problems found, fixes, before/after metrics) and the application redesign.

---

## Limitations

- Models are trained on historical, self-reported and public data, so predictions are **estimates, not guarantees** of admission.
- Applicant datasets can carry selection and reporting bias. The fairness analysis in Task 8 is a first audit, not a certification.
- Rankings change every year; the rankings table is a 2026 snapshot.
