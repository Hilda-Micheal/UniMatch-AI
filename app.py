from __future__ import annotations

import html
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from xgboost.core import XGBoostError


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
TASK5 = OUTPUTS / "task5_output"
TASK6 = OUTPUTS / "task6_output"
TASK7 = OUTPUTS / "task7_output"
TASK8 = OUTPUTS / "task8_output"
RAW_DATA = ROOT / "data" / "raw"

GRAD_LIMITS = {
    "gre_score": (260.0, 340.0),
    "toefl_score": (0.0, 120.0),
    "sop": (0.0, 5.0),
    "lor": (0.0, 5.0),
    "research": (0.0, 1.0),
}
UNIVERSITY_LIMITS = {
    "grev": (130.0, 170.0),
    "greq": (130.0, 170.0),
    "grea": (0.0, 6.0),
    "toeflscore": (0.0, 120.0),
    "researchexp": (0.0, 50.0),
    "industryexp": (0.0, 50.0),
}


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required project artifact was not found: {path}")
    return pd.read_csv(path, **kwargs)


@st.cache_data(show_spinner="Loading project data and model reports...")
def load_data() -> dict[str, Any]:
    """Load only the historical columns required by the live inference workflow."""
    feature_columns_path = TASK6 / "models" / "feature_columns.json"
    dnn_config_path = TASK7 / "dnn_config.json"
    for path in (feature_columns_path, dnn_config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required project artifact was not found: {path}")

    history_columns = [
        "univname",
        "department",
        "specialization",
        "cgpa_pct",
        "grev",
        "greq",
        "toeflscore",
        "researchexp",
        "industryexp",
        "toppercgpa",
        "admit_chance_index",
        "profile_distance",
        "split",
        "admit",
    ]
    history = _read_csv(TASK5 / "university_features_full.csv", usecols=history_columns)
    graduate = _read_csv(TASK5 / "feature_matrix_graduate.csv")
    graduate_metrics = _read_csv(TASK6 / "evaluation_report_graduate.csv")
    university_metrics = _read_csv(TASK6 / "evaluation_report_university.csv")
    rankings = _read_csv(RAW_DATA / "World_University_Rankings_2026.csv")

    with feature_columns_path.open(encoding="utf-8") as file:
        feature_columns = json.load(file)
    with dnn_config_path.open(encoding="utf-8") as file:
        dnn_config = json.load(file)
    university_features = _read_csv(
        TASK5 / "feature_matrix_university.csv",
        usecols=feature_columns["university"],
    )

    optional_outputs: dict[str, pd.DataFrame] = {}
    for name, path in {
        "graduate_correlation": TASK5 / "correlation_matrix_graduate.csv",
        "university_correlation": TASK5 / "correlation_matrix_university.csv",
        "shap": TASK8 / "shap_report.csv",
        "lime": TASK8 / "lime_report.csv",
        "fairness": TASK8 / "fairness_report.csv",
        "fairness_by_group": TASK8 / "fairness_by_group.csv",
        "trust": TASK8 / "trust_score_report.csv",
        "trust_scores": TASK8 / "trust_scores.csv",
        "counterfactuals": TASK8 / "counterfactual_analysis.csv",
        "university_importance": TASK5 / "feature_importance_university.csv",
        "graduate_importance": TASK5 / "feature_importance_graduate.csv",
        "permutation_importance": TASK8 / "feature_importance_permutation.csv",
        "dnn_comparison": TASK7 / "dnn_vs_ml_comparison.csv",
        "dnn_evaluation": TASK7 / "evaluation_report_dnn.csv",
        "dnn_hyperparameters": TASK7 / "hyperparameter_report.csv",
        "dnn_overfitting": TASK7 / "overfitting_check.csv",
        "graduate_training_history": TASK7 / "training_history_graduate.csv",
        "university_training_history": TASK7 / "training_history_university.csv",
    }.items():
        if path.is_file():
            optional_outputs[name] = pd.read_csv(path, index_col=0 if "correlation" in name else None)

    return {
        "history": history,
        "graduate": graduate,
        "university_features": university_features,
        "graduate_metrics": graduate_metrics,
        "university_metrics": university_metrics,
        "rankings": rankings,
        "feature_columns": feature_columns,
        "dnn_config": dnn_config,
        "reports": optional_outputs,
    }


@st.cache_resource(show_spinner="Loading the selected machine-learning model...")
def load_ml_model(dataset: str, model_name: str) -> Any:
    model_path = TASK6 / "models" / dataset / f"{model_name.lower().replace(' ', '_')}.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(f"Required ML model was not found: {model_path}")
    return joblib.load(model_path)


@st.cache_resource(show_spinner="Loading the deep-learning model...")
def load_dnn_model(dataset: str) -> Any:
    model_path = TASK7 / f"dnn_{dataset}.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"Required DNN model was not found: {model_path}")
    try:
        from tensorflow import keras
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required for deep-learning predictions. Install the packages "
            "listed in requirements.txt and restart the app."
        ) from error
    return keras.models.load_model(model_path, compile=False)


@st.cache_resource(show_spinner=False)
def load_scaler(dataset: str, scaler_type: str) -> Any:
    if scaler_type == "dnn":
        path = TASK7 / f"dnn_scaler_{dataset}.pkl"
    else:
        path = TASK6 / "models" / dataset / "scaler.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Required feature scaler was not found: {path}")
    return joblib.load(path)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_profile(profile: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field, (minimum, maximum) in GRAD_LIMITS.items():
        value = profile.get(field)
        if not _finite(value) or not minimum <= float(value) <= maximum:
            errors.append(f"{field.replace('_', ' ').title()} must be between {minimum:g} and {maximum:g}.")

    scale = profile.get("cgpa_scale")
    cgpa = profile.get("cgpa")
    if not _finite(scale) or float(scale) not in (4.0, 10.0, 100.0):
        errors.append("CGPA scale must be 4, 10, or 100.")
    else:
        if not _finite(cgpa) or not 0 <= float(cgpa) <= float(scale):
            errors.append(f"CGPA must be between 0 and {float(scale):g} for the selected scale.")
        topper = profile.get("toppercgpa")
        if not _finite(topper) or not 0 <= float(topper) <= float(scale):
            errors.append(f"Topper CGPA must be between 0 and {float(scale):g} for the selected scale.")

    for field, (minimum, maximum) in UNIVERSITY_LIMITS.items():
        value = profile.get(field)
        if not _finite(value) or not minimum <= float(value) <= maximum:
            errors.append(f"{field.replace('_', ' ').title()} must be between {minimum:g} and {maximum:g}.")

    if not profile.get("department"):
        errors.append("Choose a department from the available historical categories.")
    if not profile.get("specialization"):
        errors.append("Choose a specialization from the available historical categories.")
    return errors


def robust_minmax(value: float, series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        raise ValueError("Cannot compute a feature scale because its training data is empty.")
    low, high = values.quantile([0.01, 0.99]).astype(float)
    return float(np.clip(value, low, high) - low) / float(high - low + 1e-9)


def graduate_feature_row(profile: dict[str, Any], graduate_data: pd.DataFrame) -> pd.DataFrame:
    weights = {
        "gre_score": 0.25,
        "toefl_score": 0.15,
        "cgpa": 0.30,
        "sop": 0.10,
        "lor": 0.10,
        "research": 0.10,
    }
    chance_index = sum(
        robust_minmax(float(profile[column]), graduate_data[column]) * weight
        for column, weight in weights.items()
    )
    return pd.DataFrame(
        [{"admit_chance_index": chance_index, "toefl_score": float(profile["toefl_score"]), "sop": float(profile["sop"])}]
    )


def _category_code(value: str, series: pd.Series) -> int:
    categories = sorted(series.fillna("nan").astype(str).unique())
    try:
        return categories.index(str(value))
    except ValueError as error:
        raise ValueError(f"{value!r} is not in the model's historical category vocabulary.") from error


def _normalised_frequency(value: str, series: pd.Series) -> float:
    frequencies = series.fillna("nan").astype(str).value_counts(normalize=True)
    if str(value) not in frequencies:
        raise ValueError(f"{value!r} is not in the model's historical category vocabulary.")
    return float(frequencies[str(value)])


def _profile_distance(profile: dict[str, Any], university: str, history: pd.DataFrame) -> float:
    train_admits = history[(history["split"] == "train") & (history["admit"] == 1)]
    cohort = train_admits[train_admits["univname"].astype(str) == str(university)]
    if cohort.empty:
        median_distance = pd.to_numeric(history["profile_distance"], errors="coerce").median()
        if pd.isna(median_distance):
            raise ValueError("Historical profile-distance reference statistics are unavailable.")
        return float(median_distance)

    applicant = {
        "cgpa_pct": min(float(profile["cgpa"]) / float(profile["cgpa_scale"]) * 100.0, 100.0),
        "greq": float(profile["greq"]),
        "grev": float(profile["grev"]),
        "toeflscore": float(profile["toeflscore"]),
    }
    distances: list[float] = []
    for column, value in applicant.items():
        mean = pd.to_numeric(cohort[column], errors="coerce").mean()
        standard_deviation = pd.to_numeric(cohort[column], errors="coerce").std()
        if pd.isna(mean):
            continue
        if pd.isna(standard_deviation) or standard_deviation == 0:
            standard_deviation = 1.0
        distances.append(abs((value - float(mean)) / float(standard_deviation)))
    if not distances:
        raise ValueError(f"No valid admitted-cohort statistics are available for {university}.")
    return float(np.mean(distances))


def university_feature_row(
    profile: dict[str, Any],
    university: str,
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    cgpa_pct = min(float(profile["cgpa"]) / float(profile["cgpa_scale"]) * 100.0, 100.0)
    index_weights = {
        "cgpa_pct": 0.30,
        "greq": 0.20,
        "grev": 0.15,
        "toeflscore": 0.15,
        "researchexp": 0.12,
        "industryexp": 0.08,
    }
    values = {
        "cgpa_pct": cgpa_pct,
        "greq": float(profile["greq"]),
        "grev": float(profile["grev"]),
        "toeflscore": float(profile["toeflscore"]),
        "researchexp": float(profile["researchexp"]),
        "industryexp": float(profile["industryexp"]),
    }
    chance_index = sum(
        robust_minmax(values[column], history[column]) * weight
        for column, weight in index_weights.items()
    )
    distance = _profile_distance(profile, university, history)
    fit_score = 1.0 / (1.0 + distance)
    row = {
        "univname_freq": _normalised_frequency(university, history["univname"]),
        "univname_label": _category_code(university, history["univname"]),
        "cgpa_pct": cgpa_pct,
        "admit_chance_index": chance_index,
        "profile_distance": distance,
        "department_label": _category_code(profile["department"], history["department"]),
        "grev": float(profile["grev"]),
        "toeflscore": float(profile["toeflscore"]),
        "greq": float(profile["greq"]),
        "specialization_freq": _normalised_frequency(profile["specialization"], history["specialization"]),
        "toppercgpa": float(profile["toppercgpa"]),
        "specialization_label": _category_code(profile["specialization"], history["specialization"]),
        "department_freq": _normalised_frequency(profile["department"], history["department"]),
        "grea": float(profile["grea"]),
        "industryexp": float(profile["industryexp"]),
    }
    return pd.DataFrame([row]), fit_score


def _ml_probability(
    model: Any,
    dataset: str,
    feature_row: pd.DataFrame,
) -> float:
    expected_columns = feature_row.columns.tolist()
    if dataset == "graduate":
        expected_columns = ["admit_chance_index", "toefl_score", "sop"]
    if dataset == "university":
        expected_columns = load_data()["feature_columns"]["university"]
    ordered = feature_row.reindex(columns=expected_columns)
    if ordered.isna().any().any():
        raise ValueError("One or more required model features could not be generated.")
    model_input = ordered
    if model.__class__.__name__ in {"LogisticRegression", "SVC"}:
        model_input = load_scaler(dataset, "ml").transform(ordered)
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"The selected {model.__class__.__name__} model does not support probability output.")
    return float(model.predict_proba(model_input)[0, 1])


def _dnn_probability(dataset: str, feature_row: pd.DataFrame) -> float:
    config = load_data()["dnn_config"][dataset]
    ordered = feature_row.reindex(columns=config["features"])
    if ordered.isna().any().any():
        raise ValueError("One or more required DNN features could not be generated.")
    scaled = load_scaler(dataset, "dnn").transform(ordered)
    probability = load_dnn_model(dataset).predict(scaled, verbose=0).reshape(-1)
    if len(probability) != 1 or not np.isfinite(probability[0]):
        raise ValueError(f"The {dataset} DNN returned an invalid probability.")
    return float(probability[0])


def _ranking_lookup(rankings: pd.DataFrame) -> dict[str, dict[str, Any]]:
    def key(value: Any) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value).lower())

    lookup: dict[str, dict[str, Any]] = {}
    for _, row in rankings.iterrows():
        lookup[key(row["university"])] = {
            "country": row.get("country"),
            "region": row.get("region"),
            "avg_rank_all3": pd.to_numeric(row.get("avg_rank_all3"), errors="coerce"),
        }
    return lookup


def recommend_universities(
    profile: dict[str, Any],
    data: dict[str, Any],
    ml_model_name: str,
    top_n: int,
) -> pd.DataFrame:
    history = data["history"]
    counts = history["univname"].dropna().astype(str).value_counts()
    candidates = sorted(counts[counts >= 20].index.tolist())
    if not candidates:
        raise ValueError("No university candidates meet the minimum historical sample size.")
    ml_model = load_ml_model("university", ml_model_name)
    ranking_info = _ranking_lookup(data["rankings"])
    results: list[dict[str, Any]] = []
    for university in candidates:
        feature_row, fit_score = university_feature_row(profile, university, history)
        ml_probability = _ml_probability(ml_model, "university", feature_row)
        dnn_probability = _dnn_probability("university", feature_row)
        match = ranking_info.get(re.sub(r"[^a-z0-9]", "", university.lower()), {})
        rank = match.get("avg_rank_all3")
        ranking_score = 1.0 / (1.0 + float(rank) / 100.0) if _finite(rank) else 0.0
        results.append(
            {
                "University": university,
                "Estimated admission probability": (ml_probability + dnn_probability) / 2.0,
                "ML estimate": ml_probability,
                "DNN estimate": dnn_probability,
                "Profile fit": fit_score,
                "Average rank": rank,
                "Country": match.get("country", "Not listed"),
                "Recommendation score": (
                    0.55 * ((ml_probability + dnn_probability) / 2.0)
                    + 0.30 * fit_score
                    + 0.15 * ranking_score
                ),
                "_features": feature_row,
            }
        )
    table = pd.DataFrame(results).sort_values("Recommendation score", ascending=False).head(top_n)
    return table.reset_index(drop=True)


def _sensitivity(
    feature_row: pd.DataFrame,
    training_features: pd.DataFrame,
    dataset: str,
    ml_model_name: str,
) -> pd.DataFrame:
    ml_model = load_ml_model(dataset, ml_model_name)
    current_ml = _ml_probability(ml_model, dataset, feature_row)
    current_dnn = _dnn_probability(dataset, feature_row)
    rows: list[dict[str, Any]] = []
    baselines = training_features.median(numeric_only=True)
    for column in feature_row.columns:
        if column not in baselines.index or pd.isna(baselines[column]):
            continue
        changed = feature_row.copy()
        changed.loc[:, column] = float(baselines[column])
        ml_change = current_ml - _ml_probability(ml_model, dataset, changed)
        dnn_change = current_dnn - _dnn_probability(dataset, changed)
        rows.append(
            {
                "Feature": column,
                "ML probability change vs training median": ml_change,
                "DNN probability change vs training median": dnn_change,
                "Largest absolute change": max(abs(ml_change), abs(dnn_change)),
            }
        )
    return pd.DataFrame(rows).sort_values("Largest absolute change", ascending=False).reset_index(drop=True)


def build_report_html(result: dict[str, Any]) -> str:
    profile_table = pd.DataFrame([result["profile"]]).to_html(index=False, escape=True, border=0)
    recommendations = result["recommendations"].drop(columns=["_features"], errors="ignore")
    recommendations_html = recommendations.to_html(index=False, escape=True, border=0, float_format=lambda n: f"{n:.3f}")
    sensitivity_html = result["sensitivity"].drop(columns=["Largest absolute change"]).to_html(
        index=False, escape=True, border=0, float_format=lambda n: f"{n:+.3f}"
    )
    university_sensitivity_html = result["university_sensitivity"].drop(
        columns=["Largest absolute change"]
    ).to_html(index=False, escape=True, border=0, float_format=lambda n: f"{n:+.3f}")
    top_recommendation = result["recommendations"].iloc[0]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>UniMatch AI applicant report</title>
<style>body{{font:16px/1.55 Arial,sans-serif;max-width:1000px;margin:40px auto;padding:0 20px;color:#182230}}
h1,h2{{color:#183b56}}table{{border-collapse:collapse;width:100%;margin:16px 0 32px}}th,td{{border:1px solid #d9e2ec;padding:9px;text-align:left}}th{{background:#edf4f8}}
.note{{padding:12px;background:#fff6dc;border-left:4px solid #e8ac2b}}</style></head>
<body><h1>UniMatch AI applicant report</h1><p>Generated {html.escape(generated)}</p>
<h2>Applicant profile</h2>{profile_table}
<h2>Graduate model results</h2><p>Selected ML model: {html.escape(result["graduate_ml_model"])} —
estimated probability of belonging to the high-chance class (historical threshold ≥ 0.70):
<strong>{result["graduate_ml_probability"]:.1%}</strong>. DNN estimate:
<strong>{result["graduate_dnn_probability"]:.1%}</strong>.</p>
<h2>Recommended universities</h2>{recommendations_html}
<h2>Graduate model local sensitivity</h2>{sensitivity_html}
<h2>Top recommendation sensitivity</h2><p>{html.escape(str(top_recommendation["University"]))}</p>
{university_sensitivity_html}
<p class="note">Predictions are historical-model estimates, not admission decisions or guarantees.
Sensitivity values are one-feature-at-a-time comparisons against training-set medians; they are not
causal effects. Ranking data is a 2026 snapshot. This report does not estimate tuition or affordability.</p>
</body></html>"""


def _prediction(profile: dict[str, Any], data: dict[str, Any], graduate_ml_name: str, university_ml_name: str, top_n: int) -> dict[str, Any]:
    graduate_features = graduate_feature_row(profile, data["graduate"])
    graduate_ml_model = load_ml_model("graduate", graduate_ml_name)
    graduate_ml_probability = _ml_probability(graduate_ml_model, "graduate", graduate_features)
    graduate_dnn_probability = _dnn_probability("graduate", graduate_features)
    recommendations = recommend_universities(profile, data, university_ml_name, top_n)
    sensitivity = _sensitivity(
        graduate_features,
        data["graduate"][["admit_chance_index", "toefl_score", "sop"]],
        "graduate",
        graduate_ml_name,
    )
    top_university_sensitivity = _sensitivity(
        recommendations.iloc[0]["_features"],
        data["university_features"],
        "university",
        university_ml_name,
    )
    return {
        "profile": profile.copy(),
        "graduate_ml_model": graduate_ml_name,
        "university_ml_model": university_ml_name,
        "top_n": top_n,
        "graduate_ml_probability": graduate_ml_probability,
        "graduate_dnn_probability": graduate_dnn_probability,
        "recommendations": recommendations,
        "sensitivity": sensitivity,
        "university_sensitivity": top_university_sensitivity,
        "graduate_features": graduate_features,
    }


def _default_model(metrics: pd.DataFrame) -> str:
    if "model" not in metrics or metrics.empty:
        raise ValueError("Model evaluation output is empty or lacks its model column.")
    return str(metrics.sort_values("roc_auc", ascending=False).iloc[0]["model"])


def _available_models(metrics: pd.DataFrame, dataset: str) -> tuple[list[str], str | None]:
    if "model" not in metrics or metrics.empty:
        raise ValueError(f"{dataset.title()} model evaluation output is empty or lacks its model column.")
    model_names = metrics["model"].astype(str).tolist()
    if "XGBoost" not in model_names:
        return model_names, None

    try:
        load_ml_model(dataset, "XGBoost")
    except XGBoostError as error:
        model_names.remove("XGBoost")
        warning = (
            f"The saved {dataset} XGBoost model could not be loaded with the installed XGBoost "
            f"version ({error}). It is disabled for this session; rerun Task 6 in this environment "
            "to regenerate a compatible model. Other models remain available."
        )
        return model_names, warning
    return model_names, None


def _show_image(path: Path, caption: str) -> None:
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")


def main() -> None:
    st.set_page_config(page_title="UniMatch AI", page_icon="🎓", layout="wide")
    st.markdown(
        """
        <style>
        :root {
            color-scheme: light;
            --primary-color: #2457c5;
            --background-color: #f5f7fb;
            --secondary-background-color: #ffffff;
            --text-color: #172b4d;
            --font: sans-serif;
        }
        html, body, .stApp, [data-testid="stAppViewContainer"] {
            background-color: #f5f7fb !important;
            color: #172b4d !important;
        }
        [data-testid="stAppViewContainer"] > .main {background: transparent;}
        [data-testid="stHeader"] {background: rgba(245, 247, 251, 0.9);}
        .block-container {padding: 2rem 2rem 3rem; max-width: 1280px;}
        h1, h2, h3, h4, h5, h6 {color: #172b4d;}
        [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
        [data-testid="stCaptionContainer"], label, [data-testid="stWidgetLabel"] {
            color: #35465d;
        }
        [data-testid="stSidebar"] {
            background: #edf2f8;
            border-right: 1px solid #dbe3ee;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: #172b4d;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dbe3ee;
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 2px 8px rgba(23, 43, 77, 0.04);
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {color: #172b4d;}
        [data-baseweb="input"], [data-baseweb="select"] > div,
        [data-baseweb="textarea"] {
            background-color: #ffffff;
            color: #172b4d;
            border-color: #cbd5e1;
        }
        input, textarea {color: #172b4d !important; background-color: #ffffff !important;}
        [data-baseweb="popover"] {color: #172b4d;}
        button[role="tab"] {color: #52627a;}
        button[role="tab"][aria-selected="true"] {
            color: #1947a8;
            border-bottom-color: #2457c5;
        }
        [data-testid="stForm"] {
            background: #ffffff;
            border: 1px solid #dbe3ee;
            border-radius: 14px;
            padding: 1.25rem;
        }
        [data-testid="stBaseButton-primary"] {background: #2457c5; border-color: #2457c5;}
        [data-testid="stBaseButton-secondary"] {
            color: #1947a8;
            border-color: #b9c8dd;
            background: #ffffff;
        }
        button:focus-visible, input:focus-visible, textarea:focus-visible {
            outline: 3px solid #7aa2f7;
            outline-offset: 2px;
        }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid #dbe3ee;
            border-radius: 10px;
            overflow: hidden;
        }
        @media (max-width: 640px) {
            .block-container {padding: 1.25rem 1rem 2rem;}
            [data-testid="stForm"] {padding: 0.9rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("🎓 UniMatch AI")
    st.caption("A model-backed graduate admissions and university recommendation workspace")
    try:
        data = load_data()
    except (FileNotFoundError, OSError, ValueError, KeyError, pd.errors.ParserError) as error:
        st.error(f"Required project data could not be loaded: {error}")
        st.info("Restore the repository's documented data and outputs artifacts, then reload the app.")
        st.stop()

    try:
        graduate_models, graduate_model_warning = _available_models(data["graduate_metrics"], "graduate")
        university_models, university_model_warning = _available_models(
            data["university_metrics"], "university"
        )
        graduate_available_metrics = data["graduate_metrics"][
            data["graduate_metrics"]["model"].astype(str).isin(graduate_models)
        ]
        university_available_metrics = data["university_metrics"][
            data["university_metrics"]["model"].astype(str).isin(university_models)
        ]
        grad_default = _default_model(graduate_available_metrics)
        university_default = _default_model(university_available_metrics)
    except (KeyError, ValueError) as error:
        st.error(f"Model evaluation metadata is invalid: {error}")
        st.stop()

    with st.sidebar:
        st.header("Prediction settings")
        if graduate_model_warning:
            st.warning(graduate_model_warning)
        if university_model_warning:
            st.warning(university_model_warning)
        graduate_ml_name = st.selectbox(
            "Graduate ML model",
            graduate_models,
            index=graduate_models.index(grad_default),
        )
        university_ml_name = st.selectbox(
            "University ML model",
            university_models,
            index=university_models.index(university_default),
        )
        top_n = st.slider("Recommendations to show", min_value=3, max_value=15, value=8)
        st.caption("DNN predictions are included alongside the selected ML models.")

    tabs = st.tabs(["Overview", "Applicant profile", "Recommendations", "Explainability", "Reports"])
    result = st.session_state.get("unimatch_result")

    with tabs[0]:
        st.subheader("A single workflow from profile to explanation")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Graduate records", f"{len(data['graduate']):,}")
        metric_cols[1].metric("University applicant records", f"{len(data['history']):,}")
        metric_cols[2].metric("Candidate universities", f"{data['history']['univname'].nunique():,}")
        metric_cols[3].metric("Ranked institutions", f"{len(data['rankings']):,}")
        st.write(
            "Enter a profile to validate score ranges, engineer the same selected features used in "
            "training, run the saved ML and deep-learning models, rank university matches, and export "
            "a privacy-conscious HTML report."
        )
        st.warning(
            "These are estimates from historical, self-reported datasets. They are not admission "
            "decisions, financial advice, or guarantees."
        )
        if result:
            if (
                result["graduate_ml_model"] != graduate_ml_name
                or result["university_ml_model"] != university_ml_name
                or result["top_n"] != top_n
            ):
                st.info("Settings changed since the last run. Submit the profile again to refresh these results.")
            st.caption(
                f"Current results use graduate {result['graduate_ml_model']} and university "
                f"{result['university_ml_model']}."
            )
            left, right = st.columns(2)
            left.metric(
                "Graduate ML estimate",
                f"{result['graduate_ml_probability']:.1%}",
                help="Estimated probability of the class labeled chance_of_admit >= 0.70 in the graduate training set.",
            )
            right.metric(
                "Graduate DNN estimate",
                f"{result['graduate_dnn_probability']:.1%}",
                help="Same historical high-chance classification target, predicted by the saved DNN.",
            )

    with tabs[1]:
        st.subheader("Applicant profile and prediction")
        invalid_categories = {"-", "0", "none", "nan", "n/a"}
        department_counts = data["history"]["department"].dropna().astype(str).value_counts()
        specialization_counts = data["history"]["specialization"].dropna().astype(str).value_counts()
        department_counts = department_counts[
            ~department_counts.index.str.strip().str.casefold().isin(invalid_categories)
        ]
        specialization_counts = specialization_counts[
            ~specialization_counts.index.str.strip().str.casefold().isin(invalid_categories)
        ]
        departments = sorted(
            department_counts.index.tolist()
        )
        specializations = sorted(specialization_counts.index.tolist())
        if not departments or not specializations:
            st.error("No valid department or specialization categories are available in the historical data.")
            st.stop()
        default_department = str(department_counts.idxmax())
        default_specialization = str(specialization_counts.idxmax())
        with st.form("applicant_profile"):
            st.markdown("#### Graduate admissions profile")
            col1, col2, col3 = st.columns(3)
            with col1:
                gre_score = st.number_input("GRE total", min_value=260.0, max_value=340.0, value=315.0, step=1.0)
                toefl_score = st.number_input("TOEFL", min_value=0.0, max_value=120.0, value=105.0, step=1.0)
                cgpa_scale = st.selectbox("CGPA scale", [4.0, 10.0, 100.0], index=1)
            with col2:
                cgpa = st.number_input("CGPA", min_value=0.0, max_value=float(cgpa_scale), value=min(8.5, float(cgpa_scale)), step=0.1)
                sop = st.number_input("Statement of Purpose strength", min_value=0.0, max_value=5.0, value=3.5, step=0.5)
                lor = st.number_input("Recommendation strength", min_value=0.0, max_value=5.0, value=3.5, step=0.5)
            with col3:
                research = st.checkbox("Research experience", value=True)
                rating = st.slider("Target-university rating (1–5)", 1, 5, 3)
                st.caption("Rating is retained in the profile but is not one of the selected graduate model features.")

            st.markdown("#### University matching profile")
            col1, col2, col3 = st.columns(3)
            with col1:
                grev = st.number_input("GRE verbal", min_value=130.0, max_value=170.0, value=155.0, step=1.0)
                greq = st.number_input("GRE quantitative", min_value=130.0, max_value=170.0, value=160.0, step=1.0)
                grea = st.number_input("GRE analytical writing", min_value=0.0, max_value=6.0, value=4.0, step=0.5)
            with col2:
                researchexp = st.number_input("Research experience (years)", min_value=0.0, max_value=50.0, value=1.0, step=0.5)
                industryexp = st.number_input("Industry experience (years)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
                toppercgpa = st.number_input("Undergraduate college topper CGPA", min_value=0.0, max_value=float(cgpa_scale), value=min(float(cgpa), float(cgpa_scale)), step=0.1)
            with col3:
                department = st.selectbox(
                    "Department (historical categories)",
                    departments,
                    index=departments.index(default_department) if default_department in departments else 0,
                )
                specialization = st.selectbox(
                    "Specialization (historical categories)",
                    specializations,
                    index=specializations.index(default_specialization) if default_specialization in specializations else 0,
                )
                st.caption("Categories are restricted to the training vocabulary so label encodings match the saved models.")
            submitted = st.form_submit_button("Validate profile and run predictions", type="primary")

        if submitted:
            profile = {
                "gre_score": gre_score,
                "toefl_score": toefl_score,
                "cgpa": cgpa,
                "cgpa_scale": cgpa_scale,
                "sop": sop,
                "lor": lor,
                "research": int(research),
                "university_rating": rating,
                "grev": grev,
                "greq": greq,
                "grea": grea,
                "toeflscore": toefl_score,
                "researchexp": researchexp,
                "industryexp": industryexp,
                "toppercgpa": toppercgpa,
                "department": department,
                "specialization": specialization,
            }
            errors = validate_profile(profile)
            if errors:
                st.error("Please correct the profile before running predictions.")
                for error in errors:
                    st.write(f"- {error}")
            else:
                try:
                    st.session_state["unimatch_result"] = _prediction(
                        profile, data, graduate_ml_name, university_ml_name, top_n
                    )
                    st.success("Profile validated; ML and DNN predictions and recommendations are ready.")
                    st.rerun()
                except (FileNotFoundError, OSError, ImportError, EOFError, ValueError, TypeError, RuntimeError, KeyError) as error:
                    st.error(f"Prediction could not be completed: {error}")

    with tabs[2]:
        st.subheader("University recommendations")
        if not result:
            st.info("Complete the applicant profile and run predictions to generate recommendations.")
        else:
            st.caption(
                "Recommendation score = 55% average ML/DNN admission estimate + 30% profile fit + "
                "15% ranking signal. Unranked universities receive no ranking-signal points. Tuition "
                "and affordability are not available in these source data."
            )
            visible = result["recommendations"].drop(columns=["_features"], errors="ignore").copy()
            for column in ("Estimated admission probability", "ML estimate", "DNN estimate", "Profile fit", "Recommendation score"):
                visible[column] = visible[column].map(lambda value: f"{value:.1%}")
            visible["Average rank"] = visible["Average rank"].map(
                lambda value: f"{float(value):.0f}" if _finite(value) else "Not ranked"
            )
            st.dataframe(visible, width="stretch", hide_index=True)
            st.download_button(
                "Download recommendations as CSV",
                result["recommendations"].drop(columns=["_features"], errors="ignore").to_csv(index=False).encode("utf-8"),
                file_name="unimatch_recommendations.csv",
                mime="text/csv",
            )

    with tabs[3]:
        st.subheader("Explainability and responsible use")
        if result:
            st.markdown(
                "The chart below compares each live prediction with a counterfactual input where one "
                "selected model feature at a time is replaced by its training-set median. It is a "
                "model-sensitivity aid, not a causal explanation."
            )
            st.dataframe(
                result["sensitivity"].drop(columns=["Largest absolute change"]),
                width="stretch",
                hide_index=True,
            )
            st.bar_chart(
                result["sensitivity"].set_index("Feature")[
                    ["ML probability change vs training median", "DNN probability change vs training median"]
                ]
            )
            st.markdown("#### Local explanation for a recommended university")
            st.caption("The top-ranked recommendation's model features are compared one at a time to the training-set medians.")
            st.dataframe(
                result["university_sensitivity"].drop(columns=["Largest absolute change"]),
                width="stretch",
                hide_index=True,
            )
            st.bar_chart(
                result["university_sensitivity"].set_index("Feature")[
                    ["ML probability change vs training median", "DNN probability change vs training median"]
                ]
            )
        else:
            st.info("Run predictions to see local, model-specific sensitivity for this applicant.")
        st.markdown("#### Existing Task 8 global and local explainability outputs")
        _show_image(TASK8 / "shap_summary_graduate.png", "Graduate DNN global SHAP summary")
        _show_image(TASK8 / "shap_summary_university.png", "University global SHAP summary")
        _show_image(TASK8 / "feature_importance_permutation.png", "Permutation feature importance")
        reports = data["reports"]
        for key, title in (("shap", "SHAP report"), ("lime", "LIME report"), ("fairness", "Fairness report"), ("trust", "Trust-score report")):
            if key in reports:
                with st.expander(title):
                    st.dataframe(reports[key], width="stretch", hide_index=True)
        _show_image(TASK8 / "shap_waterfall_graduate.png", "Graduate SHAP local waterfall")
        _show_image(TASK8 / "shap_waterfall_university.png", "University SHAP local waterfall")
        _show_image(TASK8 / "counterfactual_summary.png", "Counterfactual analysis summary")
        _show_image(TASK8 / "trust_score_vs_accuracy.png", "Trust score versus accuracy")
        for image_name in (
            "lime_graduate.png",
            "lime_university.png",
            "explainability_dashboard_graduate_row58.png",
            "explainability_dashboard_graduate_row82.png",
            "explainability_dashboard_university_row1158.png",
            "explainability_dashboard_university_row1462.png",
        ):
            _show_image(TASK8 / image_name, image_name.replace("_", " ").removesuffix(".png"))
        st.warning(
            "The repository's fairness results use proxy groups because protected attributes are not "
            "available. They are not a certification of fairness. University label-encoded features "
            "have no ordinal meaning."
        )

    with tabs[4]:
        st.subheader("Evaluation outputs and downloadable report")
        if result:
            st.download_button(
                "Download complete applicant report (HTML)",
                build_report_html(result),
                file_name="unimatch_applicant_report.html",
                mime="text/html",
                type="primary",
            )
        else:
            st.info("Run predictions to enable a personalized HTML report.")

        with st.expander("Task 6 machine-learning evaluation"):
            left, right = st.columns(2)
            left.markdown("**Graduate models**")
            left.dataframe(data["graduate_metrics"], width="stretch", hide_index=True)
            right.markdown("**University models**")
            right.dataframe(data["university_metrics"], width="stretch", hide_index=True)
            _show_image(TASK6 / "roc_curves_graduate.png", "Graduate ML ROC curves")
            _show_image(TASK6 / "roc_curves_university.png", "University ML ROC curves")
            _show_image(TASK6 / "confusion_matrices_graduate.png", "Graduate ML confusion matrices")
            _show_image(TASK6 / "confusion_matrices_university.png", "University ML confusion matrices")

        with st.expander("Task 7 deep-learning and comparison outputs"):
            for key, label in (
                ("dnn_evaluation", "DNN evaluation"),
                ("dnn_comparison", "DNN versus ML"),
                ("dnn_hyperparameters", "DNN hyperparameters"),
                ("dnn_overfitting", "Overfitting checks"),
                ("graduate_training_history", "Graduate training history"),
                ("university_training_history", "University training history"),
            ):
                if key in data["reports"]:
                    st.markdown(f"**{label}**")
                    st.dataframe(data["reports"][key], width="stretch", hide_index=True)
            _show_image(TASK7 / "accuracy_loss_curves.png", "DNN training and validation curves")
            _show_image(TASK7 / "dnn_roc_confusion.png", "DNN ROC and confusion matrices")

        with st.expander("Task 5 feature importance"):
            for key, label in (("graduate_importance", "Graduate"), ("university_importance", "University")):
                if key in data["reports"]:
                    st.markdown(f"**{label} feature importance**")
                    st.dataframe(data["reports"][key], width="stretch", hide_index=True)
            for key, label in (
                ("graduate_correlation", "Graduate correlation matrix"),
                ("university_correlation", "University correlation matrix"),
            ):
                if key in data["reports"]:
                    st.markdown(f"**{label}**")
                    st.dataframe(data["reports"][key], width="stretch")
            if "permutation_importance" in data["reports"]:
                st.markdown("**Task 8 permutation importance**")
                st.dataframe(data["reports"]["permutation_importance"], width="stretch", hide_index=True)

        with st.expander("Task 8 fairness, trust and counterfactual tables"):
            for key, label in (
                ("fairness_by_group", "Fairness metrics by proxy group"),
                ("trust_scores", "Trust scores"),
                ("counterfactuals", "Counterfactual analysis"),
            ):
                if key in data["reports"]:
                    st.markdown(f"**{label}**")
                    st.dataframe(data["reports"][key], width="stretch", hide_index=True)

        st.markdown("#### Saved LIME case studies")
        lime_files = (
            "lime_graduate_confident_admit.html",
            "lime_graduate_borderline.html",
            "lime_graduate_confident_reject.html",
            "lime_university_confident_admit.html",
            "lime_university_borderline.html",
            "lime_university_confident_reject.html",
        )
        lime_columns = st.columns(3)
        for index, filename in enumerate(lime_files):
            path = TASK8 / filename
            if path.is_file():
                lime_columns[index % 3].download_button(
                    filename.replace("_", " ").replace(".html", ""),
                    path.read_bytes(),
                    file_name=filename,
                    mime="text/html",
                    key=f"download_{filename}",
                )

        st.caption(
            "Personalized reports contain only the submitted profile fields and model outputs; "
            "historical usernames and profile links are not loaded into or exported by this app."
        )


if __name__ == "__main__":
    main()
