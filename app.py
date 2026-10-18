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

# Plain-language names for model features shown to applicants.
FEATURE_LABELS = {
    "gre_score": "GRE total",
    "toefl_score": "TOEFL",
    "toeflscore": "TOEFL",
    "university_rating": "Target-university rating",
    "sop": "Statement of purpose",
    "lor": "Recommendation letters",
    "cgpa": "CGPA (out of 10)",
    "cgpa_pct": "CGPA (%)",
    "research": "Research experience",
    "researchexp": "Research experience (years)",
    "industryexp": "Industry experience (years)",
    "grev": "GRE verbal",
    "greq": "GRE quantitative",
    "grea": "GRE analytical writing",
    "toppercgpa": "Class-topper CGPA",
    "admit_chance_index": "Overall academic strength",
    "fit_score": "Fit with past admits",
    "profile_distance": "Distance from past admits",
    "department_freq": "How common your department is",
    "specialization_freq": "How common your specialization is",
}


def feature_label(column: str) -> str:
    return FEATURE_LABELS.get(column, column.replace("_", " ").capitalize())


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required project artifact was not found: {path}")
    return pd.read_csv(path, **kwargs)


@st.cache_data(show_spinner="Loading project data and model reports...")
def load_data() -> dict[str, Any]:
    """Load only the historical columns required by the live inference workflow."""
    feature_columns_path = TASK6 / "models" / "feature_columns.json"
    dnn_config_path = TASK7 / "dnn_config.json"
    encodings_path = TASK5 / "category_encodings.json"
    for path in (feature_columns_path, dnn_config_path, encodings_path):
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
    with encodings_path.open(encoding="utf-8") as file:
        encodings = json.load(file)
    university_features = _read_csv(
        TASK5 / "feature_matrix_university.csv",
        usecols=[*feature_columns["university"], "split"],
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
        "encodings": encodings,
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

    if not profile.get("program"):
        errors.append("Choose a program from the available historical categories.")
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


def graduate_feature_row(profile: dict[str, Any]) -> pd.DataFrame:
    """Raw graduate attributes; the saved feature list decides which of them the models use."""
    return pd.DataFrame(
        [
            {
                "gre_score": float(profile["gre_score"]),
                "toefl_score": float(profile["toefl_score"]),
                "university_rating": float(profile["university_rating"]),
                "sop": float(profile["sop"]),
                "lor": float(profile["lor"]),
                # The graduate training data records CGPA out of 10.
                "cgpa": float(profile["cgpa"]) / float(profile["cgpa_scale"]) * 10.0,
                "research": float(profile["research"]),
            }
        ]
    )


def onehot_columns(encodings: dict[str, Any]) -> set[str]:
    return {
        column
        for attribute in ("program", "univname", "department", "specialization")
        for column in encodings[attribute]["columns"].values()
    }


def _encode_categories(encodings: dict[str, Any], values: dict[str, str]) -> dict[str, float]:
    """Rebuild Task 5's one-hot and frequency columns. Levels outside the training vocabulary
    (e.g. a department with fewer than the minimum training rows) get all-zero indicators."""
    row: dict[str, float] = {}
    for attribute, value in values.items():
        mapping = encodings[attribute]["columns"]
        for level, column in mapping.items():
            row[column] = 1.0 if level == str(value) else 0.0
        if "frequency" in encodings[attribute]:
            row[f"{attribute}_freq"] = float(encodings[attribute]["frequency"].get(str(value), 0.0))
    return row


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
    encodings: dict[str, Any],
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
        **values,
        "admit_chance_index": chance_index,
        "profile_distance": distance,
        "fit_score": fit_score,
        "grea": float(profile["grea"]),
        "toppercgpa": float(profile["toppercgpa"]),
        **_encode_categories(
            encodings,
            {
                "program": profile["program"],
                "univname": university,
                "department": profile["department"],
                "specialization": profile["specialization"],
            },
        ),
    }
    return pd.DataFrame([row]), fit_score


def _ml_probability(
    model: Any,
    dataset: str,
    feature_row: pd.DataFrame,
) -> float:
    expected_columns = load_data()["feature_columns"][dataset]
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
        feature_row, fit_score = university_feature_row(profile, university, history, data["encodings"])
        ml_probability = _ml_probability(ml_model, "university", feature_row)
        dnn_probability = _dnn_probability("university", feature_row)
        match = ranking_info.get(re.sub(r"[^a-z0-9]", "", university.lower()), {})
        rank = match.get("avg_rank_all3")
        ranking_score = 1.0 / (1.0 + float(rank) / 100.0) if _finite(rank) else 0.0
        chance = (ml_probability + dnn_probability) / 2.0
        results.append(
            {
                "University": university,
                "Category": admission_category(chance),
                "Admission chance": chance,
                "Match score": 0.55 * chance + 0.30 * fit_score + 0.15 * ranking_score,
                "Profile fit": fit_score,
                "World rank (avg.)": rank,
                "Country": match.get("country", "Not listed"),
                "ML estimate": ml_probability,
                "Neural network estimate": dnn_probability,
                "_features": feature_row,
            }
        )
    table = pd.DataFrame(results).sort_values("Match score", ascending=False).head(top_n)
    return table.reset_index(drop=True)


def admission_category(chance: float) -> str:
    if chance >= 0.70:
        return "Safe"
    if chance >= 0.40:
        return "Target"
    return "Reach"


def _sensitivity(
    feature_row: pd.DataFrame,
    training_features: pd.DataFrame,
    dataset: str,
    ml_model_name: str,
    skip: set[str] = frozenset(),
) -> pd.DataFrame:
    ml_model = load_ml_model(dataset, ml_model_name)
    current_ml = _ml_probability(ml_model, dataset, feature_row)
    current_dnn = _dnn_probability(dataset, feature_row)
    rows: list[dict[str, Any]] = []
    baselines = training_features.median(numeric_only=True)
    for column in feature_row.columns:
        # One-hot category indicators have no meaningful "median" replacement, so they are skipped.
        if column in skip or column not in baselines.index or pd.isna(baselines[column]):
            continue
        changed = feature_row.copy()
        changed.loc[:, column] = float(baselines[column])
        ml_change = current_ml - _ml_probability(ml_model, dataset, changed)
        dnn_change = current_dnn - _dnn_probability(dataset, changed)
        rows.append(
            {
                "Feature": column,
                "Your value": float(feature_row[column].iloc[0]),
                "Typical applicant": float(baselines[column]),
                "ML probability change vs training median": ml_change,
                "DNN probability change vs training median": dnn_change,
                "Average change": (ml_change + dnn_change) / 2.0,
                "Largest absolute change": max(abs(ml_change), abs(dnn_change)),
            }
        )
    return pd.DataFrame(rows).sort_values("Largest absolute change", ascending=False).reset_index(drop=True)


def build_report_html(result: dict[str, Any]) -> str:
    profile_table = pd.DataFrame([result["profile"]]).to_html(index=False, escape=True, border=0)
    recommendations = result["recommendations"].drop(columns=["_features"], errors="ignore")
    recommendations_html = recommendations.to_html(index=False, escape=True, border=0, float_format=lambda n: f"{n:.3f}")
    def sensitivity_table(table: pd.DataFrame) -> str:
        return (
            table.drop(columns=["Largest absolute change"])
            .assign(Feature=lambda frame: frame["Feature"].map(feature_label))
            .to_html(index=False, escape=True, border=0, float_format=lambda n: f"{n:+.3f}")
        )

    sensitivity_html = sensitivity_table(result["sensitivity"])
    university_sensitivity_html = sensitivity_table(result["university_sensitivity"])
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
    graduate_columns = data["feature_columns"]["graduate"]
    graduate_features = graduate_feature_row(profile)[graduate_columns]
    graduate_ml_model = load_ml_model("graduate", graduate_ml_name)
    graduate_ml_probability = _ml_probability(graduate_ml_model, "graduate", graduate_features)
    graduate_dnn_probability = _dnn_probability("graduate", graduate_features)
    recommendations = recommend_universities(profile, data, university_ml_name, top_n)
    graduate_training = data["graduate"][data["graduate"]["split"] == "train"]
    sensitivity = _sensitivity(
        graduate_features,
        graduate_training[graduate_columns],
        "graduate",
        graduate_ml_name,
    )
    top_university_sensitivity = university_sensitivity(
        recommendations.iloc[0]["_features"], data, university_ml_name
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


def university_sensitivity(feature_row: pd.DataFrame, data: dict[str, Any], ml_model_name: str) -> pd.DataFrame:
    training = data["university_features"][data["university_features"]["split"] == "train"]
    return _sensitivity(
        feature_row,
        training[data["feature_columns"]["university"]],
        "university",
        ml_model_name,
        skip=onehot_columns(data["encodings"]),
    )


def _default_model(metrics: pd.DataFrame) -> str:
    if "model" not in metrics or metrics.empty:
        raise ValueError("Model evaluation output is empty or lacks its model column.")
    # Task 6 selects models on the validation split; roc_auc is the held-out test score.
    criterion = "val_roc_auc" if "val_roc_auc" in metrics else "roc_auc"
    return str(metrics.sort_values(criterion, ascending=False).iloc[0]["model"])


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
        st.image(str(path), caption=caption, width="stretch", alt=caption)


TAB_LABELS = (
    ":material/edit_note: Your profile",
    ":material/school: Your matches",
    ":material/lightbulb: Why these results",
    ":material/analytics: Model details",
)
CGPA_DEFAULTS = {4.0: (3.4, 3.9), 10.0: (8.5, 9.5), 100.0: (85.0, 95.0)}


def _format_value(value: float) -> str:
    return f"{value:.4g}"


def outlook_label(chance: float) -> str:
    if chance >= 0.70:
        return "Strong"
    if chance >= 0.40:
        return "Competitive"
    return "Developing"


def render_drivers(sensitivity: pd.DataFrame, subject: str) -> None:
    """Explain one prediction as 'working in your favour' / 'holding you back' lists plus a chart."""
    table = sensitivity.assign(
        Factor=sensitivity["Feature"].map(feature_label),
        Effect=sensitivity["Average change"] * 100.0,
    ).sort_values("Effect", key=lambda column: column.abs(), ascending=False)

    helping = table[table["Effect"] >= 1.0]
    hurting = table[table["Effect"] <= -1.0]
    left, right = st.columns(2)
    for container, rows, title, colour, sign in (
        (left, helping, ":material/trending_up: Working in your favour", "green", "+"),
        (right, hurting, ":material/trending_down: Holding you back", "red", ""),
    ):
        with container.container(border=True, height="stretch"):
            st.markdown(f"**{title}**")
            if rows.empty:
                st.caption("Nothing here changes the estimate by a full percentage point.")
            for _, row in rows.iterrows():
                st.markdown(
                    f"- **{row['Factor']}**: yours {_format_value(row['Your value'])}, typical "
                    f"{_format_value(row['Typical applicant'])} → :{colour}[{sign}{row['Effect']:.0f} points]"
                )

    import altair as alt

    chart = (
        alt.Chart(table)
        .mark_bar()
        .encode(
            x=alt.X("Effect:Q", title="Change in estimated chance (percentage points)"),
            y=alt.Y("Factor:N", sort=table["Factor"].tolist(), title=None),
            color=alt.condition("datum.Effect >= 0", alt.value("#2E8B57"), alt.value("#D1495B")),
            tooltip=[
                alt.Tooltip("Factor:N"),
                alt.Tooltip("Your value:Q", format=".4~g"),
                alt.Tooltip("Typical applicant:Q", format=".4~g"),
                alt.Tooltip("Effect:Q", title="Change (points)", format="+.1f"),
            ],
        )
    )
    st.altair_chart(chart, width="stretch", alt=f"How each factor changes the estimated chance for {subject}")


def render_profile_tab(data: dict[str, Any], graduate_ml_name: str, university_ml_name: str, top_n: int) -> None:
    result = st.session_state.get("unimatch_result")
    if result:
        st.success(
            "Open the **Your matches** tab to see them, or edit your profile below and run it again.",
            title="Your matches are ready",
            icon=":material/check_circle:",
        )

    invalid_categories = {"-", "0", "none", "nan", "n/a"}
    department_counts = data["history"]["department"].dropna().astype(str).value_counts()
    specialization_counts = data["history"]["specialization"].dropna().astype(str).value_counts()
    department_counts = department_counts[~department_counts.index.str.strip().str.casefold().isin(invalid_categories)]
    specialization_counts = specialization_counts[
        ~specialization_counts.index.str.strip().str.casefold().isin(invalid_categories)
    ]
    departments = sorted(department_counts.index.tolist())
    specializations = sorted(specialization_counts.index.tolist())
    if not departments or not specializations:
        st.error("No valid department or specialization categories are available in the historical data.")
        st.stop()
    programs = sorted(data["encodings"]["program"]["columns"])
    graduate_features = data["feature_columns"]["graduate"]

    st.markdown("Fill in what you know. Every field has a **?** with a short explanation.")

    # Outside the form so the CGPA limits below update as soon as the scale changes.
    cgpa_scale = st.segmented_control(
        "Your university's grading scale",
        [4.0, 10.0, 100.0],
        default=10.0,
        required=True,
        format_func=lambda scale: f"Out of {scale:g}",
        key="cgpa_scale",
        help="Pick the scale your CGPA is reported on. It is converted automatically for the models.",
    )
    cgpa_default, topper_default = CGPA_DEFAULTS[float(cgpa_scale)]

    with st.form("applicant_profile", border=False):
        with st.container(border=True):
            st.markdown("##### :material/school: Academic record")
            col1, col2 = st.columns(2)
            cgpa = col1.number_input(
                "Your CGPA",
                min_value=0.0,
                max_value=float(cgpa_scale),
                value=cgpa_default,
                step=0.1 if cgpa_scale < 100 else 1.0,
                help="Your cumulative grade point average on the scale chosen above.",
            )
            toppercgpa = col2.number_input(
                "Highest CGPA in your class",
                min_value=0.0,
                max_value=float(cgpa_scale),
                value=topper_default,
                step=0.1 if cgpa_scale < 100 else 1.0,
                help="The class topper's CGPA in your undergraduate program. If you don't know it, a close guess is fine.",
            )

        with st.container(border=True):
            st.markdown("##### :material/quiz: Test scores")
            col1, col2, col3, col4 = st.columns(4)
            grev = col1.number_input("GRE verbal", 130, 170, 155, help="GRE Verbal Reasoning, 130–170.")
            greq = col2.number_input("GRE quantitative", 130, 170, 160, help="GRE Quantitative Reasoning, 130–170.")
            grea = col3.number_input(
                "GRE writing", 0.0, 6.0, 4.0, step=0.5, help="GRE Analytical Writing, 0–6 in half-point steps."
            )
            toefl_score = col4.number_input("TOEFL", 0, 120, 105, help="TOEFL iBT total score, 0–120.")
            st.caption("Your GRE total is calculated for you as verbal + quantitative.")

        with st.container(border=True):
            st.markdown("##### :material/description: Application documents")
            col1, col2 = st.columns(2)
            sop = col1.slider(
                "Statement of purpose strength",
                1.0,
                5.0,
                3.5,
                step=0.5,
                help="Your honest self-rating: 1 = weak, 3 = average, 5 = outstanding.",
            )
            lor = col2.slider(
                "Recommendation letters strength",
                1.0,
                5.0,
                3.5,
                step=0.5,
                help="How strong you expect your letters to be: 1 = weak, 3 = average, 5 = outstanding.",
            )

        with st.container(border=True):
            st.markdown("##### :material/work: Experience")
            col1, col2 = st.columns(2)
            researchexp = col1.number_input(
                "Research experience (years)",
                0.0,
                50.0,
                1.0,
                step=0.5,
                help="Time spent on research projects, theses or lab work. Use 0 if none.",
            )
            industryexp = col2.number_input(
                "Work experience (years)",
                0.0,
                50.0,
                0.0,
                step=0.5,
                help="Full-time industry experience. Internships can be counted as a fraction of a year.",
            )

        with st.container(border=True):
            st.markdown("##### :material/menu_book: What you want to study")
            program = st.segmented_control(
                "Program",
                programs,
                default="MS" if "MS" in programs else programs[0],
                required=True,
                help="The degree you are applying for.",
            )
            col1, col2 = st.columns(2)
            department = col1.selectbox(
                "Department",
                departments,
                index=departments.index(str(department_counts.idxmax())),
                help="Start typing to search. These are the departments past applicants wrote down.",
            )
            specialization = col2.selectbox(
                "Specialization",
                specializations,
                index=specializations.index(str(specialization_counts.idxmax())),
                help="Start typing to search. Choose the closest match to your intended area.",
            )
            rating = 3
            if "university_rating" in graduate_features:
                rating = st.slider(
                    "How selective are your target universities?",
                    1,
                    5,
                    3,
                    help="1 = less selective, 5 = most selective.",
                )

        submitted = st.form_submit_button(
            "Find my matches", type="primary", icon=":material/search:", width="stretch"
        )

    if not submitted:
        return
    profile = {
        "gre_score": float(grev + greq),
        "toefl_score": float(toefl_score),
        "cgpa": cgpa,
        "cgpa_scale": float(cgpa_scale),
        "sop": sop,
        "lor": lor,
        "research": int(researchexp > 0),
        "university_rating": rating,
        "grev": float(grev),
        "greq": float(greq),
        "grea": grea,
        "toeflscore": float(toefl_score),
        "researchexp": researchexp,
        "industryexp": industryexp,
        "toppercgpa": toppercgpa,
        "program": program,
        "department": department,
        "specialization": specialization,
    }
    errors = validate_profile(profile)
    if errors:
        st.error("\n".join(f"- {error}" for error in errors), title="Please correct your profile")
        return
    try:
        with st.spinner("Scoring 54 universities against your profile..."):
            st.session_state["unimatch_result"] = _prediction(
                profile, data, graduate_ml_name, university_ml_name, top_n
            )
    except (FileNotFoundError, OSError, ImportError, EOFError, ValueError, TypeError, RuntimeError, KeyError) as error:
        st.error(str(error), title="Prediction could not be completed")
        return
    st.session_state["show_results"] = True
    st.rerun()


def render_matches_tab(result: dict[str, Any] | None, settings_changed: bool) -> None:
    if not result:
        st.info(
            "Fill in the **Your profile** tab and press **Find my matches**.",
            title="No results yet",
            icon=":material/edit_note:",
        )
        return
    if settings_changed:
        st.info("You changed a setting in the sidebar. Run your profile again to update these results.")

    recommendations = result["recommendations"]
    outlook = (result["graduate_ml_probability"] + result["graduate_dnn_probability"]) / 2.0
    counts = recommendations["Category"].value_counts()

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Overall admission outlook",
        f"{outlook_label(outlook)} · {outlook:.0%}",
        border=True,
        help="How likely your profile is to belong to the strong group of past applicants "
        "(those with at least a 70% admission chance). Average of two models.",
    )
    col2.metric(
        "Your top match",
        str(recommendations.iloc[0]["University"]),
        border=True,
        help="The university with the highest match score below.",
    )
    col3.metric(
        "Safe · Target · Reach",
        f"{counts.get('Safe', 0)} · {counts.get('Target', 0)} · {counts.get('Reach', 0)}",
        border=True,
        help="How many of your matches fall into each category (see the explanation below the table).",
    )

    st.markdown(f"#### Your top {len(recommendations)} university matches")
    st.caption("Sorted by match score. Hover over a column title for a short explanation.")
    visible = recommendations.drop(columns=["_features", "ML estimate", "Neural network estimate"])
    st.dataframe(
        visible,
        hide_index=True,
        column_config={
            "University": st.column_config.TextColumn("University", pinned=True),
            "Category": st.column_config.TextColumn(
                "Category", help="Safe ≥ 70% admission chance, Target 40–70%, Reach < 40%."
            ),
            "Admission chance": st.column_config.ProgressColumn(
                "Admission chance",
                help="Estimated chance of being admitted to this university, based on past applicants.",
                format="percent",
                min_value=0.0,
                max_value=1.0,
            ),
            "Match score": st.column_config.ProgressColumn(
                "Match score",
                help="55% admission chance + 30% profile fit + 15% world ranking.",
                format="percent",
                min_value=0.0,
                max_value=1.0,
            ),
            "Profile fit": st.column_config.ProgressColumn(
                "Profile fit",
                help="How close your CGPA, GRE and TOEFL are to students this university admitted before.",
                format="percent",
                min_value=0.0,
                max_value=1.0,
            ),
            "World rank (avg.)": st.column_config.NumberColumn(
                "World rank (avg.)",
                help="Average of the QS, THE and ARWU 2026 ranks. Empty if the university is not in the rankings table.",
                format="%.0f",
            ),
            "Country": st.column_config.TextColumn("Country"),
        },
        alt="Recommended universities with admission chance, match score, profile fit and ranking",
    )

    with st.expander("How to read this table", icon=":material/help:", expanded=True):
        st.markdown(
            """
| Column | What it means |
|---|---|
| **University** | A university where at least 20 past applicants reported a decision, so the estimate has data behind it. |
| **Category** | **Safe**: admission chance of 70% or more. **Target**: 40–70%, a realistic fit. **Reach**: under 40%, ambitious but possible. A balanced list usually has a few of each. |
| **Admission chance** | The estimated chance that you would be admitted to this university. It comes from about 53,000 past applications with known outcomes, and is the average of two different models (see below). |
| **Match score** | The single number used to sort the list: 55% admission chance + 30% profile fit + 15% world ranking. A high score means a university that is both reachable and a good fit. |
| **Profile fit** | How similar your CGPA, GRE verbal, GRE quantitative and TOEFL are to the students this university admitted in the past. 100% would mean you match their typical admitted student exactly. |
| **World rank (avg.)** | The average of the university's QS, THE and ARWU 2026 world ranks. Empty when it is not one of the 57 universities in the rankings table; those get no ranking points in the match score. |
| **Country** | Where the university is, from the rankings table ("Not listed" when it is not ranked). |
"""
        )
        st.caption(
            "Estimates come from historical, self-reported data. They are not admission decisions or "
            "guarantees, and they do not account for tuition or funding."
        )

    with st.expander("See each model's estimate", icon=":material/compare_arrows:"):
        st.markdown(
            "Two different models score every university: a classic machine-learning model "
            f"(**{result['university_ml_model']}**) and a **neural network**. The admission chance above "
            "is their average. When they disagree by a lot, treat that university's estimate with more caution."
        )
        st.dataframe(
            recommendations[["University", "ML estimate", "Neural network estimate"]],
            hide_index=True,
            column_config={
                column: st.column_config.ProgressColumn(column, format="percent", min_value=0.0, max_value=1.0)
                for column in ("ML estimate", "Neural network estimate")
            },
            alt="Machine-learning and neural-network admission estimates per university",
        )

    with st.container(horizontal=True):
        st.download_button(
            "Download full report (HTML)",
            build_report_html(result),
            file_name="unimatch_applicant_report.html",
            mime="text/html",
            type="primary",
            icon=":material/download:",
        )
        st.download_button(
            "Download table (CSV)",
            recommendations.drop(columns=["_features"]).to_csv(index=False).encode("utf-8"),
            file_name="unimatch_recommendations.csv",
            mime="text/csv",
            icon=":material/table_view:",
        )


def render_why_tab(result: dict[str, Any] | None, data: dict[str, Any]) -> None:
    st.markdown(
        "This tab explains **why** you got your results: which parts of your profile push your "
        "estimates up or down, and what the models learned from past applicants in general."
    )
    with st.expander("How are these explanations calculated?", icon=":material/help:"):
        st.markdown(
            """
For each factor, we take your profile and swap **just that one value** for the value of a
*typical applicant* (the median of past applicants). Then we ask both models again.

- If your estimate **drops** when your value is replaced, your value was **working in your favour**.
- If it **rises**, your value was **holding you back**.

The size is shown in **percentage points**: "+5 points" means your value moves your estimated chance
from, say, 60% to 65% compared with an average applicant. Effects are averaged over the
machine-learning model and the neural network.

This shows what the *models* respond to. It is not proof that changing one thing will change an
admission decision, and factors can interact.
"""
        )

    if not result:
        st.info(
            "Run your profile first to get explanations about **you**. General insights are shown below.",
            title="No personal results yet",
            icon=":material/edit_note:",
        )
    else:
        st.markdown("#### What drives your overall admission outlook")
        st.caption(
            "Based on the graduate-admissions model, which compares your profile with 500 past applicants."
        )
        render_drivers(result["sensitivity"], "your overall outlook")

        st.markdown("#### Why a university scored the way it did")
        recommendations = result["recommendations"]
        names = recommendations["University"].tolist()
        chosen = st.selectbox(
            "Explain this university",
            names,
            help="Pick any of your matches to see what drives its admission estimate.",
        )
        row = recommendations[recommendations["University"] == chosen].iloc[0]
        st.caption(
            f"Estimated admission chance at {chosen}: **{row['Admission chance']:.0%}** "
            f"({row['Category']}), profile fit **{row['Profile fit']:.0%}**."
        )
        if chosen == names[0]:
            sensitivity = result["university_sensitivity"]
        else:
            with st.spinner(f"Explaining {chosen}..."):
                sensitivity = university_sensitivity(row["_features"], data, result["university_ml_model"])
        render_drivers(sensitivity, chosen)
        st.caption(
            "*Overall academic strength* is a combined score of your CGPA, GRE, TOEFL and experience. "
            "*Fit with past admits* compares you with students this university admitted before."
        )

    st.divider()
    st.markdown("#### What the models learned from past applicants")
    st.caption("These charts describe the models in general, not your profile.")
    reports = data["reports"]

    with st.expander("Which factors matter most overall", icon=":material/leaderboard:"):
        st.markdown(
            """
**SHAP summary.** Each dot is one past applicant. A dot far to the **right** means that factor pushed
that applicant's predicted chance **up**; far to the **left** means it pushed it **down**. The colour
shows whether the applicant's value was high (red) or low (blue). Factors at the top matter most.
"""
        )
        _show_image(TASK8 / "shap_summary_graduate.png", "Overall outlook models: what matters most")
        _show_image(TASK8 / "shap_summary_university.png", "University models: what matters most")
        st.markdown(
            "**Permutation importance.** How much worse the models get when one factor is shuffled at "
            "random. A longer bar means the model relies on that factor more."
        )
        _show_image(TASK8 / "feature_importance_permutation.png", "How much each model relies on each factor")

    with st.expander("Example explanations for single applicants", icon=":material/person_search:"):
        st.markdown(
            """
These show how one prediction is built up, factor by factor, for a few past applicants.

- **Waterfall (SHAP):** starts from the average prediction and adds each factor's push up (red) or
  down (blue) until it reaches that applicant's prediction.
- **LIME:** fits a simple rule-based model around one applicant. Green bars push towards *admit*,
  red bars towards *reject*. Shown for a confident admit, a borderline case and a confident reject.
"""
        )
        _show_image(TASK8 / "shap_waterfall_graduate.png", "Waterfall for one past applicant (overall outlook)")
        _show_image(TASK8 / "shap_waterfall_university.png", "Waterfall for one past applicant (university model)")
        _show_image(TASK8 / "lime_graduate.png", "LIME explanations (overall outlook)")
        _show_image(TASK8 / "lime_university.png", "LIME explanations (university model)")
        for path in sorted(TASK8.glob("explainability_dashboard_*.png")):
            dataset = "overall outlook" if "graduate" in path.name else "university model"
            _show_image(path, f"Full explanation dashboard for one past applicant ({dataset})")

    with st.expander("What would it take to improve?", icon=":material/rocket_launch:"):
        st.markdown(
            "For past applicants the model predicted would **not** reach the strong group, we searched "
            "for the smallest realistic improvements (for example +0.1 CGPA or +2 GRE points at a time) "
            "that would change the prediction."
        )
        counterfactuals = reports.get("counterfactuals")
        if counterfactuals is not None and not counterfactuals.empty:
            found = counterfactuals[counterfactuals["p_after_ml"].notna()]
            needed = found["changes"].str.extractall(r"(\w+) \+")[0].map(feature_label).value_counts()
            summary = ", ".join(f"{name} ({count})" for name, count in needed.head(3).items())
            st.markdown(
                f"A realistic change was found for **{len(found)} of {len(counterfactuals)}** applicants. "
                f"The improvements needed most often: {summary}."
            )
        _show_image(TASK8 / "counterfactual_summary.png", "Which improvements were needed most often")

    with st.expander("How much to trust a prediction", icon=":material/verified:"):
        st.markdown(
            """
Every prediction gets a **trust score** from three checks:
how sure the model is, whether the two models agree, and how similar the applicant is to the people
the models learned from. The chart checks that it works: predictions with **high** trust should be
right more often than predictions with **low** trust.
"""
        )
        _show_image(TASK8 / "trust_score_vs_accuracy.png", "Accuracy of predictions by trust level")
        if "trust" in reports:
            st.dataframe(reports["trust"], hide_index=True, alt="Accuracy by trust band")

    with st.expander("Fairness checks", icon=":material/balance:"):
        st.markdown(
            """
We checked whether the models treat groups of applicants differently. The datasets have no gender,
ethnicity or nationality, so we compare the groups that exist: research vs no research, program
(MS vs PhD) and ranked vs unranked universities.

- **Demographic parity difference:** the gap in how often each group is predicted to be admitted.
- **Equal opportunity difference:** the gap in how often truly admitted applicants are recognised.
- **Base rate gap:** the gap in how often each group was *actually* admitted. A model gap close to
  this one mostly reflects real differences in the data rather than bias.

Gaps above 0.10 are flagged **review**. This is a first check, not a fairness certification.
"""
        )
        if "fairness" in reports:
            st.dataframe(reports["fairness"], hide_index=True, alt="Fairness summary by group")

    st.warning(
        "These are estimates from historical, self-reported data. Use them to plan a balanced list, "
        "not as the only basis for decisions.",
        icon=":material/info:",
    )


def render_details_tab(data: dict[str, Any]) -> None:
    st.markdown(
        "Technical results behind the app, for readers who want to check how well the models work."
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("Graduate records", f"{len(data['graduate']):,}", border=True)
    metric_cols[1].metric("University applications", f"{len(data['history']):,}", border=True)
    metric_cols[2].metric("Candidate universities", f"{data['history']['univname'].nunique():,}", border=True)
    metric_cols[3].metric("Ranked institutions", f"{len(data['rankings']):,}", border=True)

    with st.expander("Glossary of the metrics", icon=":material/menu_book:"):
        st.markdown(
            """
- **ROC-AUC:** how well a model separates admitted from rejected applicants (1.0 = perfect, 0.5 = guessing).
- **val_roc_auc:** ROC-AUC on the *validation* set, used to choose the best model.
- **Accuracy / precision / recall / F1:** measured on the *test* set, which is never used for any choice.
  Precision = of the predicted admits, how many were admitted; recall = of the real admits, how many were found.
- **Minority-class columns:** the same measures for the smaller class, to check it is not ignored.
- The university data is split **by applicant**, so the test set only contains people the models never saw.
"""
        )

    reports = data["reports"]
    with st.expander("Machine-learning models (Task 6)", icon=":material/model_training:"):
        left, right = st.columns(2)
        left.markdown("**Overall outlook (graduate) models**")
        left.dataframe(data["graduate_metrics"], hide_index=True, alt="Graduate model evaluation")
        right.markdown("**University models**")
        right.dataframe(data["university_metrics"], hide_index=True, alt="University model evaluation")
        _show_image(TASK6 / "roc_curves_graduate.png", "Graduate ML ROC curves")
        _show_image(TASK6 / "roc_curves_university.png", "University ML ROC curves")
        _show_image(TASK6 / "confusion_matrices_graduate.png", "Graduate ML confusion matrices")
        _show_image(TASK6 / "confusion_matrices_university.png", "University ML confusion matrices")

    with st.expander("Neural networks and comparison (Task 7)", icon=":material/hub:"):
        for key, label in (
            ("dnn_comparison", "Neural network versus best ML model"),
            ("dnn_evaluation", "Neural network evaluation"),
            ("dnn_hyperparameters", "Settings tried"),
            ("dnn_overfitting", "Overfitting check"),
        ):
            if key in reports:
                st.markdown(f"**{label}**")
                st.dataframe(reports[key], hide_index=True, alt=label)
        _show_image(TASK7 / "accuracy_loss_curves.png", "Neural network training and validation curves")
        _show_image(TASK7 / "dnn_roc_confusion.png", "Neural network ROC and confusion matrices")

    with st.expander("Feature importance and correlations (Task 5)", icon=":material/insights:"):
        for key, label in (
            ("graduate_importance", "Graduate feature importance"),
            ("university_importance", "University feature importance"),
            ("permutation_importance", "Permutation importance (Task 8)"),
        ):
            if key in reports:
                st.markdown(f"**{label}**")
                st.dataframe(reports[key], hide_index=key != "graduate_importance", alt=label)
        for key, label in (
            ("graduate_correlation", "Graduate correlation matrix"),
            ("university_correlation", "University correlation matrix"),
        ):
            if key in reports:
                st.markdown(f"**{label}**")
                st.dataframe(reports[key], alt=label)

    with st.expander("Explainability tables (Task 8)", icon=":material/table_chart:"):
        for key, label in (
            ("shap", "SHAP report"),
            ("lime", "LIME report"),
            ("fairness_by_group", "Fairness metrics by group"),
            ("counterfactuals", "Counterfactual analysis"),
        ):
            if key in reports:
                st.markdown(f"**{label}**")
                st.dataframe(reports[key], hide_index=True, alt=label)
        st.markdown("**Interactive LIME case studies** (open in a browser)")
        with st.container(horizontal=True):
            for filename in (
                "lime_graduate_confident_admit.html",
                "lime_graduate_borderline.html",
                "lime_graduate_confident_reject.html",
                "lime_university_confident_admit.html",
                "lime_university_borderline.html",
                "lime_university_confident_reject.html",
            ):
                path = TASK8 / filename
                if path.is_file():
                    st.download_button(
                        filename.removeprefix("lime_").removesuffix(".html").replace("_", " "),
                        path.read_bytes(),
                        file_name=filename,
                        mime="text/html",
                        key=f"download_{filename}",
                        icon=":material/download:",
                    )

    st.caption(
        "Personalized reports contain only the submitted profile fields and model outputs; "
        "historical usernames and profile links are not loaded into or exported by this app."
    )


def main() -> None:
    st.set_page_config(page_title="UniMatch AI", page_icon="🎓", layout="wide")
    st.title("🎓 UniMatch AI")
    st.caption("Find graduate programs that fit your profile, and see why.")
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
        grad_default = _default_model(
            data["graduate_metrics"][data["graduate_metrics"]["model"].astype(str).isin(graduate_models)]
        )
        university_default = _default_model(
            data["university_metrics"][data["university_metrics"]["model"].astype(str).isin(university_models)]
        )
    except (KeyError, ValueError) as error:
        st.error(f"Model evaluation metadata is invalid: {error}")
        st.stop()

    with st.sidebar:
        st.header("Settings")
        top_n = st.slider(
            "Universities to show",
            min_value=3,
            max_value=15,
            value=8,
            help="How many matches to list in the Your matches tab.",
        )
        with st.expander("Advanced: choose the models", icon=":material/tune:"):
            st.caption(
                "The defaults are the models that scored best on held-out data. A neural network is "
                "always used alongside the model chosen here."
            )
            for warning in (graduate_model_warning, university_model_warning):
                if warning:
                    st.warning(warning)
            graduate_ml_name = st.selectbox(
                "Overall outlook model", graduate_models, index=graduate_models.index(grad_default)
            )
            university_ml_name = st.selectbox(
                "University model", university_models, index=university_models.index(university_default)
            )

    with st.container(horizontal=True):
        for icon, step, text in (
            (":material/edit_note:", "1. Tell us about you", "Scores, grades and what you want to study."),
            (":material/school:", "2. See your matches", "Universities ranked by chance, fit and ranking."),
            (":material/lightbulb:", "3. Understand why", "What helps and what holds you back."),
        ):
            with st.container(border=True):
                st.markdown(f"**{icon} {step}**")
                st.caption(text)

    # Jump to the results after a successful run (state must be set before the tabs are created).
    if st.session_state.pop("show_results", False):
        st.session_state["active_tab"] = TAB_LABELS[1]
    tabs = st.tabs(TAB_LABELS, key="active_tab", on_change="rerun")

    result = st.session_state.get("unimatch_result")
    settings_changed = bool(result) and (
        result["graduate_ml_model"] != graduate_ml_name
        or result["university_ml_model"] != university_ml_name
        or result["top_n"] != top_n
    )
    with tabs[0]:
        render_profile_tab(data, graduate_ml_name, university_ml_name, top_n)
    with tabs[1]:
        render_matches_tab(result, settings_changed)
    with tabs[2]:
        render_why_tab(result, data)
    with tabs[3]:
        render_details_tab(data)


if __name__ == "__main__":
    main()
