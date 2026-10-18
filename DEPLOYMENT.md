# UniMatch AI deployment

## What the app needs

The Streamlit entry point is `app.py`. At startup it reads the trained models, scalers, feature
metadata, evaluation outputs and source ranking table from the repository's existing `outputs/` and
`data/raw/` folders. It does not train models at runtime.

Keep these paths and artifacts together when deploying:

- `outputs/task5_output/feature_matrix_graduate.csv`
- `outputs/task5_output/feature_matrix_university.csv`
- `outputs/task5_output/university_features_full.csv`
- `outputs/task6_output/models/` and evaluation report CSVs
- `outputs/task7_output/dnn_config.json`, `dnn_*.keras`, and `dnn_scaler_*.pkl`
- `outputs/task8_output/` for the saved XAI and fairness displays
- `data/raw/World_University_Rankings_2026.csv`

The app loads only the university history columns needed for feature construction. It does not load
or expose historical applicant usernames or profile links. Avoid adding those identifiers to an
applicant report or public deployment.

## Run locally

From the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit prints the local URL (normally `http://localhost:8501`). The first DNN prediction may take
longer while TensorFlow loads the saved network.

## Deploy to Streamlit Community Cloud

1. Push the repository and required model/data artifacts to a GitHub repository accessible to
   Streamlit Community Cloud. The app cannot run if large model files or required CSVs are omitted.
   If GitHub file-size limits prevent committing an artifact, configure Git LFS or a private,
   access-controlled artifact store and change the app's artifact loading paths accordingly; do not
   make private applicant records public.
2. In Streamlit Community Cloud, create an app from that repository and branch, set the main file
   path to `app.py`, and deploy from the repository root.
3. Cloud installs dependencies from `requirements.txt`. Ensure the selected Python runtime and the
   installed scikit-learn/XGBoost/TensorFlow versions can load the serialized models. Pickled
   scikit-learn objects are not guaranteed to be compatible across library versions; use the same
   versions used to train the artifacts, or retrain and export models in the deployment environment.
4. Review app logs after deployment. A missing model, scaler, or source CSV is reported explicitly
   in the app; restore the artifact rather than suppressing the error.

The app checks whether the saved XGBoost pickles can be loaded before offering XGBoost in the model
selector. If the installed XGBoost version cannot read an artifact, that option is disabled with a
warning while the other saved classifiers remain available. Rerun Task 6 with the deployment
environment's XGBoost version to regenerate compatible model artifacts.

This application requires no API keys or Streamlit secrets. Do not add credentials or personal
applicant information to source control.

## Before sharing a public deployment

- Confirm all required model artifacts are present and predictions complete for both ML and DNN.
- Verify the installed libraries against the model-training environment. The current repository does
  not include a lockfile or a recorded training-environment export.
- Explain that outputs are estimates based on historical/self-reported data and are not admission
  guarantees. Rankings are a 2026 snapshot.
- Recommendations do **not** estimate tuition or affordability; no compatible fee data is included.
- The displayed fairness results use available proxy groups and are not a fairness certification.
- Local sensitivity explanations change one feature at a time and are not causal explanations.
- Do not treat the output as the sole basis for an admissions, funding, or other consequential
  decision.
