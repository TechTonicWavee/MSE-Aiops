# MSE-Aiops: model monitoring and retraining

This repository demonstrates an end-to-end MLOps workflow for a synthetic
customer-churn classifier. It creates its own labelled data, validates an
incoming batch, evaluates the current champion, retrains a separate candidate,
records a quality-gate decision, and logs candidate metrics to MLflow. No real
customer data is used.

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest
python -m mlops_monitoring
```

On Windows, activate the environment with `.venv\Scripts\activate`.

The first run bootstraps `models/champion.pkl` from `data/train.csv`. Later
runs load that saved champion; they do not silently replace it. The workflow
uses `data/new_batch.csv` as the incoming labelled batch, and writes
`models/candidate.pkl`, `reports/evaluation_report.json`, and
`reports/quality_gate_decision.json`. These are generated outputs and are
intentionally not committed.

To use a different output directory:

```bash
python -m mlops_monitoring --output-dir /path/to/run-artifacts
```

## What the workflow does

1. **Prepare data:** generates deterministic historical, incoming, and
   independent gate-validation datasets. It validates required columns,
   missing values, duplicate IDs/records, and binary targets, then reports
   batch size and churn distribution.
2. **Evaluate champion:** predicts on the labelled incoming batch and records
   accuracy, precision, recall, and F1 in `reports/evaluation_report.json`.
3. **Retrain candidate:** combines historical training rows and incoming
   labelled rows and fits the same scikit-learn preprocessing/model pipeline.
   The independent validation dataset is not used for fitting.
4. **Apply the quality gate:** compares champion and candidate on the same
   independent validation data. A candidate is eligible only when its F1 is
   no worse than the champion and its recall is no more than 0.02 lower.
   F1 balances precision and recall for churn detection, while the recall
   tolerance protects against missing too many at-risk customers. The
   classifier uses balanced class weights to account for the less common churn
   class. The decision, metrics, rule, and UTC timestamp are recorded in
   `reports/quality_gate_decision.json`; the champion is never automatically
   overwritten. Promotion requires a separately reviewed, explicit action.
5. **Track and demonstrate:** saves `models/candidate.pkl`,
   `data/combined_training_data.csv`, and JSON reports. Candidate parameters and
   validation metrics are logged to the local SQLite-backed MLflow store in
   `mlflow.db`.

The generated `evaluation_report.json` contains the incoming-data checks and the
champion's incoming-batch metrics as well as both models' gate-validation
metrics. `quality_gate_decision.json` records the eligibility decision and
rationale.

## CI/CD

`.github/workflows/mlops-monitoring.yml` runs tests and the complete workflow
for pushes, pull requests, and manual dispatches. It uploads the reports,
datasets, and candidate as a run artifact. For a non-PR run, a second job
publishes the candidate as a separate CI artifact **only if** the quality gate
marks it eligible. This is candidate-artifact publication, not deployment to
production; the champion remains unchanged.

## Project layout

```text
src/mlops_monitoring/   data generation, validation, training and workflow
tests/                  validation and end-to-end tests
.github/workflows/      CI and gated candidate-artifact publication
data/, models/,         generated CSV, model, and report evidence
reports/, mlflow.db     generated reports and MLflow tracking database
```
