import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)
TARGET = "churn"
ID_COLUMN = "record_id"
NUMERIC_FEATURES = (
    "customer_age",
    "tenure_months",
    "monthly_spend",
    "support_tickets",
)
CATEGORICAL_FEATURES = ("contract_type",)
REQUIRED_COLUMNS = (ID_COLUMN, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES, TARGET)
QUALITY_RECALL_TOLERANCE = 0.02


def generate_synthetic_data(size: int, seed: int, id_prefix: str) -> pd.DataFrame:
    """Create reproducible, labelled customer-churn data without external data."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 81, size=size)
    tenure = rng.integers(1, 73, size=size)
    spend = np.clip(rng.normal(72, 24, size=size), 15, 180).round(2)
    tickets = np.clip(rng.poisson(1.2, size=size), 0, 8)
    contract = rng.choice(
        ["monthly", "annual", "two_year"],
        size=size,
        p=[0.48, 0.34, 0.18],
    )

    log_odds = (
        -1.45
        + 0.42 * tickets
        - 0.035 * tenure
        + 0.012 * (spend - 70)
        + np.where(contract == "monthly", 0.75, 0)
        + np.where(contract == "two_year", -0.65, 0)
        + 0.004 * (age - 45)
    )
    probability = 1 / (1 + np.exp(-log_odds))

    data = pd.DataFrame(
        {
            ID_COLUMN: [f"{id_prefix}-{index:06d}" for index in range(size)],
            "customer_age": age,
            "tenure_months": tenure,
            "monthly_spend": spend,
            "support_tickets": tickets,
            "contract_type": contract,
            TARGET: rng.binomial(1, probability),
        }
    )
    return data.loc[:, REQUIRED_COLUMNS]


def validate_dataset(data: pd.DataFrame, name: str) -> dict[str, Any]:
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"{name} is missing required columns: {missing_columns}")

    missing_values = int(data.loc[:, REQUIRED_COLUMNS].isna().sum().sum())
    if missing_values:
        raise ValueError(f"{name} contains {missing_values} missing value(s)")

    duplicate_ids = int(data[ID_COLUMN].duplicated().sum())
    duplicate_records = int(data.drop(columns=ID_COLUMN).duplicated().sum())
    if duplicate_ids or duplicate_records:
        raise ValueError(
            f"{name} contains {duplicate_ids} duplicate ID(s) and "
            f"{duplicate_records} duplicate record(s)"
        )

    target_values = set(data[TARGET].unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"{name} target '{TARGET}' must contain only 0 and 1")
    if data.empty:
        raise ValueError(f"{name} must contain at least one record")

    counts = data[TARGET].value_counts().sort_index()
    return {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "missing_values": missing_values,
        "duplicate_ids": duplicate_ids,
        "duplicate_records": duplicate_records,
        "target_distribution": {
            str(label): int(count) for label, count in counts.items()
        },
    }


def build_model() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocess),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def calculate_metrics(model: Pipeline, data: pd.DataFrame) -> dict[str, float]:
    predictions = model.predict(data.loc[:, [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]])
    actual = data[TARGET]
    return {
        "accuracy": float(accuracy_score(actual, predictions)),
        "precision": float(precision_score(actual, predictions, zero_division=0)),
        "recall": float(recall_score(actual, predictions, zero_division=0)),
        "f1": float(f1_score(actual, predictions, zero_division=0)),
    }


def _write_json(path: Path, contents: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contents, indent=2) + "\n", encoding="utf-8")


def _track_mlflow(
    output_dir: Path,
    training_rows: int,
    candidate_metrics: dict[str, float],
) -> dict[str, str]:
    try:
        import mlflow
    except ImportError:
        LOGGER.info("MLflow is not installed; skipping optional run tracking.")
        return {"status": "skipped", "reason": "mlflow is not installed"}

    tracking_db = (output_dir / "mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
    mlflow.set_experiment("synthetic-churn-monitoring")
    with mlflow.start_run(run_name="candidate-retraining"):
        mlflow.log_param("training_rows", training_rows)
        mlflow.log_param("model", "logistic_regression_pipeline")
        mlflow.log_metrics(
            {f"validation_{name}": value for name, value in candidate_metrics.items()}
        )
        run_id = mlflow.active_run().info.run_id
    return {"status": "logged", "run_id": run_id}


def run_workflow(output_dir: Path = Path(".")) -> dict[str, Any]:
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    historical_path = data_dir / "train.csv"
    if historical_path.exists():
        historical = pd.read_csv(historical_path)
    else:
        historical = generate_synthetic_data(1000, seed=17, id_prefix="historical")
        historical.to_csv(historical_path, index=False)
    validate_dataset(historical, "historical training data")

    incoming_path = data_dir / "new_batch.csv"
    if incoming_path.exists():
        incoming = pd.read_csv(incoming_path)
    else:
        incoming = generate_synthetic_data(300, seed=29, id_prefix="incoming")
        incoming.to_csv(incoming_path, index=False)
    validation = generate_synthetic_data(500, seed=41, id_prefix="validation")
    validation_path = data_dir / "gate_validation_data.csv"
    validation.to_csv(validation_path, index=False)
    incoming_summary = validate_dataset(incoming, "incoming labelled batch")
    validate_dataset(validation, "gate validation data")
    print(
        f"Incoming labelled batch: {incoming_summary['rows']} rows; "
        f"target distribution (churn): {incoming_summary['target_distribution']}"
    )

    champion_path = models_dir / "champion.pkl"
    champion_initialized = not champion_path.exists()
    if champion_initialized:
        champion = build_model()
        champion.fit(
            historical.loc[:, [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]],
            historical[TARGET],
        )
        joblib.dump(champion, champion_path)
        LOGGER.info("Bootstrapped the initial champion from historical data.")
    else:
        champion = joblib.load(champion_path)

    incoming_metrics = calculate_metrics(champion, incoming)
    combined = pd.concat([historical, incoming], ignore_index=True)
    validate_dataset(combined, "combined training data")
    combined_path = data_dir / "combined_training_data.csv"
    combined.to_csv(combined_path, index=False)

    candidate = build_model()
    candidate.fit(
        combined.loc[:, [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]],
        combined[TARGET],
    )
    candidate_path = models_dir / "candidate.pkl"
    joblib.dump(candidate, candidate_path)

    champion_validation_metrics = calculate_metrics(champion, validation)
    candidate_validation_metrics = calculate_metrics(candidate, validation)
    eligible = (
        candidate_validation_metrics["f1"] >= champion_validation_metrics["f1"]
        and candidate_validation_metrics["recall"]
        >= champion_validation_metrics["recall"] - QUALITY_RECALL_TOLERANCE
    )
    gate_decision = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "eligible_for_promotion": eligible,
        "decision": "eligible_for_promotion" if eligible else "retain_champion",
        "rule": (
            "candidate F1 must be >= champion F1 and candidate recall must be "
            f">= champion recall - {QUALITY_RECALL_TOLERANCE:.2f}"
        ),
        "criteria": {
            "candidate_f1_at_least_champion_f1": (
                candidate_validation_metrics["f1"]
                >= champion_validation_metrics["f1"]
            ),
            "candidate_recall_within_tolerance": (
                candidate_validation_metrics["recall"]
                >= champion_validation_metrics["recall"]
                - QUALITY_RECALL_TOLERANCE
            ),
            "maximum_recall_regression": QUALITY_RECALL_TOLERANCE,
        },
        "champion_validation_metrics": champion_validation_metrics,
        "candidate_validation_metrics": candidate_validation_metrics,
        "champion_updated": False,
        "candidate_model": str(candidate_path),
        "champion_model": str(champion_path),
    }
    _write_json(reports_dir / "quality_gate_decision.json", gate_decision)

    tracking_status = _track_mlflow(
        output_dir, len(combined), candidate_validation_metrics
    )
    evaluation = {
        "incoming_batch": incoming_summary,
        "champion_metrics_on_incoming_batch": incoming_metrics,
        "gate_validation": {
            "rows": int(len(validation)),
            "champion_metrics": champion_validation_metrics,
            "candidate_metrics": candidate_validation_metrics,
        },
        "training": {
            "historical_rows": int(len(historical)),
            "incoming_rows_added": int(len(incoming)),
            "combined_rows": int(len(combined)),
            "candidate_model": str(candidate_path),
            "champion_initialized_this_run": champion_initialized,
            "mlflow": tracking_status,
        },
        "quality_gate": gate_decision,
    }
    _write_json(reports_dir / "evaluation_report.json", evaluation)
    return {
        "evaluation_report": str(reports_dir / "evaluation_report.json"),
        "quality_gate_report": str(reports_dir / "quality_gate_decision.json"),
        "candidate_model": str(candidate_path),
        "champion_model": str(champion_path),
        "eligible_for_promotion": eligible,
        "champion_updated": False,
    }
