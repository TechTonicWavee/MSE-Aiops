import hashlib
import json

import pandas as pd
import pytest

from mlops_monitoring.workflow import (
    REQUIRED_COLUMNS,
    generate_synthetic_data,
    run_workflow,
    validate_dataset,
)


def test_synthetic_dataset_is_valid_and_reproducible():
    first = generate_synthetic_data(50, seed=7, id_prefix="test")
    second = generate_synthetic_data(50, seed=7, id_prefix="test")

    pd.testing.assert_frame_equal(first, second)
    assert tuple(first.columns) == REQUIRED_COLUMNS
    assert validate_dataset(first, "test")["rows"] == 50


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.drop(columns=["contract_type"]), "missing required columns"),
        (lambda data: data.assign(monthly_spend=float("nan")), "missing value"),
        (
            lambda data: data.assign(record_id=lambda frame: frame["record_id"].iloc[0]),
            "duplicate ID",
        ),
    ],
)
def test_invalid_data_is_rejected(mutation, message):
    data = generate_synthetic_data(10, seed=9, id_prefix="invalid")

    with pytest.raises(ValueError, match=message):
        validate_dataset(mutation(data), "invalid batch")


def test_workflow_writes_reports_and_separate_model_artifacts(tmp_path):
    output_dir = tmp_path
    result = run_workflow(output_dir)
    champion_path = output_dir / "models/champion.pkl"
    champion_hash = hashlib.sha256(champion_path.read_bytes()).digest()
    run_workflow(output_dir)

    evaluation = json.loads((output_dir / "reports/evaluation_report.json").read_text())
    gate = json.loads((output_dir / "reports/quality_gate_decision.json").read_text())

    assert evaluation["incoming_batch"]["rows"] == 300
    assert set(evaluation["champion_metrics_on_incoming_batch"]) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
    }
    assert champion_path.exists()
    assert (output_dir / "models/candidate.pkl").exists()
    assert hashlib.sha256(champion_path.read_bytes()).digest() == champion_hash
    assert evaluation["training"]["champion_initialized_this_run"] is False
    assert evaluation["training"]["mlflow"]["status"] == "logged"
    assert evaluation["training"]["mlflow"]["run_id"]
    assert gate["champion_updated"] is False
    assert gate["timestamp_utc"]
    assert gate["rule"]
    assert gate["eligible_for_promotion"] is result["eligible_for_promotion"]
