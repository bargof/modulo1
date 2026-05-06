from __future__ import annotations

import json

from proyecto.evaluation.metrics import ClassificationMetrics
from proyecto.evaluation.reporting import upsert_model_metrics_report


def test_upsert_model_metrics_report_creates_file(tmp_path) -> None:
    """Valida que el reporte de métricas se cree correctamente."""
    report_path = tmp_path / "model_metrics.json"

    metrics = ClassificationMetrics(
        accuracy=0.75,
        macro_f1=0.70,
        weighted_f1=0.74,
        confusion_matrix=[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
    )

    upsert_model_metrics_report(
        report_path=report_path,
        model_name="Test Model",
        metrics=metrics,
    )

    assert report_path.exists()

    with report_path.open("r", encoding="utf-8") as file:
        report_data = json.load(file)

    assert len(report_data) == 1
    assert report_data[0]["model"] == "Test Model"
    assert report_data[0]["accuracy"] == 0.75
    assert report_data[0]["macro_f1"] == 0.70
    assert report_data[0]["weighted_f1"] == 0.74
    assert report_data[0]["confusion_matrix"] == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]


def test_upsert_model_metrics_report_updates_existing_model(tmp_path) -> None:
    """Valida que el reporte actualice un modelo existente sin duplicarlo."""
    report_path = tmp_path / "model_metrics.json"

    initial_metrics = ClassificationMetrics(
        accuracy=0.70,
        macro_f1=0.65,
        weighted_f1=0.68,
        confusion_matrix=[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
    )

    updated_metrics = ClassificationMetrics(
        accuracy=0.80,
        macro_f1=0.75,
        weighted_f1=0.78,
        confusion_matrix=[
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
        ],
    )

    upsert_model_metrics_report(
        report_path=report_path,
        model_name="Test Model",
        metrics=initial_metrics,
    )

    upsert_model_metrics_report(
        report_path=report_path,
        model_name="Test Model",
        metrics=updated_metrics,
    )

    with report_path.open("r", encoding="utf-8") as file:
        report_data = json.load(file)

    assert len(report_data) == 1
    assert report_data[0]["model"] == "Test Model"
    assert report_data[0]["accuracy"] == 0.80
    assert report_data[0]["macro_f1"] == 0.75
    assert report_data[0]["weighted_f1"] == 0.78
    assert report_data[0]["confusion_matrix"] == [
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
    ]
