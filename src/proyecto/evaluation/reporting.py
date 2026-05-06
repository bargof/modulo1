from __future__ import annotations

import json
from pathlib import Path

from proyecto.evaluation.metrics import ClassificationMetrics


def upsert_model_metrics_report(
    report_path: Path,
    model_name: str,
    metrics: ClassificationMetrics,
) -> None:
    """Crea o actualiza el reporte JSON con métricas de modelos.

    Parameters
    ----------
    report_path : Path
        Ruta del archivo JSON donde se guardan las métricas.
    model_name : str
        Nombre del modelo evaluado.
    metrics : ClassificationMetrics
        Métricas calculadas para el modelo.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as file:
            report_data = json.load(file)
    else:
        report_data = []

    new_record = {
        "model": model_name,
        "accuracy": round(metrics.accuracy, 4),
        "macro_f1": round(metrics.macro_f1, 4),
        "weighted_f1": round(metrics.weighted_f1, 4),
        "confusion_matrix": metrics.confusion_matrix,
    }

    report_data = [record for record in report_data if record["model"] != model_name]
    report_data.append(new_record)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report_data, file, indent=2)
