from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import accuracy_score, f1_score


@dataclass
class ClassificationMetrics:
    """Métricas principales para un problema de clasificación."""

    accuracy: float
    macro_f1: float
    weighted_f1: float


def calculate_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> ClassificationMetrics:
    """Calcula métricas principales de clasificación.

    Parameters
    ----------
    y_true : list[int]
        Etiquetas reales.
    y_pred : list[int]
        Etiquetas predichas por el modelo.

    Returns
    -------
    ClassificationMetrics
        Métricas de accuracy, macro F1 y weighted F1.
    """
    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro"),
        weighted_f1=f1_score(y_true, y_pred, average="weighted"),
    )
