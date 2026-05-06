from __future__ import annotations

from proyecto.evaluation.metrics import calculate_classification_metrics


def test_calculate_classification_metrics() -> None:
    """Valida que se calculen métricas básicas de clasificación."""
    y_true = [0, 1, 2, 2]
    y_pred = [0, 1, 1, 2]

    metrics = calculate_classification_metrics(y_true=y_true, y_pred=y_pred)

    assert 0 <= metrics.accuracy <= 1
    assert 0 <= metrics.macro_f1 <= 1
    assert 0 <= metrics.weighted_f1 <= 1
    assert metrics.confusion_matrix == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 1],
    ]
