from __future__ import annotations

import logging

import pandas as pd
from proyecto.domain.models_interface import BaseSentimentModel
from proyecto.evaluation.metrics import (
    ClassificationMetrics,
    calculate_classification_metrics,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def train_and_evaluate_sentiment_model(
    model: BaseSentimentModel,
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[BaseSentimentModel, ClassificationMetrics]:
    """Entrena y evalúa un modelo de clasificación de sentimiento.

    Parameters
    ----------
    model : BaseSentimentModel
        Modelo de sentimiento que cumple la interfaz base.
    df : pd.DataFrame
        Dataset procesado con columnas text, label y label_id.
    test_size : float
        Proporción del dataset usada para prueba.
    random_state : int
        Semilla para reproducibilidad.

    Returns
    -------
    tuple[BaseSentimentModel, ClassificationMetrics]
        Modelo entrenado y métricas calculadas sobre el conjunto de prueba.
    """
    logger.info("Iniciando entrenamiento y evaluación del modelo")

    _validate_training_dataframe(df)

    x = df[["text"]]
    y = df["label_id"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        "Split realizado: train=%s filas, test=%s filas",
        len(x_train),
        len(x_test),
    )

    model.train(x_train, y_train)

    y_pred = model.predict(x_test)

    metrics = calculate_classification_metrics(
        y_true=y_test.tolist(),
        y_pred=y_pred,
    )

    logger.info(
        "Evaluación terminada | accuracy=%.4f | macro_f1=%.4f | weighted_f1=%.4f",
        metrics.accuracy,
        metrics.macro_f1,
        metrics.weighted_f1,
    )

    return model, metrics


def _validate_training_dataframe(df: pd.DataFrame) -> None:
    """Valida que el DataFrame tenga las columnas necesarias para entrenamiento.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset procesado para entrenamiento.

    Raises
    ------
    ValueError
        Si falta alguna columna requerida.
    """
    required_columns = {"text", "label", "label_id"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Faltan columnas requeridas para entrenamiento: {missing_columns}"
        )
