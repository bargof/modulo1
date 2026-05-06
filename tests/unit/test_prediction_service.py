from __future__ import annotations

import pandas as pd
import pytest
from proyecto.application.services.prediction_service import SentimentPredictionService


class FakeSentimentModel:
    """Modelo fake para probar el servicio de predicción."""

    def train(self, features_train: pd.DataFrame, target_train: pd.Series) -> None:
        """Simula entrenamiento sin hacer nada."""
        pass

    def predict(self, features: pd.DataFrame) -> list[int]:
        """Regresa siempre la clase positive."""
        return [2]

    def predict_proba(self, features: pd.DataFrame) -> list[list[float]]:
        """Regresa probabilidades fake para negative, neutral y positive."""
        return [[0.05, 0.15, 0.80]]

    def save(self, path: str) -> None:
        """Simula guardado sin hacer nada."""
        pass

    def load(self, path: str) -> None:
        """Simula carga sin hacer nada."""
        pass


def test_predict_text_returns_sentiment_prediction() -> None:
    """Valida que el servicio regrese una predicción estructurada."""
    model = FakeSentimentModel()
    service = SentimentPredictionService(model=model)

    result = service.predict_text("The company reported higher profits.")

    assert result.text == "The company reported higher profits."
    assert result.predicted_label == "positive"
    assert result.predicted_label_id == 2
    assert result.probabilities == {
        "negative": 0.05,
        "neutral": 0.15,
        "positive": 0.80,
    }


def test_predict_text_strips_input_text() -> None:
    """Valida que el servicio limpie espacios al inicio y al final."""
    model = FakeSentimentModel()
    service = SentimentPredictionService(model=model)

    result = service.predict_text("  The company reported higher profits.  ")

    assert result.text == "The company reported higher profits."


def test_predict_text_raises_error_for_empty_text() -> None:
    """Valida que el servicio falle si el texto está vacío."""
    model = FakeSentimentModel()
    service = SentimentPredictionService(model=model)

    with pytest.raises(ValueError, match="El texto no puede estar vacío"):
        service.predict_text("   ")
