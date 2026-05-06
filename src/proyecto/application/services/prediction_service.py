from __future__ import annotations

import pandas as pd
from proyecto.domain.models_interface import BaseSentimentModel
from proyecto.domain.sentiment_prediction import SentimentPrediction

LABEL_ID_TO_NAME = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


class SentimentPredictionService:
    """Servicio de predicción para modelos de sentimiento."""

    def __init__(self, model: BaseSentimentModel) -> None:
        """Inicializa el servicio con un modelo de sentimiento.

        Parameters
        ----------
        model : BaseSentimentModel
            Modelo entrenado que cumple la interfaz base.
        """
        self.model = model

    def predict_text(self, text: str) -> SentimentPrediction:
        """Predice el sentimiento de un texto financiero.

        Parameters
        ----------
        text : str
            Texto financiero ingresado por el usuario.

        Returns
        -------
        SentimentPrediction
            Resultado con etiqueta predicha y probabilidades por clase.
        """
        clean_text = text.strip()

        if not clean_text:
            raise ValueError("El texto no puede estar vacío.")

        features = pd.DataFrame({"text": [clean_text]})

        predicted_label_id = self.model.predict(features)[0]
        probabilities_raw = self.model.predict_proba(features)[0]

        predicted_label = LABEL_ID_TO_NAME[predicted_label_id]

        probabilities = {
            LABEL_ID_TO_NAME[index]: float(probability)
            for index, probability in enumerate(probabilities_raw)
        }

        return SentimentPrediction(
            text=clean_text,
            predicted_label=predicted_label,
            predicted_label_id=predicted_label_id,
            probabilities=probabilities,
        )
