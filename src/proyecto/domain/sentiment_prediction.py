from dataclasses import dataclass


@dataclass
class SentimentPrediction:
    """Resultado de una predicción de sentimiento."""

    text: str
    predicted_label: str
    predicted_label_id: int
    probabilities: dict[str, float]
