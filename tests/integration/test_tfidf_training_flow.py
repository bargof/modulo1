from __future__ import annotations

import pandas as pd
from proyecto.application.services.training_service import (
    train_and_evaluate_sentiment_model,
)
from proyecto.models.tfidf_logistic_model import TfidfLogisticSentimentModel


def test_tfidf_training_flow_runs() -> None:
    """Valida que el flujo de entrenamiento TF-IDF corra de inicio a fin."""
    df = pd.DataFrame(
        {
            "text": [
                "The company reported higher profits.",
                "Revenue increased during the quarter.",
                "Sales declined compared with last year.",
                "The company reported lower earnings.",
                "The company announced its annual meeting.",
                "The firm opened a new office.",
                "Operating profit increased significantly.",
                "Losses widened during the period.",
                "The board approved the proposal.",
            ],
            "label": [
                "positive",
                "positive",
                "negative",
                "negative",
                "neutral",
                "neutral",
                "positive",
                "negative",
                "neutral",
            ],
            "label_id": [
                2,
                2,
                0,
                0,
                1,
                1,
                2,
                0,
                1,
            ],
        }
    )

    model = TfidfLogisticSentimentModel(
        max_features=100,
        ngram_range=(1, 1),
        c_value=1.0,
        class_weight=None,
        random_state=42,
    )

    trained_model, metrics = train_and_evaluate_sentiment_model(
        model=model,
        df=df,
        test_size=0.33,
        random_state=42,
    )

    predictions = trained_model.predict(
        pd.DataFrame(
            {
                "text": [
                    "The company reported strong profit growth.",
                    "Sales declined sharply.",
                    "The company announced a meeting.",
                ]
            }
        )
    )

    probabilities = trained_model.predict_proba(
        pd.DataFrame(
            {
                "text": [
                    "The company reported strong profit growth.",
                    "Sales declined sharply.",
                    "The company announced a meeting.",
                ]
            }
        )
    )

    assert len(predictions) == 3
    assert len(probabilities) == 3
    assert all(prediction in [0, 1, 2] for prediction in predictions)

    assert 0 <= metrics.accuracy <= 1
    assert 0 <= metrics.macro_f1 <= 1
    assert 0 <= metrics.weighted_f1 <= 1
