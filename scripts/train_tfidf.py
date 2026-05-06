from __future__ import annotations

import logging

import pandas as pd
from proyecto.application.services.training_service import (
    train_and_evaluate_sentiment_model,
)
from proyecto.config.logging import setup_logging
from proyecto.config.settings import settings
from proyecto.models.tfidf_logistic_model import TfidfLogisticSentimentModel

setup_logging(level=settings.log_level, log_file=str(settings.log_file))

logger = logging.getLogger(__name__)


def main() -> None:
    """Entrena y guarda el modelo TF-IDF + Logistic Regression."""
    dataset_path = settings.processed_data_dir / "financial_phrasebank_processed.csv"
    model_path = settings.models_dir / "tfidf_logistic_model.pkl"

    logger.info("Leyendo dataset procesado desde %s", dataset_path)
    df = pd.read_csv(dataset_path)

    logger.info("Creando modelo TF-IDF + Logistic Regression")
    model = TfidfLogisticSentimentModel(
        max_features=10_000,
        ngram_range=(1, 2),
        c_value=3.0,
        class_weight="balanced",
        random_state=42,
    )

    trained_model, metrics = train_and_evaluate_sentiment_model(
        model=model,
        df=df,
        test_size=0.2,
        random_state=42,
    )

    logger.info("Métricas finales del modelo TF-IDF")
    logger.info("Accuracy: %.4f", metrics.accuracy)
    logger.info("Macro F1: %.4f", metrics.macro_f1)
    logger.info("Weighted F1: %.4f", metrics.weighted_f1)

    trained_model.save(str(model_path))

    logger.info("Modelo guardado en %s", model_path)


if __name__ == "__main__":
    main()
