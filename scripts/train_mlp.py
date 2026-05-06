from __future__ import annotations

import logging

import pandas as pd
from proyecto.application.services.training_service import (
    train_and_evaluate_sentiment_model,
)
from proyecto.config.logging import setup_logging
from proyecto.config.settings import settings
from proyecto.models.mlp_model import EmbeddingMLPSentimentModel

setup_logging(level=settings.log_level, log_file=str(settings.log_file))

logger = logging.getLogger("proyecto.scripts.train_embedding_mlp_model")


def main() -> None:
    """Entrena y guarda el modelo de embeddings + PyTorch MLP."""
    dataset_path = settings.processed_data_dir / "financial_phrasebank_processed.csv"
    model_path = settings.models_dir / "embedding_mlp_model.pt"

    logger.info("Leyendo dataset procesado desde %s", dataset_path)
    df = pd.read_csv(dataset_path)

    logger.info("Creando modelo Embeddings + PyTorch MLP")
    model = EmbeddingMLPSentimentModel(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim=128,
        output_dim=3,
        dropout=0.2,
        learning_rate=1e-3,
        batch_size=32,
        epochs=20,
        random_state=42,
    )

    trained_model, metrics = train_and_evaluate_sentiment_model(
        model=model,
        df=df,
        test_size=0.2,
        random_state=42,
    )

    logger.info("Métricas finales del modelo Embeddings + MLP")
    logger.info("Accuracy: %.4f", metrics.accuracy)
    logger.info("Macro F1: %.4f", metrics.macro_f1)
    logger.info("Weighted F1: %.4f", metrics.weighted_f1)

    trained_model.save(str(model_path))

    logger.info("Modelo guardado en %s", model_path)


if __name__ == "__main__":
    main()
