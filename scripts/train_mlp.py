from __future__ import annotations

import logging

import mlflow
import pandas as pd
from proyecto.application.services.training_service import (
    train_and_evaluate_sentiment_model,
)
from proyecto.config.logging import setup_logging
from proyecto.config.settings import settings
from proyecto.evaluation.reporting import upsert_model_metrics_report
from proyecto.models.mlp_model import EmbeddingMLPSentimentModel

setup_logging(level=settings.log_level, log_file=str(settings.log_file))

logger = logging.getLogger("proyecto.scripts.train_embedding_mlp_model")


def main() -> None:
    """Entrena, evalúa, guarda y registra el modelo MLP.

    El experimento se registra en MLflow."""
    dataset_path = settings.processed_data_dir / "financial_phrasebank_processed.csv"
    model_path = settings.models_dir / "embedding_mlp_model.pt"

    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    hidden_dim = 128
    output_dim = 3
    dropout = 0.2
    learning_rate = 1e-3
    batch_size = 32
    epochs = 20
    random_state = 42
    test_size = 0.2

    logger.info(
        "Configurando MLflow con tracking URI: %s", settings.mlflow_tracking_uri
    )
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    logger.info("Leyendo dataset procesado desde %s", dataset_path)
    df = pd.read_csv(dataset_path)

    logger.info("Creando modelo Embeddings + PyTorch MLP")
    model = EmbeddingMLPSentimentModel(
        embedding_model_name=embedding_model_name,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        random_state=random_state,
    )

    with mlflow.start_run(run_name="embedding_mlp_pytorch"):
        logger.info("Registrando parámetros en MLflow")

        mlflow.log_param("model_type", "embedding_mlp_pytorch")
        mlflow.log_param("embedding_model_name", embedding_model_name)
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("output_dim", output_dim)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", test_size)

        logger.info("Entrenando y evaluando modelo")
        trained_model, metrics = train_and_evaluate_sentiment_model(
            model=model,
            df=df,
            test_size=test_size,
            random_state=random_state,
        )

        logger.info("Registrando métricas en MLflow")
        mlflow.log_metric("accuracy", metrics.accuracy)
        mlflow.log_metric("macro_f1", metrics.macro_f1)
        mlflow.log_metric("weighted_f1", metrics.weighted_f1)

        logger.info("Métricas finales del modelo Embeddings + PyTorch MLP")
        logger.info("Accuracy: %.4f", metrics.accuracy)
        logger.info("Macro F1: %.4f", metrics.macro_f1)
        logger.info("Weighted F1: %.4f", metrics.weighted_f1)

        logger.info("Guardando modelo en %s", model_path)
        trained_model.save(str(model_path))

        logger.info("Registrando modelo como artefacto en MLflow")
        mlflow.log_artifact(str(model_path), artifact_path="model")

        logger.info("Run de MLflow terminado correctamente")

    # agrega a reports/
    report_path = settings.reports_dir / "model_metrics.json"

    upsert_model_metrics_report(
        report_path=report_path,
        model_name="Embeddings + PyTorch MLP",
        metrics=metrics,
    )


if __name__ == "__main__":
    main()
