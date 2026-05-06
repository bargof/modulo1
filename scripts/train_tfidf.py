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
from proyecto.models.tfidf_logistic_model import TfidfLogisticSentimentModel

setup_logging(level=settings.log_level, log_file=str(settings.log_file))

logger = logging.getLogger("proyecto.scripts.train_tfidf_model")


def main() -> None:
    """Entrena, evalúa, guarda y registra el modelo TF-IDF en MLflow."""
    dataset_path = settings.processed_data_dir / "financial_phrasebank_processed.csv"
    model_path = settings.models_dir / "tfidf_logistic_model.pkl"

    max_features = 10_000
    ngram_range = (1, 2)
    c_value = 3.0
    class_weight = "balanced"
    random_state = 42
    test_size = 0.2

    logger.info(
        "Configurando MLflow con tracking URI: %s", settings.mlflow_tracking_uri
    )
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    logger.info("Leyendo dataset procesado desde %s", dataset_path)
    df = pd.read_csv(dataset_path)

    logger.info("Creando modelo TF-IDF + Logistic Regression")
    model = TfidfLogisticSentimentModel(
        max_features=max_features,
        ngram_range=ngram_range,
        c_value=c_value,
        class_weight=class_weight,
        random_state=random_state,
    )

    with mlflow.start_run(run_name="tfidf_logistic_regression"):
        logger.info("Registrando parámetros en MLflow")

        mlflow.log_param("model_type", "tfidf_logistic_regression")
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("ngram_range", str(ngram_range))
        mlflow.log_param("c_value", c_value)
        mlflow.log_param("class_weight", class_weight)
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

        logger.info("Métricas finales del modelo TF-IDF")
        logger.info("Accuracy: %.4f", metrics.accuracy)
        logger.info("Macro F1: %.4f", metrics.macro_f1)
        logger.info("Weighted F1: %.4f", metrics.weighted_f1)

        logger.info("Guardando modelo en %s", model_path)
        trained_model.save(str(model_path))

        logger.info("Registrando modelo como artefacto en MLflow")
        mlflow.log_artifact(str(model_path), artifact_path="model")

        logger.info("Run de MLflow terminado correctamente")

    # agrega a reports
    report_path = settings.reports_dir / "model_metrics.json"

    upsert_model_metrics_report(
        report_path=report_path,
        model_name="TF-IDF + Logistic Regression",
        metrics=metrics,
    )


if __name__ == "__main__":
    main()
