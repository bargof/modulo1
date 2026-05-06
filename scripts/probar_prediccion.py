from __future__ import annotations

import argparse
import logging

from proyecto.application.services.prediction_service import SentimentPredictionService
from proyecto.config.logging import setup_logging
from proyecto.config.settings import settings
from proyecto.models.mlp_model import EmbeddingMLPSentimentModel
from proyecto.models.tfidf_logistic_model import TfidfLogisticSentimentModel

setup_logging(level=settings.log_level, log_file=str(settings.log_file))

logger = logging.getLogger("proyecto.scripts.predict_text")


def main() -> None:
    """Carga un modelo entrenado y predice el sentimiento de un texto."""
    parser = argparse.ArgumentParser(
        description="Predice sentimiento financiero usando un modelo entrenado."
    )

    parser.add_argument(
        "--model",
        choices=["tfidf", "mlp"],
        default="tfidf",
        help="Modelo a usar para inferencia.",
    )

    parser.add_argument(
        "--text",
        default="The company reported higher profits and strong revenue growth.",
        help="Texto financiero a clasificar.",
    )

    args = parser.parse_args()

    if args.model == "tfidf":
        model_path = settings.models_dir / "tfidf_logistic_model.pkl"
        model = TfidfLogisticSentimentModel()
    else:
        model_path = settings.models_dir / "embedding_mlp_model.pt"
        model = EmbeddingMLPSentimentModel()

    logger.info("Cargando modelo %s desde %s", args.model, model_path)

    model.load(str(model_path))

    prediction_service = SentimentPredictionService(model=model)
    result = prediction_service.predict_text(args.text)

    print("Texto:")
    print(result.text)
    print()
    print(f"Modelo: {args.model}")
    print(f"Predicción: {result.predicted_label}")
    print()
    print("Probabilidades:")

    for label, probability in result.probabilities.items():
        print(f"- {label}: {probability:.4f}")


if __name__ == "__main__":
    main()
