from __future__ import annotations

import logging

import pandas as pd
from proyecto.application.services.dataset_preprocessing_service import (
    preprocess_financial_phrasebank,
)
from proyecto.config.logging import setup_logging
from proyecto.config.settings import settings

setup_logging(level=settings.log_level, log_file=str(settings.log_file))

logger = logging.getLogger(__name__)


def main() -> None:
    """Ejecuta el preprocesamiento del dataset Financial PhraseBank."""
    input_path = settings.raw_data_dir / "financial_phrasebank.csv"
    output_path = settings.processed_data_dir / "financial_phrasebank_processed.csv"

    logger.info("Leyendo dataset raw desde %s", input_path)
    raw_df = pd.read_csv(input_path)

    logger.info("Preprocesando dataset")
    processed_df = preprocess_financial_phrasebank(raw_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Guardando dataset procesado en %s", output_path)
    processed_df.to_csv(output_path, index=False)

    logger.info("Dataset procesado guardado correctamente")


if __name__ == "__main__":
    main()
