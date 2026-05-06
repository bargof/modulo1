from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from proyecto.domain.models_interface import BaseSentimentModel
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class TfidfLogisticSentimentModel(BaseSentimentModel):
    """Modelo de sentimiento basado en TF-IDF y Logistic Regression."""

    def __init__(
        self,
        max_features: int = 10_000,
        ngram_range: tuple[int, int] = (1, 2),
        c_value: float = 3.0,
        class_weight: str | None = "balanced",
        random_state: int = 42,
    ) -> None:
        """Inicializa el modelo TF-IDF + Logistic Regression.

        Parameters
        ----------
        max_features : int
            Número máximo de términos que conservará TF-IDF.
        ngram_range : tuple[int, int]
            Rango de n-gramas que usará TF-IDF.
        c_value : float
            Parámetro de regularización de Logistic Regression.
        class_weight : str | None
            Estrategia de ponderación de clases.
        random_state : int
            Semilla para reproducibilidad.
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.c_value = c_value
        self.class_weight = class_weight
        self.random_state = random_state

        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        """Construye el pipeline de scikit-learn."""
        text_preprocessor = ColumnTransformer(
            transformers=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        stop_words=None,
                        ngram_range=self.ngram_range,
                        max_features=self.max_features,
                    ),
                    "text",
                )
            ],
            remainder="drop",
        )

        return Pipeline(
            steps=[
                ("preprocessor", text_preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=self.random_state,
                        C=self.c_value,
                        class_weight=self.class_weight,
                    ),
                ),
            ]
        )

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Entrena el modelo con datos de entrenamiento.

        Parameters
        ----------
        x_train : pd.DataFrame
            DataFrame con una columna `text`.
        y_train : pd.Series
            Etiquetas reales de entrenamiento.
        """
        logger.info("Entrenando modelo TF-IDF + Logistic Regression")
        self.pipeline.fit(x_train, y_train)
        logger.info("Modelo TF-IDF entrenado correctamente")

    def predict(self, x: pd.DataFrame) -> list[int]:
        """Predice clases de sentimiento.

        Parameters
        ----------
        x : pd.DataFrame
            DataFrame con una columna `text`.

        Returns
        -------
        list[int]
            Clases predichas.
        """
        predictions = self.pipeline.predict(x)
        return predictions.tolist()

    def predict_proba(self, x: pd.DataFrame) -> list[list[float]]:
        """Predice probabilidades por clase.

        Parameters
        ----------
        x : pd.DataFrame
            DataFrame con una columna `text`.

        Returns
        -------
        list[list[float]]
            Probabilidades por clase para cada texto.
        """
        probabilities = self.pipeline.predict_proba(x)
        return probabilities.tolist()

    def save(self, path: str) -> None:
        """Guarda el modelo entrenado en disco.

        Parameters
        ----------
        path : str
            Ruta donde se guardará el modelo.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Guardando modelo TF-IDF en %s", output_path)

        with output_path.open("wb") as file:
            joblib.dump(self.pipeline, file)

        logger.info("Modelo TF-IDF guardado correctamente")

    def load(self, path: str) -> None:
        """Carga un modelo entrenado desde disco.

        Parameters
        ----------
        path : str
            Ruta del modelo guardado.
        """
        input_path = Path(path)

        logger.info("Cargando modelo TF-IDF desde %s", input_path)

        with input_path.open("rb") as file:
            self.pipeline = joblib.load(file)

        logger.info("Modelo TF-IDF cargado correctamente")
