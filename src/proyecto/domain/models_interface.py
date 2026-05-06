from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseSentimentModel(ABC):
    """Interfaz base para modelos de clasificación de sentimiento.

    Esta clase define el contrato que deben cumplir todos los modelos
    de sentimiento del proyecto. Permite intercambiar modelos concretos
    sin modificar el código que los utiliza.
    """

    @abstractmethod
    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Entrena el modelo con datos de entrenamiento.

        Parameters
        ----------
        x_train : pd.DataFrame
            Datos de entrada para entrenamiento.
        y_train : pd.Series
            Etiquetas reales de entrenamiento.
        """
        pass

    @abstractmethod
    def predict(self, x: pd.DataFrame) -> list[int]:
        """Predice la clase de sentimiento para nuevos textos.

        Parameters
        ----------
        x : pd.DataFrame
            Datos de entrada con textos financieros.

        Returns
        -------
        list[int]
            Lista de clases predichas.
        """
        pass

    @abstractmethod
    def predict_proba(self, x: pd.DataFrame) -> list[list[float]]:
        """Predice probabilidades por clase para nuevos textos.

        Parameters
        ----------
        x : pd.DataFrame
            Datos de entrada con textos financieros.

        Returns
        -------
        list[list[float]]
            Probabilidades por clase para cada observación.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Guarda el modelo entrenado en disco.

        Parameters
        ----------
        path : str
            Ruta donde se guardará el modelo.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Carga un modelo previamente entrenado desde disco.

        Parameters
        ----------
        path : str
            Ruta desde donde se cargará el modelo.
        """
        pass
