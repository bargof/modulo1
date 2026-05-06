from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from proyecto.domain.models_interface import BaseSentimentModel
from proyecto.models.pytorch_dataset import FinancialSentimentDataset
from proyecto.models.pytorch_mlp import SentimentMLP
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class EmbeddingMLPSentimentModel(BaseSentimentModel):
    """Modelo de sentimiento basado en embeddings y una red MLP en PyTorch."""

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim: int = 128,
        output_dim: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 20,
        random_state: int = 42,
    ) -> None:
        """Inicializa el modelo de embeddings + MLP.

        Parameters
        ----------
        embedding_model_name : str
            Nombre del modelo de sentence-transformers usado para generar embeddings.
        hidden_dim : int
            Número de neuronas en la capa oculta del MLP.
        output_dim : int
            Número de clases de salida.
        dropout : float
            Proporción de dropout usada en el MLP.
        learning_rate : float
            Tasa de aprendizaje del optimizador.
        batch_size : int
            Tamaño de batch durante entrenamiento.
        epochs : int
            Número de épocas de entrenamiento.
        random_state : int
            Semilla para reproducibilidad.
        """
        self.embedding_model_name = embedding_model_name
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.model: SentimentMLP | None = None

    def train(self, features_train: pd.DataFrame, target_train: pd.Series) -> None:
        """Entrena el modelo con textos financieros y etiquetas.

        Parameters
        ----------
        features_train : pd.DataFrame
            DataFrame con una columna `text`.
        target_train : pd.Series
            Etiquetas reales de entrenamiento.
        """
        logger.info("Generando embeddings de entrenamiento")
        embeddings = self._encode_texts(features_train["text"].tolist())

        input_dim = embeddings.shape[1]

        self.model = SentimentMLP(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dropout=self.dropout,
        ).to(self.device)

        dataset = FinancialSentimentDataset(
            embeddings=embeddings,
            labels=target_train.to_numpy(),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        logger.info("Iniciando entrenamiento del modelo PyTorch")

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            for batch_embeddings, batch_labels in dataloader:
                batch_embeddings = batch_embeddings.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()

                logits = self.model(batch_embeddings)
                loss = criterion(logits, batch_labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_embeddings.size(0)

            epoch_loss = running_loss / len(dataloader.dataset)

            logger.info(
                "Epoch %s/%s | train_loss=%.4f",
                epoch + 1,
                self.epochs,
                epoch_loss,
            )

        logger.info("Entrenamiento del modelo PyTorch terminado")

    def predict(self, features: pd.DataFrame) -> list[int]:
        """Predice clases de sentimiento para nuevos textos.

        Parameters
        ----------
        features : pd.DataFrame
            DataFrame con una columna `text`.

        Returns
        -------
        list[int]
            Clases predichas.
        """
        probabilities = self.predict_proba(features)
        predictions = np.argmax(probabilities, axis=1)

        return predictions.tolist()

    def predict_proba(self, features: pd.DataFrame) -> list[list[float]]:
        """Predice probabilidades por clase para nuevos textos.

        Parameters
        ----------
        features : pd.DataFrame
            DataFrame con una columna `text`.

        Returns
        -------
        list[list[float]]
            Probabilidades por clase para cada texto.
        """
        if self.model is None:
            raise ValueError("El modelo PyTorch no ha sido entrenado o cargado.")

        embeddings = self._encode_texts(features["text"].tolist())

        self.model.eval()

        with torch.no_grad():
            tensor_embeddings = torch.tensor(
                embeddings,
                dtype=torch.float32,
                device=self.device,
            )

            logits = self.model(tensor_embeddings)
            probabilities = torch.softmax(logits, dim=1)

        return probabilities.cpu().numpy().tolist()

    def save(self, path: str) -> None:
        """Guarda el modelo PyTorch entrenado en disco.

        Parameters
        ----------
        path : str
            Ruta donde se guardará el modelo.
        """
        if self.model is None:
            raise ValueError("No hay modelo entrenado para guardar.")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "embedding_model_name": self.embedding_model_name,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout": self.dropout,
            "input_dim": self.model.network[0].in_features,
        }

        logger.info("Guardando modelo PyTorch en %s", output_path)

        with output_path.open("wb") as file:
            torch.save(checkpoint, file)

        logger.info("Modelo PyTorch guardado correctamente")

    def load(self, path: str) -> None:
        """Carga un modelo PyTorch entrenado desde disco.

        Parameters
        ----------
        path : str
            Ruta desde donde se cargará el modelo.
        """
        input_path = Path(path)

        logger.info("Cargando modelo PyTorch desde %s", input_path)

        checkpoint = torch.load(
            input_path,
            map_location=self.device,
        )

        self.embedding_model_name = checkpoint["embedding_model_name"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.output_dim = checkpoint["output_dim"]
        self.dropout = checkpoint["dropout"]

        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        self.model = SentimentMLP(
            input_dim=checkpoint["input_dim"],
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dropout=self.dropout,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        logger.info("Modelo PyTorch cargado correctamente")

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Convierte textos en embeddings numéricos.

        Parameters
        ----------
        texts : list[str]
            Lista de textos financieros.

        Returns
        -------
        np.ndarray
            Arreglo de embeddings.
        """
        return self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
