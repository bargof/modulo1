from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class FinancialSentimentDataset(Dataset):
    """Dataset de PyTorch para embeddings de textos financieros."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        """Inicializa el dataset con embeddings y etiquetas.

        Parameters
        ----------
        embeddings : np.ndarray
            Arreglo con los embeddings de los textos.
        labels : np.ndarray
            Arreglo con las etiquetas numéricas de sentimiento.
        """
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        """Regresa el número de observaciones del dataset.

        Returns
        -------
        int
            Número total de ejemplos.
        """
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Regresa un embedding y su etiqueta correspondiente.

        Parameters
        ----------
        index : int
            Índice del ejemplo a recuperar.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Embedding y etiqueta asociada.
        """
        return self.embeddings[index], self.labels[index]
