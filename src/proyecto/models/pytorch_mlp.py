from __future__ import annotations

import torch
from torch import nn


class SentimentMLP(nn.Module):
    """Red neuronal MLP para clasificación de sentimiento financiero."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 3,
        dropout: float = 0.2,
    ) -> None:
        """Inicializa las capas de la red neuronal.

        Parameters
        ----------
        input_dim : int
            Dimensión del embedding de entrada.
        hidden_dim : int
            Número de neuronas en la capa oculta.
        output_dim : int
            Número de clases de salida.
        dropout : float
            Proporción de neuronas apagadas durante entrenamiento.
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Ejecuta el forward pass del modelo.

        Parameters
        ----------
        inputs : torch.Tensor
            Batch de embeddings.

        Returns
        -------
        torch.Tensor
            Logits para cada clase.
        """
        return self.network(inputs)
