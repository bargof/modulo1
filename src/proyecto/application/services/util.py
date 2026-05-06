import csv
import random
from collections import deque
from collections.abc import Generator

import numpy as np

"""
@author: bargof
@date: 15/04/2026
"""


def fijar_seeds(seed: int = 42) -> None:
    """Fija todas las seeds relevantes para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)

    # Si PyTorch está disponible:
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Para reproducibilidad total en GPU (puede reducir rendimiento):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def leer_filas(
    archivo: str, chunk_size: int = 1000
) -> Generator[list[dict], None, None]:
    """Lee un archivo CSV de transacciones en chunks para no saturar memoria."""
    with open(archivo) as f:
        reader = csv.DictReader(f)
        chunk = []
        for row in reader:
            chunk.append(row)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:  # Último chunk incompleto
            yield chunk


def rolling_window(
    datos: list[float], window: int
) -> Generator[list[float], None, None]:
    """Genera ventanas deslizantes sobre una serie de datos."""
    ventana = deque(maxlen=window)
    for valor in datos:
        ventana.append(valor)
        if len(ventana) == window:
            yield list(ventana)
