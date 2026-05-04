import logging
import sys

# ═══════════════════════════════════════════
# Configuración de logging profesional
# ═══════════════════════════════════════════


def setup_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """
    Configura logging para la aplicación.

    Parameters
    ----------
    level : str
        Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : str, optional
        Ruta al archivo de log. Si None, solo imprime a consola.

    Returns
    -------
    logging.Logger
        Logger configurado para la aplicación.
    """
    # Formato: timestamp | nivel | módulo | mensaje
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    # Crear el logger raíz de nuestra aplicación
    # Todos los loggers hijos (pricing_derivados.data.loader, etc.)
    # heredan esta configuración automáticamente
    logger = logging.getLogger("pfinal")
    logger.setLevel(getattr(logging, level.upper()))

    # Evitar duplicar handlers si ejecutamos la celda múltiples veces
    if logger.handlers:
        logger.handlers.clear()

    # Handler de consola: imprime a stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    logger.addHandler(console_handler)

    # Handler de archivo (opcional): guarda en disco
    if log_file:
        from pathlib import Path

        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
        logger.addHandler(file_handler)

    return logger


# Instanciacion:
# setup_logging(level=settings.log_level, log_file=str(settings.log_file))
# Logger por modulo: logger = logging.getLogger(__name__)
