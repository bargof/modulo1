from __future__ import annotations

import logging

import pandas as pd
from proyecto.domain.schemas import (
    validate_processed_financial_phrasebank,
    validate_raw_financial_phrasebank,
)

logger = logging.getLogger(__name__)


LABEL_ID_TO_NAME = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

LABEL_NAME_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


def preprocess_financial_phrasebank(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesa el dataset raw de Financial PhraseBank.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original descargado de Financial PhraseBank.

    Returns
    -------
    pd.DataFrame
        DataFrame procesado con las columnas: text, label y label_id.
    """
    logger.info("Iniciando preprocesamiento de Financial PhraseBank")

    df = validate_raw_financial_phrasebank(df)
    logger.info("Dataset raw validado correctamente")

    processed_df = df.copy()

    processed_df = _standardize_columns(processed_df)
    processed_df = _standardize_labels(processed_df)
    processed_df = _clean_text_column(processed_df)
    processed_df = _remove_empty_texts(processed_df)
    processed_df = _remove_duplicate_texts(processed_df)

    processed_df = processed_df[["text", "label", "label_id"]]

    processed_df = validate_processed_financial_phrasebank(processed_df)
    logger.info("Dataset procesado validado correctamente")

    logger.info("Preprocesamiento terminado con %s filas", len(processed_df))

    return processed_df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Estandariza los nombres de columnas del dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los datos originales.

    Returns
    -------
    pd.DataFrame
        DataFrame con la columna de texto renombrada como text.
    """
    df = df.copy()

    if "sentence" in df.columns:
        df = df.rename(columns={"sentence": "text"})

    if "text" not in df.columns:
        raise ValueError("El dataset debe contener una columna 'sentence' o 'text'.")

    if "label" not in df.columns:
        raise ValueError("El dataset debe contener una columna 'label'.")

    return df


def _standardize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Estandariza las etiquetas de sentimiento.

    Convierte etiquetas numéricas, strings numéricos o etiquetas textuales a una
    representación consistente con dos columnas: label y label_id.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con una columna label.

    Returns
    -------
    pd.DataFrame
        DataFrame con etiquetas estandarizadas.
    """
    df = df.copy()

    label_values = df["label"].dropna().unique()
    logger.info("Valores únicos originales en label: %s", label_values)

    if pd.api.types.is_numeric_dtype(df["label"]):
        df["label_id"] = df["label"].astype(int)

        invalid_ids = set(df["label_id"].unique()) - set(LABEL_ID_TO_NAME.keys())
        if invalid_ids:
            raise ValueError(f"Hay label_id inválidos en el dataset: {invalid_ids}")

        df["label"] = df["label_id"].map(LABEL_ID_TO_NAME)

    else:
        label_as_text = df["label"].astype(str).str.strip().str.lower()

        if label_as_text.isin(["0", "1", "2"]).all():
            df["label_id"] = label_as_text.astype(int)
            df["label"] = df["label_id"].map(LABEL_ID_TO_NAME)
        else:
            df["label"] = label_as_text
            df["label_id"] = df["label"].map(LABEL_NAME_TO_ID)

    if df["label"].isna().any():
        invalid_ids = df.loc[df["label"].isna(), "label_id"].unique()
        raise ValueError(
            f"Algunos label_id no pudieron mapearse a nombres: {invalid_ids}"
        )

    if df["label_id"].isna().any():
        invalid_labels = df.loc[df["label_id"].isna(), "label"].unique()
        raise ValueError(
            f"Algunas etiquetas no pudieron mapearse a ids numéricos: {invalid_labels}"
        )

    df["label_id"] = df["label_id"].astype(int)

    return df


def _clean_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia mínimamente la columna de texto.

    Por ahora solo elimina espacios al inicio y al final del texto. No se eliminan
    signos de puntuación ni palabras vacías para evitar perder información útil
    para el análisis de sentimiento financiero.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con una columna text.

    Returns
    -------
    pd.DataFrame
        DataFrame con la columna text limpia.
    """
    df = df.copy()

    df["text"] = df["text"].astype(str).str.strip()

    return df


def _remove_empty_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas con textos vacíos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con una columna text.

    Returns
    -------
    pd.DataFrame
        DataFrame sin filas con textos vacíos.
    """
    df = df.copy()

    before = len(df)
    df = df[df["text"].str.len() > 0]
    after = len(df)

    removed = before - after

    if removed > 0:
        logger.warning("Se eliminaron %s filas con texto vacío", removed)

    return df


def _remove_duplicate_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina textos duplicados exactos.

    Esta limpieza reduce el riesgo de data leakage, es decir, que el mismo texto
    aparezca tanto en entrenamiento como en prueba.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con una columna text.

    Returns
    -------
    pd.DataFrame
        DataFrame sin textos duplicados exactos.
    """
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    after = len(df)

    removed = before - after

    if removed > 0:
        logger.warning("Se eliminaron %s textos duplicados", removed)

    return df
