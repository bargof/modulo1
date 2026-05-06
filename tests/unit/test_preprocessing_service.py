from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError
from proyecto.application.services.dataset_preprocessing_service import (
    preprocess_financial_phrasebank,
)


def test_preprocess_financial_phrasebank_returns_expected_columns(
    sample_raw_financial_phrasebank_df: pd.DataFrame,
) -> None:
    """Valida que el preprocesamiento regrese las columnas esperadas."""
    processed_df = preprocess_financial_phrasebank(sample_raw_financial_phrasebank_df)

    assert processed_df.columns.tolist() == ["text", "label", "label_id"]


def test_preprocess_financial_phrasebank_maps_numeric_labels(
    sample_raw_financial_phrasebank_df: pd.DataFrame,
) -> None:
    """Valida que las etiquetas numéricas se conviertan correctamente."""
    processed_df = preprocess_financial_phrasebank(sample_raw_financial_phrasebank_df)

    expected_labels = {"positive", "negative", "neutral"}
    expected_label_ids = {0, 1, 2}

    assert set(processed_df["label"]) == expected_labels
    assert set(processed_df["label_id"]) == expected_label_ids


def test_preprocess_financial_phrasebank_renames_sentence_to_text(
    sample_raw_financial_phrasebank_df: pd.DataFrame,
) -> None:
    """Valida que la columna sentence se renombre como text."""
    processed_df = preprocess_financial_phrasebank(sample_raw_financial_phrasebank_df)

    assert "text" in processed_df.columns
    assert "sentence" not in processed_df.columns


def test_preprocess_financial_phrasebank_removes_empty_texts() -> None:
    """Valida que se eliminen textos vacíos."""
    raw_df = pd.DataFrame(
        {
            "sentence": [
                "The company reported higher profits.",
                "   ",
                "",
            ],
            "label": [2, 1, 0],
        }
    )

    processed_df = preprocess_financial_phrasebank(raw_df)

    assert len(processed_df) == 1
    assert processed_df.iloc[0]["text"] == "The company reported higher profits."


def test_preprocess_financial_phrasebank_removes_duplicates() -> None:
    """Valida que se eliminen textos duplicados exactos."""
    raw_df = pd.DataFrame(
        {
            "sentence": [
                "The company reported higher profits.",
                "The company reported higher profits.",
                "Sales declined compared with last year.",
            ],
            "label": [2, 2, 0],
        }
    )

    processed_df = preprocess_financial_phrasebank(raw_df)

    assert len(processed_df) == 2
    assert processed_df["text"].duplicated().sum() == 0


def test_preprocess_financial_phrasebank_raises_error_without_text_column() -> None:
    """Valida que falle si no existe columna sentence ni text."""
    raw_df = pd.DataFrame(
        {
            "wrong_column": ["The company reported higher profits."],
            "label": [2],
        }
    )

    with pytest.raises(SchemaError, match="sentence"):
        preprocess_financial_phrasebank(raw_df)


def test_preprocess_financial_phrasebank_raises_error_without_label_column() -> None:
    """Valida que falle si no existe la columna label."""
    raw_df = pd.DataFrame(
        {
            "sentence": ["The company reported higher profits."],
        }
    )

    with pytest.raises(SchemaError, match="label"):
        preprocess_financial_phrasebank(raw_df)
