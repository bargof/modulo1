from __future__ import annotations

import pandas as pd
from pandera import Check, Column, DataFrameSchema

# ═══════════════════════════════════════
# Schema raw = contrato del dataset crudo
# ═══════════════════════════════════════

raw_financial_phrasebank_schema = DataFrameSchema(
    columns={
        "sentence": Column(
            str,
            nullable=False,
            checks=Check.str_length(
                min_value=1,
                error="La columna sentence no puede tener textos vacíos.",
            ),
            description="Texto financiero original.",
        ),
        "label": Column(
            object,
            nullable=False,
            description="Etiqueta original de sentimiento.",
        ),
    },
    strict=False,
    coerce=True,
)


# ═══════════════════════════════════════
# Schema processed = contrato del dataset procesado
# ═══════════════════════════════════════

processed_financial_phrasebank_schema = DataFrameSchema(
    columns={
        "text": Column(
            str,
            nullable=False,
            checks=Check.str_length(
                min_value=1,
                error="La columna text no puede tener textos vacíos.",
            ),
            description="Texto financiero limpio y normalizado.",
        ),
        "label": Column(
            str,
            nullable=False,
            checks=Check.isin(
                ["negative", "neutral", "positive"],
                error="La etiqueta debe ser negative, neutral o positive.",
            ),
            description="Etiqueta de sentimiento en formato texto.",
        ),
        "label_id": Column(
            int,
            nullable=False,
            checks=Check.isin(
                [0, 1, 2],
                error="label_id debe ser 0, 1 o 2.",
            ),
            description="Etiqueta de sentimiento codificada como entero.",
        ),
    },
    checks=[
        Check(
            lambda df: df.duplicated().sum() == 0,
            error="El DataFrame procesado tiene filas duplicadas.",
        ),
        Check(
            lambda df: (
                ((df["label"] == "negative") & (df["label_id"] == 0))
                | ((df["label"] == "neutral") & (df["label_id"] == 1))
                | ((df["label"] == "positive") & (df["label_id"] == 2))
            ).all(),
            error="La relación entre label y label_id no es consistente.",
        ),
    ],
    strict=True,
    coerce=True,
)


def validate_raw_financial_phrasebank(df: pd.DataFrame) -> pd.DataFrame:
    """Valida el dataset crudo de Financial PhraseBank."""
    return raw_financial_phrasebank_schema.validate(df)


def validate_processed_financial_phrasebank(df: pd.DataFrame) -> pd.DataFrame:
    """Valida el dataset procesado de Financial PhraseBank."""
    return processed_financial_phrasebank_schema.validate(df)
