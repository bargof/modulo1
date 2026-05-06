from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series


class RawFinancialPhrasebankSchema(pa.DataFrameModel):
    """Esquema para el dataset crudo financial phrasebank.

    Columnas esperadas:
    - sentence: original financial text.
    - label: sentiment label, either numeric or textual depending on the source.
    """

    sentence: Series[str] = pa.Field(nullable=False)
    label: Series[object] = pa.Field(nullable=False)

    class Config:
        strict = False
        coerce = True


class ProcessedFinancialPhrasebankSchema(pa.DataFrameModel):
    """Esquema para el dataset procesado de financial phraasebank.

    Columnas esperadas:
    - text: normalized financial text.
    - label: sentiment label as text.
    - label_id: sentiment label encoded as integer.
    """

    text: Series[str] = pa.Field(nullable=False, str_length={"min_value": 1})
    label: Series[str] = pa.Field(
        nullable=False, isin=["negative", "neutral", "positive"]
    )
    label_id: Series[int] = pa.Field(nullable=False, isin=[0, 1, 2])

    class Config:
        strict = True
        coerce = True


def validate_raw_financial_phrasebank(df: pd.DataFrame) -> pd.DataFrame:
    """Valida el dataset crudo."""
    return RawFinancialPhrasebankSchema.validate(df)


def validate_processed_financial_phrasebank(df: pd.DataFrame) -> pd.DataFrame:
    """Valida el dataset procesado."""
    return ProcessedFinancialPhrasebankSchema.validate(df)
