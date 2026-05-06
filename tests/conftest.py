from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_raw_financial_phrasebank_df() -> pd.DataFrame:
    """Crea un dataset raw pequeño para pruebas."""
    return pd.DataFrame(
        {
            "sentence": [
                "The company reported higher profits.",
                "Sales declined compared with last year.",
                "The company announced its annual meeting.",
            ],
            "label": [2, 0, 1],
        }
    )


@pytest.fixture
def sample_processed_financial_phrasebank_df() -> pd.DataFrame:
    """Crea un dataset procesado pequeño para pruebas."""
    return pd.DataFrame(
        {
            "text": [
                "The company reported higher profits.",
                "Sales declined compared with last year.",
                "The company announced its annual meeting.",
            ],
            "label": ["positive", "negative", "neutral"],
            "label_id": [2, 0, 1],
        }
    )
