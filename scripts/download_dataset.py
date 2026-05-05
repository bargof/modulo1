from pathlib import Path

import pandas as pd
from datasets import load_dataset

RAW_DATA_DIR = Path("data/raw")
OUTPUT_PATH = RAW_DATA_DIR / "financial_phrasebank.csv"


def download_phrasebank() -> pd.DataFrame:
    """
    Descarga el dataset Financial PhraseBank desde Hugging Face,
    lo convierte a pandas y lo guarda como CSV en data/raw/.
    """

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "takala/financial_phrasebank",
        "sentences_50agree",
        trust_remote_code=True,
    )
    df = dataset["train"].to_pandas()
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print("Dataset descargado correctamente.")
    print(f"Filas: {len(df)}")
    print(f"Columnas: {list(df.columns)}")
    print(f"Guardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    download_phrasebank()
