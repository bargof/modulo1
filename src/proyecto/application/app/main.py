from __future__ import annotations

from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent


def main() -> None:
    """Inicializa la navegación multipágina de Streamlit."""
    prediction_page = st.Page(
        APP_DIR / "streamlit_app.py",
        title="Probar predicción",
        icon="📈",
        default=True,
    )

    model_comparison_page = st.Page(
        APP_DIR / "pages" / "comparacion_modelos.py",
        title="Comparación de modelos",
        icon="📊",
    )

    dataset_page = st.Page(
        APP_DIR / "pages" / "dataset_exploration.py",
        title="Dataset y exploración",
        icon="🧪",
    )

    about_page = st.Page(
        APP_DIR / "pages" / "about_project.py",
        title="Acerca del proyecto",
        icon="ℹ️",
    )

    navigation = st.navigation(
        {
            "Aplicación": [prediction_page],
            "Análisis": [model_comparison_page, dataset_page],
            "Documentación": [about_page],
        }
    )

    navigation.run()
