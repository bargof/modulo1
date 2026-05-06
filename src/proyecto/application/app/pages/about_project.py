from __future__ import annotations

import streamlit as st

st.title("ℹ️ Acerca del proyecto")

st.write(
    """
    Este proyecto clasifica el sentimiento de textos financieros usando el dataset
    **Financial PhraseBank**.

    El objetivo es comparar dos enfoques:

    1. **TF-IDF + Logistic Regression**
       Baseline clásico de NLP que representa texto mediante frecuencia ponderada
       de términos.

    2. **Embeddings + PyTorch MLP**
       Modelo que convierte textos en embeddings densos usando Hugging Face y
       entrena una red neuronal en PyTorch.
    """
)

st.subheader("Arquitectura general")

st.code(
    """
Texto financiero
    ↓
Servicio de predicción
    ↓
Modelo seleccionado
    ├── TF-IDF + Logistic Regression
    └── Embeddings + PyTorch MLP
    ↓
Sentimiento + probabilidades
""",
    language="text",
)

st.subheader("Limitaciones")

st.write(
    """
    - El modelo fue entrenado con frases financieras en inglés.
    - El sentimiento financiero no equivale a una recomendación de inversión.
    - Las probabilidades reflejan confianza del modelo, no certeza del mercado.
    - El dataset contiene textos relativamente cortos, por lo que noticias largas
      podrían requerir procesamiento adicional.
    """
)
