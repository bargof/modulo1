from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from proyecto.config.settings import settings


@st.cache_data
def load_processed_dataset() -> pd.DataFrame:
    """Carga el dataset procesado."""
    dataset_path = settings.processed_data_dir / "financial_phrasebank_processed.csv"
    return pd.read_csv(dataset_path)


st.set_page_config(
    page_title="Dataset y exploración",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Dataset y exploración")

df = load_processed_dataset()

st.write(
    """
    Esta sección muestra una exploración breve del dataset procesado usado para
    entrenar los modelos.
    """
)

st.subheader("Vista previa")

st.dataframe(df.head(20), use_container_width=True)

st.subheader("Distribución de clases")

class_counts = df["label"].value_counts().reset_index()
class_counts.columns = ["label", "count"]

fig = px.bar(
    class_counts,
    x="label",
    y="count",
    title="Distribución de clases de sentimiento",
    labels={
        "label": "Sentimiento",
        "count": "Número de ejemplos",
    },
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Longitud de textos")

df["word_count"] = df["text"].astype(str).str.split().str.len()

fig = px.histogram(
    df,
    x="word_count",
    nbins=30,
    title="Distribución de longitud de textos",
    labels={
        "word_count": "Número de palabras",
    },
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Ejemplos por clase")

selected_label = st.selectbox(
    "Selecciona una clase",
    sorted(df["label"].unique()),
)

st.dataframe(
    df[df["label"] == selected_label][["text", "label"]].sample(
        min(10, len(df[df["label"] == selected_label])),
        random_state=42,
    ),
    use_container_width=True,
)
