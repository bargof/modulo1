from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st
from proyecto.config.settings import settings


@st.cache_data
def load_model_metrics() -> pd.DataFrame:
    """Carga las métricas comparativas de los modelos."""
    metrics_path = settings.reports_dir / "model_metrics.json"

    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    return pd.DataFrame(metrics)


st.set_page_config(
    page_title="Comparación de modelos",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Comparación de modelos")

st.write(
    """
    Esta sección compara el desempeño de los modelos entrenados para clasificación
    de sentimiento financiero.
    """
)

metrics_df = load_model_metrics()

# st.dataframe(metrics_df, use_container_width=True)
st.dataframe(
    metrics_df[["model", "accuracy", "macro_f1", "weighted_f1"]],
    use_container_width=True,
)

metrics_long_df = metrics_df.melt(
    id_vars="model",
    value_vars=["accuracy", "macro_f1", "weighted_f1"],
    var_name="metric",
    value_name="score",
)

fig = px.bar(
    metrics_long_df,
    x="model",
    y="score",
    color="metric",
    barmode="group",
    title="Comparación de métricas por modelo",
    labels={
        "model": "Modelo",
        "score": "Score",
        "metric": "Métrica",
    },
    range_y=[0, 1],
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    """
    La métrica principal para comparar modelos es **macro F1**, ya que el dataset
    presenta cierto desbalance entre clases.
    """
)


def render_confusion_matrix(matrix: list[list[int]], model_name: str) -> None:
    """Muestra una matriz de confusión como heatmap."""
    labels = ["negative", "neutral", "positive"]

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    fig = px.imshow(
        matrix_df,
        text_auto=True,
        title=f"Matriz de confusión — {model_name}",
        labels={
            "x": "Predicción",
            "y": "Clase real",
            "color": "Conteo",
        },
    )

    st.plotly_chart(fig, use_container_width=True)


st.subheader("Matrices de confusión")

for record in metrics_df.to_dict(orient="records"):
    st.write(f"### {record['model']}")
    render_confusion_matrix(
        matrix=record["confusion_matrix"],
        model_name=record["model"],
    )
