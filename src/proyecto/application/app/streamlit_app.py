from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from proyecto.application.services.prediction_service import SentimentPredictionService
from proyecto.config.settings import settings
from proyecto.models.mlp_model import EmbeddingMLPSentimentModel
from proyecto.models.tfidf_logistic_model import TfidfLogisticSentimentModel


@st.cache_resource
def load_tfidf_model() -> TfidfLogisticSentimentModel:
    """Carga el modelo TF-IDF entrenado."""
    model_path = settings.models_dir / "tfidf_logistic_model.pkl"

    model = TfidfLogisticSentimentModel()
    model.load(str(model_path))

    return model


@st.cache_resource
def load_mlp_model() -> EmbeddingMLPSentimentModel:
    """Carga el modelo de embeddings + MLP entrenado."""
    model_path = settings.models_dir / "embedding_mlp_model.pt"

    model = EmbeddingMLPSentimentModel()
    model.load(str(model_path))

    return model


def get_model(model_name: str):
    """Regresa el modelo seleccionado por el usuario."""
    if model_name == "TF-IDF + Logistic Regression":
        return load_tfidf_model()

    return load_mlp_model()


def render_probability_chart(probabilities: dict[str, float]) -> None:
    """Muestra una gráfica de barras con probabilidades por clase."""
    probability_df = pd.DataFrame(
        {
            "sentiment": list(probabilities.keys()),
            "probability": list(probabilities.values()),
        }
    )

    fig = px.bar(
        probability_df,
        x="sentiment",
        y="probability",
        title="Probabilidades por clase",
        labels={
            "sentiment": "Sentimiento",
            "probability": "Probabilidad",
        },
        range_y=[0, 1],
    )

    st.plotly_chart(fig, use_container_width=True)


st.title("Clasificación de Sentimiento de Noticias Financieras")

st.write(
    """
    Clasifica textos financieros como **negative**, **neutral** o **positive**.
    """
)

model_name = st.selectbox(
    "Selecciona el modelo",
    [
        "TF-IDF + Logistic Regression",
        "Embeddings + PyTorch MLP",
    ],
)

user_text = st.text_area(
    "Escribe una frase o noticia financiera",
    value="The company reported higher profits and strong revenue growth.",
    height=140,
)

if st.button("Clasificar sentimiento"):
    if not user_text.strip():
        st.warning("Escribe un texto antes de clasificar.")
        st.stop()

    model = get_model(model_name)
    prediction_service = SentimentPredictionService(model=model)
    result = prediction_service.predict_text(user_text)

    st.divider()
    st.subheader("Resultado")

    sentiment = result.predicted_label
    delta_value = {
        "positive": "+1 Positivo",
        "neutral": "0 Neutral",
        "negative": "-1 Negativo",
    }[sentiment]

    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            border=True,
            label="Predicción:",
            value="",
            delta=delta_value,
        )
    with c2:
        st.dataframe(
            pd.DataFrame(
                {
                    "sentiment": list(result.probabilities.keys()),
                    "probability": list(result.probabilities.values()),
                }
            ),
            use_container_width=True,
        )

    st.write("### Probabilidades")
    render_probability_chart(result.probabilities)
