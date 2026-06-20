import json
import os

import pandas as pd
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(ROOT_DIR, "artifacts")


@st.cache_data
def load_csv(filename, **kwargs):
    return pd.read_csv(os.path.join(ARTIFACT_DIR, filename), **kwargs)


@st.cache_data
def load_json(filename):
    with open(os.path.join(ARTIFACT_DIR, filename)) as f:
        return json.load(f)


st.set_page_config(page_title="Flight Delay Dashboard", page_icon="✈️", layout="wide")

st.title("✈️ Predicción de Retrasos de Vuelos")
st.caption(
    "Dataset: 2015 Flight Delays and Cancellations (Kaggle, usdot/flight-delays) · "
    "U.S. Department of Transportation · ~5.8 millones de vuelos domésticos"
)

model_info = load_json("model_info.json")

flights = load_csv("flights_sample.csv")

st.subheader("Resumen del dataset")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Vuelos en la muestra", f"{len(flights):,}")
col2.metric("Tasa de retraso global", f"{flights['IS_DELAYED'].mean():.1%}")
col3.metric("Aerolíneas", flights["AIRLINE"].nunique())
col4.metric("Aeropuertos de origen", flights["ORIGIN_AIRPORT"].nunique())

st.divider()

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Acerca de este proyecto")
    st.markdown(
        """
Este dashboard acompaña un proyecto completo de ciencia de datos sobre retrasos de vuelos
domésticos en EE. UU. (año 2015). El objetivo es predecir — usando solo información disponible
**antes del despegue** — si un vuelo se retrasará más de 15 minutos, y estimar cuántos minutos
exactos de retraso tendrá.

El pipeline cubre comprensión y limpieza de datos, análisis exploratorio (EDA), ingeniería
de variables sin *data leakage*, entrenamiento y evaluación de 3 modelos de clasificación,
interpretación de resultados y recomendaciones de negocio, más un modelo de regresión (bonus).

Usa el menú de la izquierda para navegar:
- **📊 EDA** — visualizaciones interactivas con filtros por aerolínea y mes.
- **🔮 Predictor** — ingresa los datos de un vuelo y obtén una predicción en vivo.
- **📈 Model Performance** — tablas comparativas e importancia de variables.
        """
    )

with col_right:
    st.subheader("Resultados del modelo")
    st.info(
        f"**Mejor clasificador:** {model_info['best_classifier']}\n\n"
        f"**Mejor regresor:** {model_info['best_regressor']}"
    )
    st.markdown("""
| Métrica | Valor |
|---|---|
| ROC-AUC (Random Forest) | **0.803** |
| Accuracy | 74.3% |
| F1-score | 67.4% |
| MAE regresión | 9.9 min |
""")
    st.caption("Entrenado sobre el dataset real de Kaggle (2015 Flight Delays).")
