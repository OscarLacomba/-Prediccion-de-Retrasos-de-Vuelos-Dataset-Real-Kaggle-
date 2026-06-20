# -Prediccion-de-Retrasos-de-Vuelos-Dataset-Real-Kaggle-
Data Science Work
Analyze historical flight data, identify the factors that contribute to delays, and develop a machine learning model capable of predicting whether a flight will be delayed.
# ✈️ Predicción de Retrasos de Vuelos

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/osky9/Predicting-Flight-Delays)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange)](https://scikit-learn.org/)

Proyecto completo de ciencia de datos para predecir retrasos en vuelos domésticos de EE. UU. — desde la limpieza de datos hasta un dashboard interactivo en producción.

**Dataset:** [2015 Flight Delays and Cancellations](https://www.kaggle.com/datasets/usdot/flight-delays) · Kaggle, `usdot/flight-delays` · Fuente: U.S. DOT, Bureau of Transportation Statistics

🔗 **[Demo en vivo → Hugging Face Space](https://huggingface.co/spaces/osky9/Predicting-Flight-Delays)**

---

## 📋 Tabla de Contenidos

- [Descripción del proyecto](#-descripción-del-proyecto)
- [Dataset](#-dataset)
- [Estructura del repositorio](#-estructura-del-repositorio)
- [Metodología](#-metodología)
- [Resultados](#-resultados)
- [Dashboard](#-dashboard)
- [Cómo correrlo localmente](#-cómo-correrlo-localmente)
- [Tech stack](#-tech-stack)
- [Recomendaciones de negocio](#-recomendaciones-de-negocio)
- [Próximos pasos](#-próximos-pasos)

---

## 🎯 Descripción del proyecto

¿Se puede predecir si un vuelo va a salir tarde — usando solo la información disponible al momento de reservar? Este proyecto responde esa pregunta con un pipeline completo de Machine Learning:

- **Clasificación:** predice si un vuelo se retrasará más de 15 minutos (IS_DELAYED = 1/0).
- **Regresión (bonus):** estima cuántos minutos exactos de retraso tendrá el vuelo.

Se cubre el ciclo completo: comprensión de datos, limpieza, EDA, ingeniería de variables (con manejo explícito de *data leakage*), entrenamiento de 3 modelos, evaluación, interpretación y recomendaciones de negocio.

---

## 📊 Dataset

| Atributo | Detalle |
|---|---|
| Fuente | [usdot/flight-delays en Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays) |
| Vuelos | ~5.8 millones de vuelos domésticos de EE. UU., año 2015 |
| Archivos | `flights.csv`, `airlines.csv` (14 aerolíneas), `airports.csv` (322 aeropuertos) |
| Variable objetivo | `IS_DELAYED` = 1 si `DEPARTURE_DELAY` > 15 min |
| Tasa de retraso global | **39.0%** de los vuelos |

> ⚠️ `flights.csv` (~580 MB) no está incluido en este repositorio por su tamaño. Ver instrucciones de descarga en [Cómo correrlo localmente](#-cómo-correrlo-localmente).

---

## 📁 Estructura del repositorio

```
.
├── app.py                              # Dashboard — página Overview (entry point)
├── pages/
│   ├── 1_📊_EDA.py                     # Visualizaciones interactivas con filtros
│   ├── 2_🔮_Predictor.py               # Predicción en vivo (clasificación + regresión)
│   └── 3_📈_Model_Performance.py       # Métricas e interpretación
├── artifacts/                          # Modelos entrenados y datos para el dashboard
│   ├── best_classifier.joblib          # Mejor modelo de clasificación (comprimido LZMA)
│   ├── best_regressor.joblib           # Mejor modelo de regresión
│   ├── model_info.json                 # Metadata de los modelos
│   ├── classification_results.csv      # Tabla de métricas de clasificación
│   ├── regression_results.csv          # Tabla de métricas de regresión
│   ├── feature_importance.csv          # Importancia de variables
│   ├── lookup_tables.json              # Tasas históricas para el Predictor
│   ├── airlines.csv                    # Nombres de aerolíneas
│   ├── airports.csv                    # Nombres de aeropuertos
│   └── flights_sample.csv              # Muestra de 20k vuelos para el EDA del dashboard
├── notebooks/
│   └── flight_delay_prediction.ipynb   # Notebook completo (Tasks 1-9 + bonus regresión)
├── scripts/
│   └── export_artifacts.py             # Exportar artifacts/ desde el notebook ejecutado
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔬 Metodología

| Task | Contenido |
|---|---|
| 1. Data Understanding | Shape, tipos, valores faltantes, análisis de variables |
| 2. Data Cleaning | Deduplicación, cancelados/desviados, parsing de fechas, outliers |
| 3. EDA | 6 visualizaciones: retraso por aerolínea, aeropuerto, mes, hora, día y distribución |
| 4. Feature Engineering | Variables temporales, de ruta e históricas (sin data leakage) |
| 5. Define Target | IS_DELAYED, análisis de desbalance de clases |
| 6. Build Models | Logistic Regression, Random Forest, XGBoost |
| 7. Evaluate | Accuracy, Precision, Recall, F1, ROC-AUC, Matriz de confusión, Curva ROC |
| 8. Interpretation | Feature importance + Permutation importance |
| 9. Recommendations | Para aerolíneas, aeropuertos y viajeros |
| Bonus | Regresión: Linear Regression, RF Regressor, XGBoost Regressor |

**Decisión clave — sin data leakage:** las variables que solo se conocen después del despegue (`ARRIVAL_DELAY`, `TAXI_OUT`, `AIR_TIME`, causas de retraso, etc.) se excluyen completamente. Las variables históricas se calculan únicamente con el set de entrenamiento.

---

## 📈 Resultados

### Clasificación — ¿Se retrasa el vuelo? (> 15 min)

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.716 | 0.616 | 0.722 | 0.665 | 0.786 |
| **Random Forest** | **0.743** | **0.667** | 0.682 | **0.674** | **0.803** |
| XGBoost | 0.734 | 0.653 | 0.680 | 0.666 | 0.791 |

→ **Mejor modelo: Random Forest** (ROC-AUC 0.803)

### Regresión (bonus) — Minutos exactos de retraso

| Modelo | MAE (min) | RMSE (min) | R² |
|---|---|---|---|
| Linear Regression | 10.236 | 12.964 | 0.349 |
| **Random Forest Regressor** | **9.914** | **12.560** | **0.389** |
| XGBoost Regressor | 10.019 | 12.694 | 0.376 |

→ **Mejor modelo: Random Forest Regressor** (R² 0.389, MAE 9.9 min)

### Hallazgos clave del EDA

- **39.0%** de los vuelos analizados se retrasaron más de 15 minutos.
- **American Airlines Inc.** tuvo la mayor tasa de retraso (67.4%); **Southwest Airlines Co.** la menor (15.7%).
- **Agosto** fue el mes con más retrasos (54.9%); el verano en general concentra la mayor demanda.
- Las **21:00** fue la hora de salida con mayor tasa de retraso (73.2%), reflejando la acumulación a lo largo del día operativo.
- Las variables más influyentes fueron: **hora de salida**, **tasa histórica de la aerolínea** y **temporada**.

---

## 🖥️ Dashboard

App de Streamlit con 4 páginas, desplegada en Hugging Face Spaces:

| Página | Contenido |
|---|---|
| Overview | Resumen del proyecto, dataset y estadísticas clave |
| EDA | Visualizaciones interactivas con filtros por aerolínea, aeropuerto y mes |
| Predictor | Ingresa un vuelo y obtén la probabilidad de retraso y los minutos estimados |
| Model Performance | Tablas comparativas e importancia de variables |

---

## 🚀 Cómo correrlo localmente

```bash
# 1. Clonar el repositorio
git clone https://github.com/osky9/Predicting-Flight-Delays.git
cd Predicting-Flight-Delays

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Correr el dashboard
streamlit run app.py
```

Los artefactos ya están incluidos en `artifacts/`, así que el dashboard funciona sin necesidad del dataset completo.

### Para correr el notebook con el dataset real

1. Descarga `flights.csv`, `airlines.csv` y `airports.csv` desde [Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays) (requiere cuenta gratuita).
2. Sube los 3 archivos a Google Drive o directamente al panel de archivos de Google Colab.
3. Abre `notebooks/flight_delay_prediction.ipynb` en Colab y ajusta `DATA_DIR` al path donde subiste los archivos.
4. Corre todas las celdas. Al final, ejecuta la celda de exportación para generar `artifacts.zip` con tus propios modelos entrenados.

---

## 🛠️ Tech stack

| Librería | Uso |
|---|---|
| pandas, numpy | Manipulación y análisis de datos |
| matplotlib, seaborn | Visualizaciones en el notebook |
| scikit-learn 1.6.1 | Preprocesamiento, modelos, evaluación |
| XGBoost 2.1.0 | Clasificación y regresión con gradient boosting |
| Streamlit 1.58 | Dashboard interactivo |
| Plotly | Gráficas interactivas en el dashboard |
| joblib | Serialización de modelos (compresión LZMA) |
| Google Colab | Entorno de entrenamiento |

---

## 💡 Recomendaciones de negocio

**Aerolíneas:** ampliar los tiempos de holgura en vuelos de tarde/noche, donde se acumula el efecto en cascada de retrasos previos. Auditar procesos de *turnaround* en las aerolíneas con mayor tasa histórica.

**Aeropuertos:** reforzar personal de pista y control de tráfico entre las 15:00 y las 21:00, y durante los meses de verano y diciembre. Los aeropuertos ATL, DEN e IAH mostraron las tasas de retraso más altas entre los de mayor volumen.

**Viajeros:** preferir vuelos antes de las 10 AM (menor acumulación de retrasos) y aerolíneas con historial de puntualidad. En meses pico, agregar más margen entre conexiones.

---

## 🔭 Próximos pasos

- Incorporar datos de clima por aeropuerto y fecha (una de las causas principales de retraso no capturada en el dataset).
- Explorar SHAP para interpretación a nivel de predicción individual.
- Reentrenar periódicamente con datos de años más recientes.
- Agregar monitoreo de drift del modelo en producción.

---

## 📄 Licencia y fuente de datos

Datos originales publicados por el U.S. Department of Transportation (dominio público), distribuidos en Kaggle como [usdot/flight-delays](https://www.kaggle.com/datasets/usdot/flight-delays). Código bajo licencia MIT.
