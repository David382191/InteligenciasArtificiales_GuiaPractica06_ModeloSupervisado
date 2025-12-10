import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

st.set_page_config(page_title="Modelo Supervisado y Clustering", layout="wide")

# ===============================================================
# FUNCIONES AUXILIARES
# ===============================================================

def preparar_datos(df):
    """Limpia los datos y crea la variable objetivo 'aprobado'."""
    df = df.copy()
    
    # Eliminar duplicados
    df = df.drop_duplicates()
    
    # Eliminar nulos críticos
    df = df.dropna(subset=["nota_final", "asistencia"])
    
    # Crear variable objetivo
    df["aprobado"] = (df["nota_final"] >= 11).astype(int)
    
    # Seleccionar columnas numéricas y categóricas
    num_cols = ["nota_final", "asistencia"]
    cat_cols = [c for c in df.columns if c not in num_cols + ["aprobado", "id", "fecha", "nombre"]]

    return df, num_cols, cat_cols


def entrenar_modelo(df, test_size=0.2, C_val=1.0):
    """Entrena la regresión logística con pipeline."""
    df, num_cols, cat_cols = preparar_datos(df)

    X = df[num_cols + cat_cols]
    y = df["aprobado"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # Preprocesamiento
    numerical = Pipeline([("scaler", StandardScaler())])
    categorical = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical, num_cols),
            ("cat", categorical, cat_cols)
        ]
    )

    modelo = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(C=C_val, max_iter=1000))
    ])

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    return modelo, X_test, y_test, y_pred


def entrenar_clustering(df, k):
    """Entrena K-means y devuelve labels y centroides."""
    df, _, _ = preparar_datos(df)

    X = df[["asistencia", "nota_final"]]

    kmeans = KMeans(n_clusters=k, random_state=42)
    df["cluster"] = kmeans.fit_predict(X)

    return df, kmeans


# ===============================================================
# INTERFAZ STREAMLIT
# ===============================================================

st.title("📘 Aplicación Machine Learning con Streamlit")
st.write("Modelo Supervisado (Clasificación) + Modelo No Supervisado (Clustering)")

st.markdown("---")

# ===========================================
# CARGA DEL DATASET
# ===========================================
uploaded = st.file_uploader("Sube el archivo academic_performance_master.csv", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)

    st.subheader("📊 Vista previa del dataset")
    st.dataframe(df.head())

    st.markdown("---")

    # ===============================================================
    # SECCIÓN 1: MODELO SUPERVISADO
    # ===============================================================
    st.header("🟦 MODELO SUPERVISADO – Clasificación (Aprobado/Reprobado)")

    test_size = st.slider("Porcentaje para Test (%)", 10, 40, 20) / 100
    C_val = st.number_input("Valor de regularización (C) en Regresión Logística", 0.01, 10.0, 1.0)

    if st.button("Entrenar Modelo Supervisado"):
        modelo, X_test, y_test, y_pred = entrenar_modelo(df, test_size, C_val)

        accuracy = accuracy_score(y_test, y_pred)

        st.success(f"Modelo entrenado correctamente. **Accuracy: {accuracy:.3f}**")

        # Mostrar reporte
        st.subheader("Reporte de Clasificación:")
        st.text(classification_report(y_test, y_pred))

        # Matriz de confusión
        st.subheader("Matriz de Confusión:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        # Guardar modelo
        joblib.dump(modelo, "modelo_supervisado.joblib")
        st.info("Modelo guardado como modelo_supervisado.joblib")

    st.markdown("---")

    # ===============================================================
    # SECCIÓN 2: CLUSTERING
    # ===============================================================
    st.header("🟩 MODELO NO SUPERVISADO – K-means Clustering")

    k = st.slider("Número de Clusters (k)", 2, 4, 3)

    if st.button("Ejecutar Clustering"):
        df_cluster, kmeans = entrenar_clustering(df, k)

        st.success("Clustering completado.")

        st.subheader("Gráfico de Clusters")
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=df_cluster["asistencia"],
            y=df_cluster["nota_final"],
            hue=df_cluster["cluster"],
            palette="viridis",
            ax=ax
        )

        # Centroides
        centroids = kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c="red", label="Centroides")
        plt.legend()
        plt.xlabel("Asistencia")
        plt.ylabel("Nota Final")

        st.pyplot(fig)

        st.subheader("Centroides del modelo:")
        st.write(pd.DataFrame(centroids, columns=["Asistencia", "Nota Final"]))

else:
    st.info("Sube el archivo CSV para comenzar.")
