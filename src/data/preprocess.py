"""
Módulo de Preprocesamiento de Datos para Predicción de Menú y Abasto por Estaciones del Año.
Proyecto Final - Fundamentos de Inteligencia Artificial (Entrega 3).
"""

import os
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Ruta absoluta obligatoria solicitada por especificación técnica
DEFAULT_DATASET_PATH = r"C:\Users\villa\Downloads\dataset_limpio.csv"
FALLBACK_RELATIVE_PATH = os.path.join("data", "processed", "dataset_limpio.csv")

# Mapeo de estaciones del año (Hemisferio Norte)
SEASON_NAMES = {
    0: "Invierno",
    1: "Primavera",
    2: "Verano",
    3: "Otoño"
}

SEASON_FROM_NAME = {
    "invierno": 0,
    "winter": 0,
    "primavera": 1,
    "spring": 1,
    "verano": 2,
    "summer": 2,
    "otono": 3,
    "otoño": 3,
    "autumn": 3,
    "fall": 3
}

# Mapeo de temperatura de bebida
BEVERAGE_TEMP_MAP = {
    "Hot": 0,
    "Not a Beverage": 1,
    "Cold": 2
}

# Columnas predictoras (Features) utilizadas por el modelo Random Forest
FEATURE_COLUMNS = [
    "estacion_id",
    "month",
    "day_of_week",
    "hour",
    "store_id",
    "unit_price",
    "transaction_qty",
    "revenue",
    "beverage_temp_enc"
]

TARGET_COLUMN = "product_category"


def obtener_estacion_id(mes: int) -> int:
    """Convierte el número de mes (1-12) en el identificador de estación del año."""
    if mes in (12, 1, 2):
        return 0  # Invierno
    elif mes in (3, 4, 5):
        return 1  # Primavera
    elif mes in (6, 7, 8):
        return 2  # Verano
    return 3      # Otoño


def resolver_ruta_dataset(ruta_solicitada: str = DEFAULT_DATASET_PATH) -> str:
    """
    Resuelve la ruta del dataset priorizando la ruta indicada (por defecto
    C:\\Users\\villa\\Downloads\\dataset_limpio.csv) o la ruta relativa del repositorio.
    """
    if ruta_solicitada and os.path.exists(ruta_solicitada):
        return ruta_solicitada
    if os.path.exists(DEFAULT_DATASET_PATH):
        return DEFAULT_DATASET_PATH
    if os.path.exists(FALLBACK_RELATIVE_PATH):
        return FALLBACK_RELATIVE_PATH
    raise FileNotFoundError(
        f"No se encontró el archivo de datos en '{ruta_solicitada}' ni en '{DEFAULT_DATASET_PATH}'."
    )


def cargar_datos(ruta_csv: str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Carga el dataset limpio usando Pandas desde la ruta configurada."""
    ruta_final = resolver_ruta_dataset(ruta_csv)
    df = pd.read_csv(ruta_final)
    return df


def transformar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae características temporales y estacionales para predecir la categoría
    del menú demandada según la temporada del año, horario y sucursal.
    """
    df_proc = df.copy()

    # Conversión de fecha y extracción de variables temporales/estacionales
    fechas = pd.to_datetime(df_proc["transaction_date"])
    df_proc["month"] = fechas.dt.month
    df_proc["day_of_week"] = fechas.dt.day_of_week
    df_proc["estacion_id"] = df_proc["month"].apply(obtener_estacion_id)
    df_proc["estacion_nombre"] = df_proc["estacion_id"].map(SEASON_NAMES)

    # Extracción de la hora de transacción
    horas = pd.to_datetime(df_proc["transaction_time"], format="%H:%M:%S", errors="coerce")
    df_proc["hour"] = horas.dt.hour.fillna(12).astype(int)

    # Codificación de la temperatura del producto/bebida
    df_proc["beverage_temp_enc"] = (
        df_proc["beverage_temp"].map(BEVERAGE_TEMP_MAP).fillna(0).astype(int)
    )

    return df_proc


def ejecutar_pipeline_preprocesamiento(
    ruta_csv: str = DEFAULT_DATASET_PATH,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, LabelEncoder, Dict[str, Any]]:
    """
    Ejecuta el pipeline completo de carga, ingeniería de características estacionales,
    división estratificada Train/Test (80/20) y escalamiento con StandardScaler.
    """
    df_raw = cargar_datos(ruta_csv)
    df_proc = transformar_variables(df_raw)

    X = df_proc[FEATURE_COLUMNS].copy()
    y_raw = df_proc[TARGET_COLUMN].values

    # Codificación de la variable objetivo (Categorías del Menú para Abasto)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # División estratificada 80% Entrenamiento / 20% Prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Escalamiento estándar ajustado únicamente sobre el conjunto de entrenamiento
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Cálculo de estadísticas de demanda por estación y categoría para recomendaciones de abasto
    resumen_estacional = (
        df_proc.groupby(["estacion_nombre", TARGET_COLUMN])["transaction_qty"]
        .sum()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "classes": label_encoder.classes_.tolist(),
        "season_names": SEASON_NAMES,
        "beverage_temp_map": BEVERAGE_TEMP_MAP,
        "total_samples": len(df_proc),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "seasonal_demand_summary": resumen_estacional
    }

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder, metadata


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, sc, le, meta = ejecutar_pipeline_preprocesamiento()
    print(f"[INFO] Preprocesamiento exitoso. Total registros: {meta['total_samples']:,}")
    print(f"[INFO] Train: {meta['train_samples']:,} | Test: {meta['test_samples']:,}")
    print(f"[INFO] Categorías del menú ({len(meta['classes'])}): {meta['classes']}")
