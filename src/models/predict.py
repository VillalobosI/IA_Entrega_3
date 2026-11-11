"""
Módulo de Inferencia Desacoplada (Contrato de Interfaz para Entrega 4 - Dashboard).
Predicción de Menú y Abasto de Inventario por Estaciones del Año.
Proyecto Final - Fundamentos de Inteligencia Artificial (Entrega 3).
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocess import (
    FEATURE_COLUMNS,
    SEASON_NAMES,
    SEASON_FROM_NAME,
    BEVERAGE_TEMP_MAP,
    obtener_estacion_id
)

# Rutas de los artefactos serializados
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "modelo_candidato.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
META_PATH = os.path.join(PROJECT_ROOT, "models", "metadata_encoder.pkl")

# 1. Carga de artefactos previamente serializados (una sola vez en memoria)
MODELO = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
SCALER = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
META_BUNDLE = joblib.load(META_PATH) if os.path.exists(META_PATH) else None

# Recomendaciones de abasto por categoría y estación del año
RECOMENDACIONES_ABASTO = {
    "Coffee": "Asegurar stock alto de grano tostado (Espresso, Gourmet y Drip) e insumos de vasos térmicos.",
    "Tea": "Incrementar inventario de infusiones (Chai, Black Tea y Herbal Tea) de alta rotación estacional.",
    "Bakery": "Programar horneado matutino prioritario de Scones, Croissants y Biscotti para venta cruzada.",
    "Drinking Chocolate": "Reforzar abasto de cacao orgánico y chocolate oscuro para bebidas calientes de temporada.",
    "Flavours": "Abastecer jarabes saborizantes complementarios para personalización de bebidas.",
    "Coffee beans": "Disponer empaques de café en grano para venta minorista en mostrador.",
    "Loose Tea": "Mantener stock de té en hoja suelta para clientes frecuentes de temporada.",
    "Branded": "Mantener inventario base de tazas y artículos promocionales de la cafetería.",
    "Packaged Chocolate": "Mantener lote de barras de chocolate empaquetado junto a caja."
}


def _cargar_artefactos_si_necesario():
    """Carga perezosa de respaldo en caso de que el script se importe antes de entrenar."""
    global MODELO, SCALER, META_BUNDLE
    if MODELO is None:
        MODELO = joblib.load(MODEL_PATH)
    if SCALER is None:
        SCALER = joblib.load(SCALER_PATH)
    if META_BUNDLE is None:
        META_BUNDLE = joblib.load(META_PATH)


def _normalizar_entrada(entrada: dict) -> tuple[dict, str]:
    """
    Convierte un diccionario de entrada del usuario/formulario en las columnas exactas
    esperadas por el modelo (permitiendo tanto nombres amigables como códigos numéricos).
    """
    mes = int(entrada.get("month", entrada.get("mes", 1)))

    if "estacion_id" in entrada:
        estacion_id = int(entrada["estacion_id"])
    elif "estacion" in entrada:
        clave_est = str(entrada["estacion"]).strip().lower()
        estacion_id = SEASON_FROM_NAME.get(clave_est, obtener_estacion_id(mes))
    else:
        estacion_id = obtener_estacion_id(mes)

    estacion_nombre = SEASON_NAMES.get(estacion_id, "Invierno")

    day_of_week = int(entrada.get("day_of_week", entrada.get("dia_semana", 2)))
    hour = int(entrada.get("hour", entrada.get("hora", 8)))
    store_id = int(entrada.get("store_id", entrada.get("sucursal_id", 5)))
    unit_price = float(entrada.get("unit_price", entrada.get("precio_unitario", 3.50)))
    transaction_qty = int(entrada.get("transaction_qty", entrada.get("cantidad", 2)))

    if "revenue" in entrada:
        revenue = float(entrada["revenue"])
    else:
        revenue = round(unit_price * transaction_qty, 2)

    if "beverage_temp_enc" in entrada:
        beverage_temp_enc = int(entrada["beverage_temp_enc"])
    else:
        temp_str = str(entrada.get("beverage_temp", entrada.get("temperatura_bebida", "Hot")))
        beverage_temp_enc = BEVERAGE_TEMP_MAP.get(temp_str, 0)

    fila_normalizada = {
        "estacion_id": estacion_id,
        "month": mes,
        "day_of_week": day_of_week,
        "hour": hour,
        "store_id": store_id,
        "unit_price": unit_price,
        "transaction_qty": transaction_qty,
        "revenue": revenue,
        "beverage_temp_enc": beverage_temp_enc
    }
    return fila_normalizada, estacion_nombre


def predecir(entrada: dict) -> dict:
    """
    Recibe las variables de entrada como un diccionario plano (provenientes del formulario)
    y retorna la inferencia estructurada lista para mostrar en pantalla y planificar el abasto.

    Parámetros:
        entrada (dict): Valores de entrada ingresados por el usuario (estación, mes, hora,
                        sucursal, precio unitario, cantidad, temperatura de bebida, etc.).

    Retorna:
        dict: Resultado que incluye la predicción principal, etiqueta de menú y abasto,
              probabilidades/confianza y estado.
    """
    _cargar_artefactos_si_necesario()

    # 2. Normalización y conversión a DataFrame para asegurar orden y nombres de columnas
    fila_limpia, estacion_nombre = _normalizar_entrada(entrada)
    df_input = pd.DataFrame([fila_limpia])[FEATURE_COLUMNS]

    # 3. Aplicación del mismo preprocesamiento (escalamiento) que en entrenamiento
    X_scaled = SCALER.transform(df_input)

    # 4. Inferencia del modelo Random Forest
    prediccion = int(MODELO.predict(X_scaled)[0])

    clases = META_BUNDLE["metadata"]["classes"]
    categoria_predicha = clases[prediccion]

    # Cálculo de probabilidades si el modelo soporta predict_proba
    try:
        probabilidades = MODELO.predict_proba(X_scaled)[0]
        probabilidad = float(probabilidades[prediccion])
        indices_top = np.argsort(probabilidades)[::-1][:3]
        top_3_menu = {
            clases[idx]: round(float(probabilidades[idx]) * 100, 2)
            for idx in indices_top
        }
    except AttributeError:
        probabilidad = None
        top_3_menu = {}

    recomendacion = RECOMENDACIONES_ABASTO.get(
        categoria_predicha,
        "Asegurar inventario preventivo en almacén para la temporada seleccionada."
    )

    # 5. Salida estructurada conforme al contrato de interfaz (Sección 5 del PDF)
    return {
        "clase_predicha": prediccion,
        "etiqueta": f"Abasto Prioritario ({estacion_nombre}): {categoria_predicha}",
        "categoria_menu": categoria_predicha,
        "estacion_anio": estacion_nombre,
        "recomendacion_abasto": recomendacion,
        "top_3_categorias_probables": top_3_menu,
        "confianza": round(probabilidad * 100, 2) if probabilidad is not None else "N/A",
        "status": "success"
    }


if __name__ == "__main__":
    print("=" * 80)
    print("DEMO DE INFERENCIA - PREDICCIÓN DE MENÚ POR ESTACIONES DEL AÑO (ABASTO)")
    print("=" * 80)

    ejemplos_prueba = [
        {
            "estacion": "Invierno",
            "month": 1,
            "day_of_week": 1,
            "hour": 8,
            "store_id": 5,
            "unit_price": 4.25,
            "transaction_qty": 1,
            "beverage_temp": "Hot"
        },
        {
            "estacion": "Primavera",
            "month": 4,
            "day_of_week": 5,
            "hour": 10,
            "store_id": 8,
            "unit_price": 3.75,
            "transaction_qty": 1,
            "beverage_temp": "Not a Beverage"
        },
        {
            "estacion": "Verano",
            "month": 6,
            "day_of_week": 3,
            "hour": 15,
            "store_id": 3,
            "unit_price": 2.55,
            "transaction_qty": 2,
            "beverage_temp": "Hot"
        }
    ]

    for i, muestra in enumerate(ejemplos_prueba, start=1):
        print(f"\n--- Prueba Unitaria #{i} ---")
        print("Entrada (JSON / Diccionario):")
        print(json.dumps(muestra, indent=2, ensure_ascii=False))
        resultado = predecir(muestra)
        print("\nSalida Real Retornada por el Modelo:")
        print(json.dumps(resultado, indent=2, ensure_ascii=False))
