"""
Script de Entrenamiento, Experimentación y Serialización del Modelo Random Forest.
Predicción de Menú por Estaciones del Año para Abasto de Inventario.
Proyecto Final - Fundamentos de Inteligencia Artificial (Entrega 3).
"""

import os
import sys
import time
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

# Permitir importación desde la raíz del proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocess import (
    DEFAULT_DATASET_PATH,
    ejecutar_pipeline_preprocesamiento,
    resolver_ruta_dataset
)


def guardar_grafica_matriz_confusion(
    cm: np.ndarray,
    clases: list,
    ruta_salida: str
) -> None:
    """Genera y guarda la representación visual de la Matriz de Confusión (Sección 4.B)."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=clases,
        yticklabels=clases
    )
    plt.title("Matriz de Confusión - Random Forest Optimizado (Predicción de Menú por Estación)")
    plt.xlabel("Categoría de Menú Predicha (Abasto)")
    plt.ylabel("Categoría de Menú Real")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def guardar_grafica_importancia_variables(
    importancias: np.ndarray,
    nombres_columnas: list,
    ruta_salida: str
) -> None:
    """Genera gráfica de importancia de variables del Random Forest."""
    indices = np.argsort(importancias)[::-1]
    plt.figure(figsize=(9, 5))
    sns.barplot(
        x=importancias[indices],
        y=[nombres_columnas[i] for i in indices],
        hue=[nombres_columnas[i] for i in indices],
        palette="viridis",
        legend=False
    )
    plt.title("Importancia de Características en la Predicción Estacional del Menú")
    plt.xlabel("Importancia Relativa (Gini Importance)")
    plt.ylabel("Variable Predictora")
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def entrenar_y_evaluar(ruta_datos: str, directorio_salida: str) -> dict:
    """
    Ejecuta la Matriz de Experimentación (Baseline vs. Ajustes de Hiperparámetros),
    selecciona el mejor modelo Random Forest y serializa todos los artefactos en disco.
    """
    os.makedirs(directorio_salida, exist_ok=True)
    ruta_real = resolver_ruta_dataset(ruta_datos)

    print(f"[INFO] Cargando dataset: {ruta_real} ...")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        label_encoder,
        metadata
    ) = ejecutar_pipeline_preprocesamiento(ruta_csv=ruta_real)

    print(
        f"[INFO] Dataset procesado correctamente ({metadata['total_samples']:,} registros)."
    )
    print(
        f"[INFO] Conjunto Train: {metadata['train_samples']:,} muestras | "
        f"Conjunto Test: {metadata['test_samples']:,} muestras"
    )
    print("-" * 80)

    # Definición de los experimentos para la Matriz de Experimentación (Sección 3 del PDF)
    experimentos = [
        {
            "id": "Exp 1: Baseline",
            "nombre": "Random Forest Base (Sin Poda / Default)",
            "hiperparametros_str": "n_estimators=100, max_depth=None, min_samples_split=2",
            "params": {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42,
                "n_jobs": -1
            }
        },
        {
            "id": "Exp 2: Ajuste A",
            "nombre": "Random Forest Poda Severa (Árboles Superficiales)",
            "hiperparametros_str": "n_estimators=50, max_depth=4, min_samples_split=20",
            "params": {
                "n_estimators": 50,
                "max_depth": 4,
                "min_samples_split": 20,
                "min_samples_leaf": 10,
                "random_state": 42,
                "n_jobs": -1
            }
        },
        {
            "id": "Exp 3: Optimizado (Ganador)",
            "nombre": "Random Forest Optimizado (Poda Regularizada)",
            "hiperparametros_str": "n_estimators=120, max_depth=12, min_samples_split=10, min_samples_leaf=4",
            "params": {
                "n_estimators": 120,
                "max_depth": 12,
                "min_samples_split": 10,
                "min_samples_leaf": 4,
                "random_state": 42,
                "n_jobs": -1
            }
        }
    ]

    resultados_experimentos = []
    mejor_modelo = None
    mejor_f1 = -1.0
    mejor_pred_test = None
    mejor_exp_info = None

    for exp in experimentos:
        print(f"[INFO] Entrenando {exp['id']} - {exp['nombre']} ({exp['hiperparametros_str']})...")
        t_inicio = time.perf_counter()

        modelo = RandomForestClassifier(**exp["params"])
        modelo.fit(X_train, y_train)
        tiempo_entrenamiento = time.perf_counter() - t_inicio

        t_inf = time.perf_counter()
        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)
        tiempo_inferencia = time.perf_counter() - t_inf

        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test, average="weighted")
        prec_test = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
        rec_test = recall_score(y_test, y_pred_test, average="weighted", zero_division=0)

        brecha = acc_train - acc_test
        if brecha > 0.05:
            diagnostico = (
                f"Overfitting severo ({acc_train*100:.2f}% en Train vs. {acc_test*100:.2f}% en Test)"
            )
        elif acc_test < 0.77:
            diagnostico = (
                f"Underfitting ({acc_train*100:.2f}% en Train, {acc_test*100:.2f}% en Test)"
            )
        else:
            diagnostico = (
                f"Modelo Ganador / Óptimo ({acc_train*100:.2f}% en Train, {acc_test*100:.2f}% en Test)"
            )

        print(
            f"[INFO] {exp['id']} finalizado en {tiempo_entrenamiento:.2f} segundos | "
            f"Train Acc: {acc_train*100:.2f}% | Test Acc: {acc_test*100:.2f}% | "
            f"Test F1-Score: {f1_test:.4f}"
        )

        registro = {
            "experimento": exp["id"],
            "modelo": exp["nombre"],
            "hiperparametros": exp["hiperparametros_str"],
            "train_accuracy": round(float(acc_train), 4),
            "test_accuracy": round(float(acc_test), 4),
            "test_f1_weighted": round(float(f1_test), 4),
            "test_precision_weighted": round(float(prec_test), 4),
            "test_recall_weighted": round(float(rec_test), 4),
            "train_time_sec": round(float(tiempo_entrenamiento), 2),
            "inference_time_sec": round(float(tiempo_inferencia), 2),
            "diagnostico": diagnostico
        }
        resultados_experimentos.append(registro)

        if f1_test > mejor_f1:
            mejor_f1 = f1_test
            mejor_modelo = modelo
            mejor_pred_test = y_pred_test
            mejor_exp_info = registro

    print("-" * 80)
    print(f"[INFO] Mejor modelo seleccionado: {mejor_exp_info['experimento']} ({mejor_exp_info['modelo']})")
    print("[RESULTADOS EN TEST SET]")
    reporte_texto = classification_report(
        y_test,
        mejor_pred_test,
        target_names=metadata["classes"],
        digits=2,
        zero_division=0
    )
    print(reporte_texto)

    # Matriz de confusión e importancia de variables
    cm = confusion_matrix(y_test, mejor_pred_test)
    ruta_cm = os.path.join(directorio_salida, "confusion_matrix.png")
    guardar_grafica_matriz_confusion(cm, metadata["classes"], ruta_cm)

    ruta_fi = os.path.join(directorio_salida, "feature_importance.png")
    guardar_grafica_importancia_variables(
        mejor_modelo.feature_importances_,
        metadata["feature_columns"],
        ruta_fi
    )

    # Serialización de artefactos requeridos en la Sección 2.A y 5.A del PDF
    ruta_modelo_pkl = os.path.join(directorio_salida, "modelo_candidato.pkl")
    ruta_modelo_joblib = os.path.join(directorio_salida, "rf_menu_estacional_v1.joblib")
    ruta_scaler_pkl = os.path.join(directorio_salida, "scaler.pkl")
    ruta_metadata_pkl = os.path.join(directorio_salida, "metadata_encoder.pkl")

    joblib.dump(mejor_modelo, ruta_modelo_pkl, compress=3)
    joblib.dump(mejor_modelo, ruta_modelo_joblib, compress=3)
    joblib.dump(scaler, ruta_scaler_pkl)
    joblib.dump(
        {
            "label_encoder": label_encoder,
            "metadata": metadata
        },
        ruta_metadata_pkl
    )

    # Guardar reporte estructurado en JSON para trazabilidad
    resumen_final = {
        "dataset_path": ruta_real,
        "total_samples": metadata["total_samples"],
        "train_samples": metadata["train_samples"],
        "test_samples": metadata["test_samples"],
        "feature_columns": metadata["feature_columns"],
        "classes": metadata["classes"],
        "feature_importances": {
            col: round(float(imp), 4)
            for col, imp in zip(metadata["feature_columns"], mejor_modelo.feature_importances_)
        },
        "experiments": resultados_experimentos,
        "winner": mejor_exp_info,
        "confusion_matrix": cm.tolist(),
        "classification_report": reporte_texto
    }

    ruta_json = os.path.join(directorio_salida, "metrics_report.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resumen_final, f, indent=4, ensure_ascii=False)

    print(f"[SUCCESS] Modelo serializado con éxito en: {ruta_modelo_pkl}")
    print(f"[SUCCESS] Copia compatible guardada en: {ruta_modelo_joblib}")
    print(f"[SUCCESS] Escalador guardado en: {ruta_scaler_pkl}")
    print(f"[SUCCESS] Metadatos y codificador guardados en: {ruta_metadata_pkl}")
    print(f"[SUCCESS] Matriz de confusión exportada en: {ruta_cm}")

    return resumen_final


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento del modelo Random Forest para predicción de menú por estación."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Ruta al archivo dataset_limpio.csv"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models",
        help="Directorio donde se guardarán los modelos serializados y reportes"
    )
    args = parser.parse_args()
    entrenar_y_evaluar(ruta_datos=args.data, directorio_salida=args.output)


if __name__ == "__main__":
    main()
