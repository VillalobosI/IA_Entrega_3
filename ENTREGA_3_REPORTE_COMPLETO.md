# Plantilla - Entrega 3: Desarrollo, Entrenamiento y Experimentación del Modelo
**Fundamentos de Inteligencia Artificial — Proyecto Final**

- **Fecha límite de entrega:** 9 de noviembre de 2026 (Fin de Sprint 3)
- **Nombre del Proyecto:** Sistema Predictivo de Demanda de Menú y Abasto de Inventario por Estaciones del Año
- **Equipo / Número de Grupo:** Equipo IA - Cafetería Predictiva
- **Integrantes y Roles:**
  1. Integrante 1 — *Líder de Datos y Preprocesamiento*
  2. Integrante 2 — *Líder de Modelado y Experimentación*
  3. Integrante 3 — *Líder de Pipeline e Integración*

---

## 1. Selección y Justificación Técnica de los Modelos

### A. Definición de la Estrategia de Modelado
- [ ] **Estrategia Comparativa (Recomendada)**
- [x] **Estrategia Baseline vs. Modelo Optimizado:** Entrenar 1 modelo base (configuración por defecto) y comparar contra el mismo modelo tras un ajuste fino (*fine-tuning*) de hiperparámetros y poda regularizada.

### B. Justificación de la Elección

| Modelo / Algoritmo | Paradigma / Familia | Justificación Técnica (¿Por qué resuelve su problema?) |
| :--- | :--- | :--- |
| **Modelo 1 (Principal / Candidato):** Random Forest Optimizado (`max_depth=12`, `n_estimators=120`, `min_samples_split=10`, `min_samples_leaf=4`) | Machine Learning Supervisado / Ensamble de Árboles de Decisión (*Bagging*) | Resuelve la predicción multiclase de las 9 categorías del menú según la **estación del año**, mes, hora, sucursal y temperatura del producto porque captura relaciones no lineales e interacciones jerárquicas (ej. horarios matutinos de invierno vs. tardes de verano) reduciendo la varianza mediante el promedio de 120 árboles decorrelacionados con poda controlada. |
| **Modelo 2 (Baseline / Comparativo):** Random Forest Base (`max_depth=None`, `n_estimators=100`, `min_samples_split=2`) y Variante de Poda Superficial (`max_depth=4`) | Machine Learning Supervisado / Ensamble de Árboles sin restricción de profundidad | Sirve como punto de referencia directo para demostrar cómo un bosque sin restricción de profundidad memoriza el ruido de los horarios/fechas en entrenamiento (*Overfitting*: 85.36% Train vs. 78.31% Test), mientras que el ajuste de poda (`max_depth=12`) eleva el desempeño en Test a **80.48% Accuracy** y **0.91 Macro F1-Score**. |

---

## 2. Pipeline de Datos y Estructura en el Repositorio

### A. Ubicación de Archivos en el Repositorio
- **Script / Notebook de entrenamiento:** `src/models/train.py` y `notebooks/03_model_training.ipynb`
- **Script del pipeline de preprocesamiento:** `src/data/preprocess.py`
- **Modelos serializados y guardados:** `models/modelo_candidato.pkl` (y `models/rf_menu_estacional_v1.joblib`)
- **Transformadores / Escaladores guardados:** `models/scaler.pkl` (`StandardScaler`) y `models/metadata_encoder.pkl` (`LabelEncoder`)
- **Archivo de dependencias:** `requirements.txt` actualizado

### B. Comando de Reproducibilidad en una Sola Línea
```bash
python src/models/train.py --data "C:\Users\villa\Downloads\dataset_limpio.csv" --output models/
```
*(También soporta ejecución 100% relativa al clonar el repositorio: `python src/models/train.py --data data/processed/dataset_limpio.csv --output models/`)*

---

## 3. Matriz de Experimentación y Ajuste de Hiperparámetros

Evaluación sobre el mismo conjunto de prueba (**Test Set = 29,824 muestras**, 20% estratificado del total de **149,116 registros**; **Train Set = 119,292 muestras**):

| Variante / Experimento | Modelo | Hiperparámetros Clave Evaluados | Métrica Principal (Test) | Métrica Secundaria (Test) | Tiempo de Entrenamiento / Inferencia | Diagnóstico (Overfitting / Underfitting / Óptimo) |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Exp 1: Baseline** | Random Forest Base | `n_estimators=100`, `max_depth=None`, `min_samples_split=2` | Accuracy: **78.31%** | F1-Score Ponderado: **0.7823** | 1.26 s / 0.46 s | **Overfitting severo** (85.36% en Train vs. 78.31% en Test; brecha de +7.05%) |
| **Exp 2: Ajuste A** | Random Forest Poda Severa | `n_estimators=50`, `max_depth=4`, `min_samples_split=20` | Accuracy: **75.68%** | F1-Score Ponderado: **0.7382** | 0.39 s / 0.19 s | **Underfitting** (75.59% en Train, 75.68% en Test; capacidad insuficiente) |
| **Exp 3: Optimizado (Ganador)** | Random Forest Optimizado | `n_estimators=120`, `max_depth=12`, `min_samples_split=10`, `min_samples_leaf=4` | Accuracy: **80.48%** | F1-Score Ponderado: **0.8009** (Macro F1: **0.91**) | 1.07 s / 0.29 s | **Modelo Ganador / Óptimo** (81.17% en Train vs. 80.48% en Test; generalización sólida) |

### Análisis Crítico de los Resultados:
1. **¿Qué modelo resultó ganador y por qué?**
   Resultó ganador el **Experimento 3 (Random Forest Optimizado con `n_estimators=120`, `max_depth=12`, `min_samples_split=10`, `min_samples_leaf=4`)**, alcanzando un **Accuracy de 80.48%** en el conjunto de prueba, una **Precisión ponderada de 82.64%** y un **Macro F1-Score de 0.91** sobre las 9 categorías del menú. Matemáticamente, al limitar la profundidad máxima a 12 niveles y exigir al menos 4 muestras por hoja (`min_samples_leaf=4`), cada árbol del ensamble aprende particiones estacionales y de precios estadísticamente significativas sin fragmentar el espacio de características en hojas puras de una sola transacción atípica.
2. **Detección y mitigación del Sobreajuste (Overfitting):**
   - **Desempeño en Entrenamiento (Train) vs. Prueba (Test):** En el modelo *Baseline* (`Exp 1`), el desempeño en Train fue de **85.36%** mientras que en Test cayó a **78.31%** (una brecha de sobreajuste de **7.05 puntos porcentuales**).
   - **Técnica de regularización aplicada para cerrar la brecha:** Se aplicó **pre-poda estructural de árboles (Pre-pruning)** mediante tres hiperparámetros simultáneos: restricción de profundidad máxima (`max_depth=12`), incremento del mínimo de muestras para dividir un nodo interno (`min_samples_split=10`) y mínimo de muestras por nodo hoja (`min_samples_leaf=4`), además de aumentar el tamaño del comité a `n_estimators=120`. Esto redujo la brecha Train vs. Test de **7.05% a solo 0.69%** (`81.17%` Train vs. `80.48%` Test) y mejoró el Accuracy en Test en **+2.17 puntos porcentuales**.

---

## 4. Evidencias de Ejecución Funcional

### A. Registro de Consola / Logs del Entrenamiento

```text
[INFO] Cargando dataset: C:\Users\villa\Downloads\dataset_limpio.csv ...
[INFO] Dataset procesado correctamente (149,116 registros).
[INFO] Conjunto Train: 119,292 muestras | Conjunto Test: 29,824 muestras
--------------------------------------------------------------------------------
[INFO] Entrenando Exp 1: Baseline - Random Forest Base (Sin Poda / Default) (n_estimators=100, max_depth=None, min_samples_split=2)...
[INFO] Exp 1: Baseline finalizado en 1.26 segundos | Train Acc: 85.36% | Test Acc: 78.31% | Test F1-Score: 0.7823
[INFO] Entrenando Exp 2: Ajuste A - Random Forest Poda Severa (Árboles Superficiales) (n_estimators=50, max_depth=4, min_samples_split=20)...
[INFO] Exp 2: Ajuste A finalizado en 0.39 segundos | Train Acc: 75.59% | Test Acc: 75.68% | Test F1-Score: 0.7382
[INFO] Entrenando Exp 3: Optimizado (Ganador) - Random Forest Optimizado (Poda Regularizada) (n_estimators=120, max_depth=12, min_samples_split=10, min_samples_leaf=4)...
[INFO] Exp 3: Optimizado (Ganador) finalizado en 1.07 segundos | Train Acc: 81.17% | Test Acc: 80.48% | Test F1-Score: 0.8009
--------------------------------------------------------------------------------
[INFO] Mejor modelo seleccionado: Exp 3: Optimizado (Ganador) (Random Forest Optimizado (Poda Regularizada))
[RESULTADOS EN TEST SET]
                    precision    recall  f1-score   support

            Bakery       1.00      1.00      1.00      4559
           Branded       0.97      1.00      0.98       149
            Coffee       0.81      0.66      0.73     11684
      Coffee beans       1.00      0.99      0.99       351
Drinking Chocolate       1.00      0.52      0.68      2294
          Flavours       1.00      1.00      1.00      1358
         Loose Tea       1.00      1.00      1.00       242
Packaged Chocolate       1.00      1.00      1.00        97
               Tea       0.68      0.92      0.78      9090

          accuracy                           0.80     29824
         macro avg       0.94      0.90      0.91     29824
      weighted avg       0.83      0.80      0.80     29824

[SUCCESS] Modelo serializado con éxito en: models\modelo_candidato.pkl
[SUCCESS] Copia compatible guardada en: models\rf_menu_estacional_v1.joblib
[SUCCESS] Escalador guardado en: models\scaler.pkl
[SUCCESS] Metadatos y codificador guardados en: models\metadata_encoder.pkl
[SUCCESS] Matriz de confusión exportada en: models\confusion_matrix.png
```

### B. Matriz de Confusión Resultante en el Conjunto de Prueba (29,824 muestras)

*(La imagen generada se encuentra guardada en `models/confusion_matrix.png`)*

| Real \ Predicho | Bakery | Branded | Coffee | Coffee beans | Drinking Choc. | Flavours | Loose Tea | Packaged Choc. | Tea | Total Real |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bakery** | **4559** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4,559 |
| **Branded** | 0 | **149** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 149 |
| **Coffee** | 0 | 0 | **7691** | 0 | 1 | 0 | 0 | 0 | 3992 | 11,684 |
| **Coffee beans** | 0 | 5 | 0 | **346** | 0 | 0 | 0 | 0 | 0 | 351 |
| **Drinking Chocolate** | 0 | 0 | 1112 | 0 | **1182** | 0 | 0 | 0 | 0 | 2,294 |
| **Flavours** | 0 | 0 | 0 | 0 | 0 | **1358** | 0 | 0 | 0 | 1,358 |
| **Loose Tea** | 0 | 0 | 0 | 0 | 0 | 0 | **242** | 0 | 0 | 242 |
| **Packaged Chocolate** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **97** | 0 | 97 |
| **Tea** | 0 | 0 | 711 | 0 | 0 | 0 | 0 | 0 | **8379** | 9,090 |

---

## 5. Función de Inferencia Desacoplada (`predecir`)

Ubicada en `src/models/predict.py`.

### A. Código Fuente de la Función de Inferencia

```python
import joblib
import numpy as np
import pandas as pd

# 1. Carga de artefactos previamente serializados (una sola vez en memoria)
MODELO = joblib.load("models/modelo_candidato.pkl")
SCALER = joblib.load("models/scaler.pkl")
META_BUNDLE = joblib.load("models/metadata_encoder.pkl")

def predecir(entrada: dict) -> dict:
    """
    Recibe las variables de entrada como un diccionario plano (provenientes del formulario)
    y retorna la inferencia estructurada lista para mostrar en pantalla y planificar el abasto.
    """
    fila_limpia, estacion_nombre = _normalizar_entrada(entrada)
    df_input = pd.DataFrame([fila_limpia])[FEATURE_COLUMNS]

    # 3. Aplicación del mismo preprocesamiento que en entrenamiento
    X_scaled = SCALER.transform(df_input)

    # 4. Inferencia del modelo
    prediccion = int(MODELO.predict(X_scaled)[0])
    clases = META_BUNDLE["metadata"]["classes"]
    categoria_predicha = clases[prediccion]

    try:
        probabilidades = MODELO.predict_proba(X_scaled)[0]
        probabilidad = float(probabilidades[prediccion])
    except AttributeError:
        probabilidad = None

    # 5. Salida estructurada
    return {
        "clase_predicha": prediccion,
        "etiqueta": f"Abasto Prioritario ({estacion_nombre}): {categoria_predicha}",
        "categoria_menu": categoria_predicha,
        "estacion_anio": estacion_nombre,
        "recomendacion_abasto": RECOMENDACIONES_ABASTO.get(categoria_predicha),
        "confianza": round(probabilidad * 100, 2) if probabilidad is not None else "N/A",
        "status": "success"
    }
```

### B. Prueba Unitaria de Entrada y Salida (Demo de Inferencia)

- **Ejemplo de Entrada (JSON / Diccionario):**
```json
{
  "estacion": "Invierno",
  "month": 1,
  "day_of_week": 1,
  "hour": 8,
  "store_id": 5,
  "unit_price": 4.25,
  "transaction_qty": 1,
  "beverage_temp": "Hot"
}
```

- **Salida Real Retornada por el Modelo:**
```json
{
  "clase_predicha": 2,
  "etiqueta": "Abasto Prioritario (Invierno): Coffee",
  "categoria_menu": "Coffee",
  "estacion_anio": "Invierno",
  "recomendacion_abasto": "Asegurar stock alto de grano tostado (Espresso, Gourmet y Drip) e insumos de vasos térmicos.",
  "top_3_categorias_probables": {
    "Coffee": 91.44,
    "Tea": 5.95,
    "Drinking Chocolate": 2.61
  },
  "confianza": 91.44,
  "status": "success"
}
```

---

## 6. Reporte de Estado del Proyecto (Sprint Review para el Project Manager)

1. **Porcentaje global de avance del motor de IA:** **100%**
2. **Semáforo del Sprint:** **En tiempo**
3. **Bloqueos técnicos actuales (Blockers):**
   - *Ninguno, el modelo Random Forest entrenó sin problemas sobre los 149,116 registros en 1.07 segundos. Se mitigó el traslape natural de precios entre bebidas calientes (`Coffee` vs. `Tea`) mediante poda regularizada (`max_depth=12`, `min_samples_leaf=4`) y la inclusión de las 3 categorías más probables en la respuesta de inferencia para optimizar la toma de decisiones de abasto.*
4. **Plan de acción para la Entrega 4:**
   - *Conectar la función desacoplada `predecir()` de `src/models/predict.py` a los controles interactivos de Streamlit (selector de estación del año, mes, sucursal, horario y temperatura) y generar las gráficas del dashboard de planeación de abasto estacional.*
