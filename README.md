# Sistema Predictivo de Menú y Abasto por Estaciones del Año (Entrega 3)

Repositorio oficial del motor de Inteligencia Artificial para la **Entrega 3: Desarrollo, Entrenamiento y Experimentación del Modelo** (Fundamentos de Inteligencia Artificial).

## Objetivo del Sistema
Predecir la categoría de menú demandada (`product_category`) en función de la **estación del año** (`Invierno`, `Primavera`, `Verano`, `Otoño`), mes, día de la semana, hora del día, ubicación de sucursal, precio unitario, volumen de transacción y temperatura del producto, permitiendo **anticipar y asegurar el abasto de inventario cuando llegue cada temporada**.

---

## Estructura del Repositorio

```text
IA_Entrega_3/
├── data/
│   └── processed/
│       └── dataset_limpio.csv          # Copia local reproducible del dataset limpio
├── models/
│   ├── modelo_candidato.pkl            # Modelo Random Forest ganador serializado
│   ├── rf_menu_estacional_v1.joblib    # Artefacto compatible joblib
│   ├── scaler.pkl                      # Transformador StandardScaler entrenado
│   ├── metadata_encoder.pkl            # LabelEncoder y metadatos estacionales
│   ├── metrics_report.json             # Registro experimental de métricas
│   ├── confusion_matrix.png            # Matriz de confusión en Test Set
│   └── feature_importance.png          # Gráfica de importancia de variables
├── notebooks/
│   └── 03_model_training.ipynb         # Notebook interactivo de experimentación
├── src/
│   ├── data/
│   │   └── preprocess.py               # Pipeline de carga, ingeniería estacional y escalado
│   └── models/
│       ├── train.py                    # Script de entrenamiento y Matriz de Experimentación
│       └── predict.py                  # Función desacoplada predecir(entrada: dict) -> dict
├── ENTREGA_3_REPORTE_COMPLETO.md       # Plantilla oficial de la Entrega 3 llenada al 100%
├── requirements.txt                    # Dependencias del entorno de Machine Learning
└── README.md                           # Documentación técnica del repositorio
```

---

## Instalación y Comando de Reproducibilidad

### 1. Instalar dependencias
```powershell
pip install -r requirements.txt
```

### 2. Ejecutar el entrenamiento (Comando en una sola línea)
Por defecto carga exactamente desde `"C:\Users\villa\Downloads\dataset_limpio.csv"` (con respaldo automático a `data/processed/dataset_limpio.csv` para reproducibilidad al clonar el repositorio):

```powershell
python src/models/train.py --data "C:\Users\villa\Downloads\dataset_limpio.csv" --output models/
```

### 3. Ejecutar la prueba unitaria de inferencia (`predecir`)
```powershell
python src/models/predict.py
```

---

## Resumen de Resultados Experimentales (Test Set: 29,824 muestras)

| Variante / Experimento | Modelo | Hiperparámetros Clave | Accuracy (Train) | Accuracy (Test) | F1-Score Ponderado (Test) | Tiempo Entrenamiento | Diagnóstico |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Exp 1: Baseline** | Random Forest Base | `n_estimators=100, max_depth=None, min_samples_split=2` | 85.36% | 78.31% | 0.7823 | 1.26 s | **Overfitting severo** (Brecha de +7.05% entre Train y Test) |
| **Exp 2: Ajuste A** | Random Forest Poda Severa | `n_estimators=50, max_depth=4, min_samples_split=20` | 75.59% | 75.68% | 0.7382 | 0.39 s | **Underfitting** (Árboles demasiado superficiales) |
| **Exp 3: Optimizado (Ganador)** | Random Forest Optimizado | `n_estimators=120, max_depth=12, min_samples_split=10, min_samples_leaf=4` | **81.17%** | **80.48%** | **0.8009** (Macro F1: **0.91**) | **1.07 s** | **Modelo Ganador / Óptimo** (Generalización sólida, brecha < 0.7%) |
