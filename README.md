# Segmentación y predicción del riesgo de deserción escolar en el Perú

Proyecto del curso de **Inteligencia Artificial** (Universidad Nacional Federico Villarreal, docente Ciro Rodríguez, periodo 2026-II).

Solución de Machine Learning que combina **K-Means** (segmentación no supervisada) y **Regresión Logística** (predicción supervisada, comparada contra KNN y un baseline) para anticipar el riesgo de deserción escolar en instituciones educativas del Perú, usando datos reales y abiertos del **SIAGIE** (Sistema de Información de Apoyo a la Gestión de la Institución Educativa), Ministerio de Educación del Perú (MINEDU).

## Tabla de contenidos

- [1. Problema](#1-problema)
- [2. Objetivo](#2-objetivo)
- [3. Dataset](#3-dataset)
- [4. Instalación](#4-instalación)
- [5. Ejecución y entrenamiento](#5-ejecución-y-entrenamiento)
- [6. Reproducir los experimentos](#6-reproducir-los-experimentos)
- [7. Resultados](#7-resultados)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Autoría](#autoría)

## 1. Problema

No existe un mecanismo sistemático, basado en los datos administrativos que ya recopila el MINEDU, que permita anticipar qué instituciones educativas del Perú presentan mayor riesgo de deserción escolar en el periodo académico siguiente. Instituciones con indicadores académicos desfavorables (atraso escolar, baja aprobación) tienden a concentrar más deserción en el año siguiente, pero esta información no se usa hoy para generar alertas tempranas.

## 2. Objetivo

Desarrollar y evaluar una solución de aprendizaje automático que combine segmentación no supervisada (K-Means) y clasificación supervisada (Regresión Logística, evaluada frente a KNN y un baseline) para predecir el riesgo de deserción escolar del periodo siguiente en instituciones educativas del Perú, utilizando datos históricos reales de matrícula y trayectoria estudiantil.

## 3. Dataset

- **Fuente:** Padrón de Matrícula y Trayectoria Estudiantil — SIAGIE, MINEDU. Portal de Datos Abiertos del Perú.
- **Periodo:** años 2021, 2022, 2023 y 2024 (4 archivos CSV, uno por año).
- **Volumen:** 2,205,115 filas originales (institución x nivel x edad x año) → 423,012 filas tras agregación → 311,454 filas en el panel final de modelado (institución-nivel-año con año siguiente disponible).
- **Cómo obtenerlo:** descargar los 4 archivos desde el Portal de Datos Abiertos del Perú buscando "Padrón de Matrícula y Trayectoria Estudiantil" (SIAGIE, MINEDU) para los años 2021-2024, y colocarlos en `data/raw/` con sus nombres originales:
  - `Matriculación_y_Trayectoria_Estudiantil_2021.csv`
  - `Matriculación_y_Trayectoria_Estudiantil_2022.csv`
  - `Matriculación_y_Trayectoria_Estudiantil_2023.csv`
  - `Matriculación_y_Trayectoria_Estudiantil_2024.csv`
- **Variable objetivo:** `riesgo_alto_desercion` — 1 si la institución-nivel tuvo al menos un estudiante retirado en el año siguiente al año base, 0 en caso contrario.
- **Dataset final procesado:** `data/processed/dataset_final_desercion_escolar.csv` (incluye las tasas construidas, el panel temporal y el cluster asignado por K-Means).

## 4. Instalación

Requiere **Python 3.10+**.

```bash
git clone <url-de-tu-repositorio>
cd <nombre-del-repositorio>
python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Dependencias (ver `requirements.txt`): `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`, `jupyter`.

## 5. Ejecución y entrenamiento

1. Coloca los 4 CSV originales en `data/raw/` (ver sección 3).
2. Abre `notebooks/Proyecto_IA_Desercion_Escolar.ipynb` en Jupyter o súbelo a Google Colab.
3. En la celda de carga de datos, ajusta la variable `RUTA_DATOS` según tu entorno:
   - Mismo directorio / Colab con archivos subidos a la sesión: `RUTA_DATOS = ""`
   - Google Drive: `RUTA_DATOS = "/content/drive/MyDrive/tu-carpeta/"`
   - Local: `RUTA_DATOS = "./data/raw/"`
4. Ejecuta todas las celdas en orden (`Run All`). El notebook realiza, de principio a fin:
   - Limpieza e integración de los 4 años de datos.
   - Construcción del panel temporal (año *t* → año *t+1*) para evitar fuga de datos.
   - Análisis exploratorio de datos (EDA), con las figuras guardadas en `figuras/`.
   - Entrenamiento y comparación de modelos: K-Means (segmentación) y Regresión Logística / KNN / baseline (clasificación).
   - Serialización del modelo final en `models/modelo_regresion_logistica_desercion_escolar.pkl`.

Para cargar el modelo ya entrenado sin reentrenar:

```python
import joblib
modelo = joblib.load("models/modelo_regresion_logistica_desercion_escolar.pkl")
modelo.predict(X_nuevo)          # X_nuevo con las mismas columnas usadas en el entrenamiento
```

## 6. Reproducir los experimentos

- **Semilla aleatoria:** `random_state=42` en todos los componentes estocásticos (K-Means, validación cruzada).
- **División de datos:** temporal, no aleatoria — entrenamiento con años base 2021 y 2022 (predicen 2022 y 2023), prueba exclusivamente con año base 2023 (predice 2024). Esto previene la fuga de información entre periodos.
- **Validación:** validación cruzada estratificada de 5 particiones (`StratifiedKFold`) sobre el conjunto de entrenamiento, además de la división temporal train/test.
- **Diseño experimental completo:** ver Sección 19 del informe (`Proyecto_IA_Desercion_Escolar.docx`).

## 7. Resultados

**Modelo 1 — K-Means (segmentación, k=2 seleccionado por silhouette score ≈0.74):**

| Cluster | % de registros | Perfil |
|---|---|---|
| 0 | ~93.5% | Bajo riesgo (aprobación ≈99%) |
| 1 | ~6.5% | Estructuralmente vulnerable (aprobación ≈76.6%) |

**Modelo 2 — Clasificación supervisada (prueba: año base 2023 → predice 2024):**

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (clase mayoritaria) | 0.895 | 0.000 | 0.000 | 0.000 | — |
| KNN (k=15) | 0.894 | 0.474 | 0.108 | 0.176 | 0.790 |
| **Regresión Logística** | 0.744 | 0.253 | 0.766 | **0.380** | **0.812** |

La Regresión Logística fue seleccionada como modelo final por su mejor F1-score, recall y ROC-AUC frente a KNN y al baseline (confirmado con validación cruzada de 5 particiones). El nivel educativo (Secundaria), el tipo de gestión y la tasa de atraso escolar del año base resultaron los factores más asociados al riesgo.

Detalle completo de metodología, EDA, discusión y limitaciones: ver `Proyecto_IA_Desercion_Escolar.docx`.

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                # los 4 CSV originales del SIAGIE (MINEDU)
│   └── processed/          # dataset_final_desercion_escolar.csv
├── notebooks/
│   └── Proyecto_IA_Desercion_Escolar.ipynb
├── models/
│   └── modelo_regresion_logistica_desercion_escolar.pkl
├── figuras/                # gráficos del EDA y de la evaluación (Figuras 1-10)
└── docs/
    └── Proyecto_IA_Desercion_Escolar.docx   # informe completo (paper)
```

## Autoría

Proyecto desarrollado por **Luis Enrique Llacua Perez**, estudiante de la Escuela de Ingeniería Informática, Facultad de Ingeniería Electrónica e Informática, Universidad Nacional Federico Villarreal — curso de Inteligencia Artificial (docente: Ciro Rodríguez), periodo académico 2026-II.
