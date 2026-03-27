# Brain Tumor MRI Classifier — EfficientNetB0

Clasificador de tumores cerebrales en imágenes MRI con transfer learning sobre EfficientNetB0, entrenado en dos fases (head + fine-tuning). Alcanza **92.37% de accuracy** y **98.34% de AUC** en el set de prueba.

---

## Dataset

| Propiedad | Detalle |
|-----------|---------|
| Fuente | [Nickparvar Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| Clases | glioma · meningioma · notumor · pituitary |
| Imágenes totales | ~7 200 (Training + Testing) |
| Splits | Training (85% train / 15% val) · Testing |

Las imágenes se preprocesan con recorte automático de márgenes negros via detección de contornos (OpenCV) antes del entrenamiento.

---

## Arquitectura

```
Input (224×224×3, rango 0–255)
       │
EfficientNetB0 (pesos ImageNet, preproceso interno)
       │
GlobalAveragePooling2D
BatchNormalization
Dense(256, relu) + L2 · Dropout(0.4)
Dense(128, relu) + L2 · Dropout(0.2)
Dense(4, softmax)
```

### Entrenamiento en dos fases

| Fase | Capas entrenables | LR | Épocas máx. |
|------|-------------------|----|-------------|
| 1 — Head | Solo cabeza (base congelada) | 1e-3 | 20 |
| 2 — Fine-tuning | Últimas 40 capas de EfficientNet | 1e-5 | 50 |

- Warmup de 2 épocas a LR=1e-6 antes del fine-tuning (doble `compile`).
- BatchNormalization de EfficientNet permanece congelada en fase 2.
- Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`.

---

## Resultados

### Métricas globales (test set)

| Métrica | Valor |
|---------|-------|
| Accuracy | **92.37%** |
| AUC | **98.34%** |

### Por clase (F1-score / Accuracy)

| Clase | Precision | Recall | F1 |
|-------|-----------|--------|----|
| glioma | — | — | **86%** |
| meningioma | — | — | **89%** |
| notumor | — | — | **97%** |
| pituitary | — | — | **97%** |

### Visualizaciones generadas

| Archivo | Contenido |
|---------|-----------|
| `training_curves.png` | Accuracy y Loss por época (fases 1 y 2) |
| `confusion_matrix.png` | Matriz de confusión normalizada |

---

## Ejecución en Kaggle

### 1. Agregar el dataset

En tu notebook de Kaggle, añade el dataset desde la barra lateral:

```
masoudnickparvar/brain-tumor-mri-dataset
```

La ruta de acceso queda como:

```
/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset
```

### 2. Activar acelerador GPU

`Settings → Accelerator → GPU T4 x2`

### 3. Ejecutar el notebook

Abre `MRI_corregido.ipynb` y ejecuta todas las celdas en orden (`Run All`).

El flujo completo es:

```
Celda 1  — (omitir en Kaggle, el dataset ya está montado)
Celda 2  — Instalar dependencias y verificar GPU
Celda 3  — Configuración global (DATA_DIR apunta a Kaggle)
Celda 4  — Preprocesamiento y construcción del dataset limpio
Celda 5  — Generadores de datos con augmentation
Celda 6  — Construcción del modelo EfficientNetB0
Celda 7  — Fase 1: entrenamiento del head
Celda 8  — Fase 2: fine-tuning con warmup
Celda 9  — Curvas de entrenamiento
Celda 10 — Evaluación en test set y matriz de confusión
Celda 11 — Definición de funciones Grad-CAM
Celda 12 — Construcción del modelo Grad-CAM y guardado
```

> **Nota:** La Celda 1 (subida de archivo) está diseñada para Google Colab. En Kaggle se omite; el dataset se accede directamente desde `DATA_DIR`.

### 4. Artefactos de salida

Al finalizar encontrarás en `/kaggle/working/`:

```
best_brain_model.keras      ← mejor checkpoint (val_accuracy)
brain_tumor_final.keras     ← modelo final completo
training_curves.png
confusion_matrix.png
```

---

## Requisitos

| Dependencia | Versión |
|-------------|---------|
| Python | 3.12 |
| TensorFlow | 2.19 |
| Keras | 3.x |
| OpenCV (`opencv-python-headless`) | ≥ 4.8 |
| scikit-learn | ≥ 1.4 |
| NumPy | ≥ 1.26 |
| Matplotlib / Seaborn | ≥ 3.8 / ≥ 0.13 |
| SciPy | ≥ 1.12 |

En Kaggle todas las dependencias excepto `opencv-python-headless` ya están disponibles en el entorno por defecto. La Celda 2 instala OpenCV automáticamente.

---

## Estructura del repositorio

```
MRI-brain-tumor/
├── MRI_corregido.ipynb   # Notebook principal (Kaggle-ready)
└── README.md
```

---

## Créditos

- Dataset: [Masoud Nickparvar](https://www.kaggle.com/masoudnickparvar) — Kaggle
- Arquitectura base: [EfficientNetB0](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019
