# 🧠 Customer Segmentation & Spending Prediction

### K-Means · SVM · Red Neuronal (Keras) · Regresión Lineal

Proyecto de Machine Learning aplicado a segmentación de clientes y
predicción de gasto, utilizando técnicas de aprendizaje no supervisado y
supervisado.

------------------------------------------------------------------------

## 📌 Descripción General

Este proyecto implementa:

-   📊 K-Means para segmentación de clientes\
-   🤖 SVM para clasificación del nivel de gasto\
-   🧠 Red Neuronal (Keras) para predicción de gasto\
-   📈 Regresión Lineal como modelo base comparativo

Se realiza análisis exploratorio, preprocesamiento avanzado, reducción
de dimensionalidad con PCA y evaluación con métricas como RMSE, MAE,
F1-score y matriz de confusión.

------------------------------------------------------------------------

# 📂 Estructura del Proyecto

    ├── DataBases/
    │   └── 0. Different_stores_data_V2.csv
    │
    ├── Kmeans.py
    ├── Red_neuronal_y_regresion.py
    ├── Informe_teorico_.pdf
    └── README.md

------------------------------------------------------------------------

# 🔎 Parte A --- Segmentación con K-Means

## 🛠️ Preprocesamiento

Se aplicó:

-   StandardScaler → Variables numéricas
-   OneHotEncoder (drop='first') → Variables categóricas
-   ColumnTransformer para integración del pipeline

Variables utilizadas:

-   gender
-   age
-   category
-   quantity
-   total_profit

------------------------------------------------------------------------

## 📊 Clustering

Se probaron 3, 4 y 5 clusters utilizando K-Means. Para visualización se
aplicó PCA reduciendo a 2 dimensiones.

### Hallazgos

-   Edad y cantidad no diferencian fuertemente los grupos.
-   category y total_profit muestran mayor separación.
-   Aumentar clusters subdivide grupos pero no mejora dispersión
    significativamente.

------------------------------------------------------------------------

# 🤖 Parte B --- Clasificación con SVM

Se clasificó el nivel de gasto en:

-   Bajo
-   Medio
-   Alto

Configuración utilizada:

``` python
SVC(kernel='linear', class_weight='balanced')
```

### Resultados

-   Accuracy ≈ 39%
-   F1-score clase "medio": 0.00
-   Alta confusión entre clases "alto" y "bajo"
-   Clase "medio" no fue correctamente predicha

Conclusión: SVM no fue efectiva debido al desbalance de clases.

------------------------------------------------------------------------

# 🧠 Parte C --- Predicción con Red Neuronal

Arquitectura:

-   Modelo Secuencial
-   Capa entrada
-   2 capas ocultas (16 y 8 neuronas, ReLU)
-   1 neurona de salida
-   Optimizador Adam
-   EarlyStopping (patience=10)
-   69 épocas óptimas

------------------------------------------------------------------------

## 📈 Comparación de Modelos

### 🔹 Red Neuronal

-   RMSE: 0.31
-   MAE: 0.14

### 🔹 Regresión Lineal

-   RMSE: 31.85
-   MAE: 22.11

La Red Neuronal supera ampliamente a la regresión lineal.

------------------------------------------------------------------------

# 🚀 Aplicación Real

-   Marketing personalizado
-   Segmentación de clientes premium
-   Proyección de ingresos
-   Planeación de inventario

------------------------------------------------------------------------

# 🧩 Tecnologías

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   TensorFlow / Keras
-   Matplotlib
-   Seaborn

------------------------------------------------------------------------

# ▶️ Cómo Ejecutar

Instalar dependencias:

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

Ejecutar segmentación:

``` bash
python V2_Kmeans.py
```

Ejecutar predicción:

``` bash
python Red_neuronal.py
```

------------------------------------------------------------------------

# 👨‍💻 Autor

-   Andres Felipe Marcillo
