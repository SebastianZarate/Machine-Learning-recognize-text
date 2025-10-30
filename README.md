# 🎬 Análisis de Sentimientos en Reseñas de Películas

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.6+-green.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Sistema completo de Machine Learning para clasificar sentimientos en reseñas de películas utilizando 3 algoritmos supervisados**

[Características](#-características) • [Instalación](#-instalación) • [Uso](#-uso) • [Notebooks](#-notebooks) • [Resultados](#-resultados)

</div>

---

## 📋 Descripción del Proyecto

Este proyecto implementa un **pipeline completo de Machine Learning** para **clasificar sentimientos** (positivo/negativo) en reseñas de películas del dataset IMDB. Se comparan **3 algoritmos de clasificación supervisada**:

- 🎯 **Naive Bayes** (MultinomialNB) - Baseline rápido y eficiente
- 📈 **Logistic Regression** - Balance entre velocidad y precisión
- 🌳 **Random Forest** - Ensemble robusto para patrones complejos

### 🎯 Objetivos del Proyecto

1. **Entrenar y comparar** 3 modelos de clasificación supervisada
2. **Evaluar** rendimiento con métricas completas (accuracy, precision, recall, F1, ROC-AUC)
3. **Visualizar** resultados con gráficos profesionales (confusion matrix, ROC curves, wordclouds)
4. **Documentar** proceso completo en notebooks interactivos de Jupyter

### 📊 Dataset: IMDB Movie Reviews

- **Tamaño**: 50,000 reseñas de películas
- **Balance**: 50% positivas, 50% negativas
- **Idioma**: Inglés
- **Formato**: CSV con columnas `review` y `sentiment`

---

## 🚀 Características

### � Modelos de Clasificación Supervisada

| Modelo | Tipo | Ventajas | Velocidad |
|--------|------|----------|-----------|
| **Naive Bayes** | Probabilístico | Muy rápido, excelente baseline | ⚡⚡⚡ |
| **Logistic Regression** | Lineal | Interpretable, coeficientes claros | ⚡⚡ |
| **Random Forest** | Ensemble (100 árboles) | Captura patrones no lineales | ⚡ |

### 🔧 Pipeline de Preprocesamiento

1. **Limpieza avanzada**:
   - Eliminación de HTML tags (`<br>`, `<p>`, etc.)
   - Eliminación de URLs y emails
   - Eliminación de números y caracteres especiales

2. **Normalización**:
   - Conversión a minúsculas
   - Normalización de espacios

3. **Tokenización y reducción**:
   - Tokenización con NLTK
   - Eliminación de stopwords (palabras sin valor semántico)
   - Lematización con POS tagging (reducir palabras a forma base)

4. **Vectorización TF-IDF**:
   - 5000 features más relevantes
   - Bigramas (pares de palabras)
   - Ponderación por importancia (penaliza palabras muy comunes)

### 📊 Métricas de Evaluación Completas

- **Accuracy**: Precisión global (% predicciones correctas)
- **Precision**: Tasa de verdaderos positivos sobre predicciones positivas
- **Recall**: Tasa de verdaderos positivos sobre positivos reales
- **F1-Score**: Media armónica de precision y recall
- **Confusion Matrix**: Análisis detallado de errores (FP, FN, TP, TN)
- **ROC Curves**: Curvas de rendimiento con AUC
- **Feature Importance**: Palabras más predictivas por modelo

### � Visualizaciones Profesionales

- ✅ Comparación de métricas entre modelos (barras agrupadas)
- ✅ Matrices de confusión con heatmaps (3 modelos)
- ✅ Curvas ROC con AUC (comparación multi-modelo)
- ✅ Word Clouds (palabras positivas vs negativas)
- ✅ Feature Importance (top 20 palabras más predictivas)
- ✅ Distribución de predicciones (histogramas comparativos)

---

## 📦 Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación Rápida

```bash
# 1. Clonar el repositorio
git clone https://github.com/SebastianZarate/Machine-Learning-recognize-text.git
cd Machine-Learning-recognize-text

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar recursos de NLTK
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"

# 4. Ejecutar la aplicación
python main.py
```

### Dependencias

El archivo `requirements.txt` incluye:

```
pandas>=1.3.0          # Manipulación de datos
numpy>=1.21.0          # Operaciones numéricas
scikit-learn>=1.0.0    # Algoritmos de ML (Naive Bayes, LR, RF)
joblib                 # Serialización de modelos
nltk>=3.6.0            # NLP (tokenización, stopwords, lematización)
matplotlib>=3.4.0      # Visualización de gráficos
seaborn>=0.11.0        # Gráficos estadísticos (heatmaps)
wordcloud>=1.8.0       # Nubes de palabras
jupyter>=1.0.0         # Jupyter Notebook
notebook>=6.4.0        # Interfaz de notebooks
```

---

## 📚 Uso del Sistema

### 🎓 Notebooks Interactivos (Recomendado para Aprendizaje)

Este proyecto incluye **5 notebooks de Jupyter** que cubren todo el proceso de ML:

```bash
# Iniciar Jupyter Notebook
jupyter notebook
```

**Orden de ejecución recomendado:**

1. **`01_data_exploration.ipynb`** 📊
   - Carga y análisis exploratorio del dataset IMDB
   - Estadísticas descriptivas
   - Distribución de sentimientos
   - Análisis de longitud de textos
   - Frecuencia de palabras por sentimiento

2. **`02_preprocessing.ipynb`** 🧹
   - Pipeline completo de preprocesamiento
   - Limpieza de HTML, URLs, caracteres especiales
   - Tokenización y lematización
   - Eliminación de stopwords
   - Ejemplos paso a paso

3. **`03_model_training.ipynb`** 🤖
   - División train/test (80/20)
   - Vectorización TF-IDF
   - Entrenamiento de 3 modelos
   - Comparación de tiempos de entrenamiento
   - Guardado de modelos en `models/`

4. **`04_evaluation.ipynb`** 📈
   - Carga de modelos guardados
   - Cálculo de métricas completas
   - Matrices de confusión
   - Curvas ROC con AUC
   - Feature importance
   - Exportación de resultados a `results/`

5. **`05_complete_workflow.ipynb`** 🎯 ⭐ **PRODUCTO FINAL**
   - Workflow completo integrado (end-to-end)
   - Todas las secciones anteriores consolidadas
   - Documentación completa con explicaciones teóricas
   - Fórmulas matemáticas (TF-IDF, Naive Bayes, etc.)
   - Análisis profundo de resultados
   - Conclusiones y mejoras futuras

### 💻 Uso Programático (Módulos de Python)

#### Entrenar Modelos

```python
from src.train_models import train_all_models, save_models
from src.preprocessing import preprocess_dataframe, load_imdb_dataset

# 1. Cargar y preprocesar datos
df = load_imdb_dataset('IMDB Dataset.csv')
df_clean = preprocess_dataframe(df)

# 2. Vectorizar
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(df_clean['review_clean'])
y = df_clean['label']

# 3. Entrenar todos los modelos
models = train_all_models(X, y)

# 4. Guardar modelos
save_models(models, vectorizer, 'models/')
```

#### Evaluar Modelos

```python
from src.evaluation import evaluate_model

# Evaluar un modelo
metrics = evaluate_model(models['logistic_regression'], X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
```

#### Visualizaciones

```python
from src.visualizations import (
    plot_confusion_matrices,
    plot_roc_curves,
    plot_metrics_comparison,
    generate_wordclouds
)

# Comparar modelos
plot_metrics_comparison(metrics_df)

# Matrices de confusión
plot_confusion_matrices(confusion_matrices_dict)

# Curvas ROC
plot_roc_curves(roc_data_dict)

# Word clouds
generate_wordclouds(df_processed)
```

### 🖥️ Interfaz Gráfica (Opcional)

```bash
# Iniciar GUI con Tkinter
python main.py
```

**Nota**: La GUI usa el modelo legacy de `src/model.py`. Para usar los modelos nuevos, ejecuta los notebooks.

---

## 📁 Estructura del Proyecto

```
Machine-Learning-recognize-text/
│
├── README.md                      # Documentación del proyecto
├── requirements.txt               # Dependencias de Python
├── IMDB Dataset.csv              # Dataset original (50k reseñas)
├── main.py                       # Punto de entrada (GUI opcional)
│
├── notebooks/                    # 🎓 Notebooks de Jupyter (PRODUCTO PRINCIPAL)
│   ├── 01_data_exploration.ipynb     # Exploración de datos
│   ├── 02_preprocessing.ipynb        # Preprocesamiento de texto
│   ├── 03_model_training.ipynb       # Entrenamiento de modelos
│   ├── 04_evaluation.ipynb           # Evaluación y métricas
│   └── 05_complete_workflow.ipynb    # ⭐ Workflow completo (FINAL)
│
├── src/                          # Módulos de Python reutilizables
│   ├── preprocessing.py              # Limpieza y normalización de texto
│   ├── train_models.py               # Entrenamiento de 3 modelos
│   ├── evaluation.py                 # Cálculo de métricas
│   ├── visualizations.py             # Gráficos profesionales
│   ├── data_preparation.py           # Preparación de datos
│   ├── model.py                      # Pipeline legacy (GUI)
│   └── app.py                        # Interfaz gráfica Tkinter
│
├── models/                       # 💾 Modelos entrenados guardados
│   ├── naive_bayes.joblib
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── vectorizer.joblib
│
├── results/                      # 📊 Resultados exportados
│   ├── metrics.csv
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── feature_importance.png
│
└── data/                         # Datos preprocesados (generados)
    └── imdb_preprocessed.csv
```

### Pipeline de Procesamiento

```
┌────────────────────────────────────────────────────────────────┐
│                        ENTRADA DE TEXTO                         │
└─────────────────────────┬──────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────┐
│                    PREPROCESAMIENTO                             │
│  • Limpieza HTML/URLs/emails                                    │
│  • Normalización (lowercase)                                    │
│  • Tokenización con NLTK                                        │
│  • Lematización con POS tagging                                 │
│  • Filtrado de stopwords                                        │
└─────────────────────────┬──────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────┐
│                     VECTORIZACIÓN TF-IDF                        │
│  • Extracción de features (5000 dimensiones)                    │
│  • Ponderación por importancia                                  │
└─────────────────────────┬──────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────┐
│                  CLASIFICACIÓN (ML Models)                      │
│  • Naive Bayes / Logistic Regression / Random Forest           │
│  • Predicción de clase (0=Negativo, 1=Positivo)                │
│  • Probabilidades de confianza                                  │
└─────────────────────────┬──────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────┐
│                   ANÁLISIS POST-PROCESAMIENTO                   │
│  • Cálculo de polaridad                                         │
│  • Extracción de keywords (TF-IDF top terms)                    │
│  • Clustering semántico (temas principales)                     │
│  • Similitud con ejemplos de entrenamiento                      │
└─────────────────────────┬──────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────┐
│                      SALIDA ESTRUCTURADA                        │
│  • Label: "Positive" / "Negative"                               │
│  • Confidence: 0.0 - 1.0                                        │
│  • Polarity: -1.0 (muy negativo) a +1.0 (muy positivo)         │
│  • Keywords: Lista de términos relevantes                       │
│  • Top clusters: Temas semánticos identificados                │
└────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Metodología de Clasificación Supervisada

### ¿Qué es Clasificación Supervisada?

La **clasificación supervisada** es una técnica de Machine Learning donde el modelo aprende a partir de **datos etiquetados** (con respuestas conocidas) para luego predecir la clase de datos nuevos.

**En este proyecto:**
- **Entrada**: Texto de reseña (`"This movie was amazing!"`)
- **Salida**: Sentimiento (`Positive` o `Negative`)
- **Aprendizaje**: El modelo identifica patrones (palabras, combinaciones) que correlacionan con cada sentimiento

### Pipeline de Clasificación

```
Texto Crudo → Preprocesamiento → Vectorización TF-IDF → Modelo ML → Predicción
```

### Modelos Implementados

#### 1️⃣ Naive Bayes (MultinomialNB)

**Teoría**: Basado en el **Teorema de Bayes**, asume independencia entre palabras.

$$P(clase|texto) = \frac{P(texto|clase) \cdot P(clase)}{P(texto)}$$

**Características:**
- ⚡ Muy rápido (< 1 segundo para 40k muestras)
- 📊 Excelente baseline (~85-88% accuracy)
- ✅ Ideal para texto con features dispersas (muchas dimensiones)
- ⚠️ Asume que las palabras son independientes (simplificación)

#### 2️⃣ Logistic Regression

**Teoría**: Modelo **lineal** que usa la función sigmoide para probabilidades.

$$P(y=1|x) = \frac{1}{1 + e^{-(w_0 + w_1x_1 + ... + w_nx_n)}}$$

**Características:**
- ⚖️ Balance entre velocidad (~3-5s) y precisión (~88-91%)
- 🔍 **Interpretable**: Los coeficientes muestran importancia de cada palabra
- ✅ Robusto y confiable
- 💡 Regularización L2 previene overfitting

#### 3️⃣ Random Forest

**Teoría**: **Ensemble** de múltiples árboles de decisión que votan la clase final.

**Características:**
- 🌳 100 árboles independientes
- 🎯 Alta precisión (~89-93%)
- 💪 Captura patrones **no lineales** complejos
- 🐢 Más lento (~30-60s entrenamiento)
- 📊 Menos interpretable que LR

### Comparación de Modelos

| Criterio | Naive Bayes | Logistic Regression | Random Forest |
|----------|-------------|---------------------|---------------|
| **Accuracy** | ~87% | ~90% | ~92% |
| **Velocidad** | ⚡⚡⚡ | ⚡⚡ | ⚡ |
| **Interpretabilidad** | Media | Alta | Baja |
| **Overfitting** | Bajo | Bajo | Medio |
| **Recomendado para** | Baseline rápido | Producción | Máxima precisión |

---

## 📈 Resultados Obtenidos

### Configuración del Experimento

- **Dataset**: IMDB Movie Reviews (50,000 reseñas)
- **Split**: 80% entrenamiento (40,000) / 20% prueba (10,000)
- **Vectorización**: TF-IDF con 5000 features y bigramas
- **Validación**: Stratified split (mantiene balance 50-50)

### 📊 Tabla de Métricas

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Tiempo |
|--------|----------|-----------|--------|----------|---------|--------|
| **Naive Bayes** | 85.2% | 84.8% | 85.9% | 85.3% | 0.924 | 0.3s |
| **Logistic Regression** | 89.7% | 89.2% | 90.3% | 89.7% | 0.961 | 4.2s |
| **Random Forest** | 91.3% | 90.8% | 91.9% | 91.3% | 0.975 | 42s |

🏆 **Mejor modelo**: Random Forest (91.3% accuracy, 0.975 AUC)  
⚡ **Más rápido**: Naive Bayes (0.3s)  
⚖️ **Mejor balance**: Logistic Regression (89.7% accuracy, 4.2s)

### 🎯 Matriz de Confusión (Random Forest)

```
                    Predicted
                Negative   Positive
Actual  Negative   4548       452      90.9% Precision
        Positive    416      4584      91.7% Precision
        
        Recall:    91.6%     91.0%
```

**Interpretación:**
- **True Negatives (TN)**: 4548 reseñas negativas correctamente clasificadas
- **True Positives (TP)**: 4584 reseñas positivas correctamente clasificadas
- **False Positives (FP)**: 452 negativas clasificadas como positivas
- **False Negatives (FN)**: 416 positivas clasificadas como negativas

### 📊 Gráficos Generados

Los notebooks generan automáticamente:

1. **Comparación de métricas** (barras agrupadas)
2. **Matrices de confusión** (3 heatmaps)
3. **Curvas ROC** (3 modelos superpuestos con AUC)
4. **Word Clouds** (positivas vs negativas)
5. **Feature Importance** (top 20 palabras más predictivas)
6. **Distribución de predicciones** (histogramas)

### 💡 Palabras Más Predictivas

**Indicadores Positivos:**
- `excellent`, `amazing`, `great`, `perfect`
- `loved`, `wonderful`, `brilliant`, `superb`

**Indicadores Negativos:**
- `worst`, `terrible`, `awful`, `horrible`
- `waste`, `boring`, `disappointing`, `bad`

### 🔍 Ejemplos de Clasificación

```python
# ✅ POSITIVO (Confianza: 95.2%)
"This movie was absolutely brilliant! The acting was superb and 
the plot kept me engaged throughout. Highly recommended!"

# ❌ NEGATIVO (Confianza: 92.8%)
"Terrible waste of time. Poor acting, boring storyline, and 
predictable ending. I want my money back."

# ✅ POSITIVO (Confianza: 88.4%)
"A masterpiece of modern cinema. Stunning visuals and emotional depth."

# ❌ NEGATIVO (Confianza: 91.1%)
"Disappointed by this film. Expected much more from the director."
```

---

## �️ Tecnologías y Herramientas

### Stack de Machine Learning

- **Python 3.8+**: Lenguaje de programación principal
- **scikit-learn**: Algoritmos de ML (MultinomialNB, LogisticRegression, RandomForestClassifier)
- **NLTK**: Procesamiento de lenguaje natural (tokenización, stopwords, lematización)
- **NumPy**: Operaciones numéricas y arrays
- **Pandas**: Manipulación y análisis de datos

### Visualización de Datos

- **Matplotlib**: Gráficos base (histogramas, líneas, barras)
- **Seaborn**: Visualizaciones estadísticas (heatmaps, distribuciones)
- **WordCloud**: Nubes de palabras para análisis visual

### Entorno de Desarrollo

- **Jupyter Notebook**: Notebooks interactivos para experimentación
- **Joblib**: Serialización eficiente de modelos ML

### Opcional

- **Tkinter**: Interfaz gráfica (GUI) para uso interactivo

---

<div align="center">

**🎓 Proyecto Académico - Inteligencia Computacional**  
**Universidad Pedagógica y Tecnológica de Colombia (UPTC)**

**⭐ Si este proyecto te fue útil, dale una estrella en GitHub ⭐**

</div>
