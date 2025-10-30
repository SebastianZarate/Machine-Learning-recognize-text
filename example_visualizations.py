"""
Ejemplo práctico de uso del módulo visualizations.py

Este script demuestra cómo usar las visualizaciones en un flujo
completo de Machine Learning.

Author: Machine Learning Workshop
Date: 2025-10-29
"""

import sys
from pathlib import Path

# Agregar src al path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from visualizations import (
    plot_model_comparison_bars,
    plot_confusion_matrices,
    plot_roc_curves_comparison,
    plot_sentiment_word_clouds,
    plot_sentiment_distribution,
    plot_text_length_distribution,
    create_comprehensive_report,
    save_all_figures
)


print("="*80)
print("📊 EJEMPLO: Visualizaciones en ML Workshop")
print("="*80 + "\n")

print("Este ejemplo demuestra cómo integrar visualizaciones en tu flujo de trabajo.\n")

# ============================================================================
# EJEMPLO 1: Después de entrenar modelos
# ============================================================================
print("="*80)
print("EJEMPLO 1: Visualizar resultados después del entrenamiento")
print("="*80 + "\n")

print("""
# Paso 1: Entrenar modelos y evaluar
from src.model import train_from_csv
from src.evaluation import evaluate_all_models

# Entrenar modelos
models = train_from_csv(
    csv_path='data/balanced_reviews.csv',
    output_path='models/review_model.joblib',
    test_size=0.2
)

# Evaluar (ya incluido automáticamente en train_from_csv)
import joblib
saved_data = joblib.load('models/review_model.joblib')
eval_results = saved_data['eval_results']
models_dict = saved_data['models']

# Paso 2: Crear visualizaciones
from src.visualizations import plot_model_comparison_bars

# Gráfico de comparación
fig = plot_model_comparison_bars(eval_results)
fig.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
""")

# ============================================================================
# EJEMPLO 2: Matrices de confusión
# ============================================================================
print("\n" + "="*80)
print("EJEMPLO 2: Visualizar matrices de confusión")
print("="*80 + "\n")

print("""
from src.visualizations import plot_confusion_matrices

# Crear matrices de confusión para todos los modelos
fig = plot_confusion_matrices(eval_results, cmap='Blues')
fig.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

# Interpretación:
# - Diagonal principal (arriba-izq y abajo-der) = predicciones correctas
# - Off-diagonal = errores (falsos positivos y falsos negativos)
# - Colores más oscuros = mayor cantidad de predicciones
""")

# ============================================================================
# EJEMPLO 3: Curvas ROC
# ============================================================================
print("\n" + "="*80)
print("EJEMPLO 3: Comparar modelos con curvas ROC")
print("="*80 + "\n")

print("""
from src.visualizations import plot_roc_curves_comparison

# Crear curvas ROC
fig = plot_roc_curves_comparison(models_dict, X_test, y_test)
fig.savefig('results/roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# Interpretación:
# - AUC = 0.5: Clasificador aleatorio (línea diagonal)
# - AUC > 0.9: Excelente rendimiento
# - AUC 0.8-0.9: Buen rendimiento
# - Más cerca de esquina superior izquierda = mejor
""")

# ============================================================================
# EJEMPLO 4: Word clouds
# ============================================================================
print("\n" + "="*80)
print("EJEMPLO 4: Analizar palabras con word clouds")
print("="*80 + "\n")

print("""
import pandas as pd
from src.visualizations import plot_sentiment_word_clouds

# Cargar datos
df = pd.read_csv('data/balanced_reviews.csv')

# Crear word clouds comparativos
fig = plot_sentiment_word_clouds(df, max_words=100)
fig.savefig('results/word_clouds.png', dpi=300, bbox_inches='tight')
plt.show()

# Interpretación:
# - Palabras más grandes = más frecuentes
# - Verde = palabras en reseñas positivas
# - Rojo = palabras en reseñas negativas
# - Identifica patrones de lenguaje por sentimiento
""")

# ============================================================================
# EJEMPLO 5: Distribución de datos
# ============================================================================
print("\n" + "="*80)
print("EJEMPLO 5: Analizar distribución del dataset")
print("="*80 + "\n")

print("""
from src.visualizations import (
    plot_sentiment_distribution,
    plot_text_length_distribution
)

# Distribución de sentimientos
fig1 = plot_sentiment_distribution(df)
fig1.savefig('results/sentiment_dist.png', dpi=300, bbox_inches='tight')
plt.show()

# Distribución de longitud de textos
fig2 = plot_text_length_distribution(df)
fig2.savefig('results/text_length_dist.png', dpi=300, bbox_inches='tight')
plt.show()

# Interpretación:
# - Verificar que el dataset esté balanceado
# - Identificar diferencias en longitud de textos por sentimiento
# - Detectar posibles sesgos en los datos
""")

# ============================================================================
# EJEMPLO 6: Reporte completo automático
# ============================================================================
print("\n" + "="*80)
print("EJEMPLO 6: Generar reporte completo automáticamente")
print("="*80 + "\n")

print("""
from src.visualizations import create_comprehensive_report, save_all_figures

# Generar todas las visualizaciones
figures = create_comprehensive_report(
    eval_results=eval_results,
    models=models_dict,
    X_test=X_test,
    y_test=y_test,
    df=df
)

# Guardar todas las figuras
save_all_figures(figures, output_dir='results/visualizations', dpi=300)

# Limpiar memoria
for fig in figures.values():
    plt.close(fig)

print("✅ Reporte completo generado en 'results/visualizations/'")
""")

# ============================================================================
# EJEMPLO 7: Integración en Jupyter Notebooks
# ============================================================================
print("\n" + "="*80)
print("EJEMPLO 7: Usar en Jupyter Notebooks")
print("="*80 + "\n")

print("""
# En Jupyter Notebook:

# Celda 1: Imports
import matplotlib.pyplot as plt
from src.visualizations import *
%matplotlib inline

# Celda 2: Cargar resultados
import joblib
saved_data = joblib.load('models/review_model.joblib')
eval_results = saved_data['eval_results']

# Celda 3: Visualizar comparación
fig = plot_model_comparison_bars(eval_results)
plt.show()  # Se muestra automáticamente en notebook

# Celda 4: Matrices de confusión
fig = plot_confusion_matrices(eval_results)
plt.show()

# Celda 5: Word clouds
df = pd.read_csv('data/balanced_reviews.csv')
fig = plot_sentiment_word_clouds(df)
plt.show()

# Las figuras se mostrarán inline en el notebook
# No es necesario guardarlas a menos que quieras exportarlas
""")

# ============================================================================
# EJEMPLO 8: Personalizar visualizaciones
# ============================================================================
print("\n" + "="*80)
print("EJEMPLO 8: Personalizar visualizaciones")
print("="*80 + "\n")

print("""
# Personalizar tamaño de figura
fig = plot_model_comparison_bars(
    eval_results,
    figsize=(16, 10),  # Más grande
    title="Mi Título Personalizado"
)

# Personalizar métricas mostradas
fig = plot_model_comparison_bars(
    eval_results,
    metrics=['accuracy', 'f1_score', 'roc_auc']  # Solo estas métricas
)

# Personalizar colores de confusion matrix
fig = plot_confusion_matrices(
    eval_results,
    cmap='Greens'  # Verde en lugar de azul
)

# Personalizar word clouds
from src.visualizations import plot_word_cloud

positive_texts = df[df['sentiment'] == 1]['review'].tolist()
fig = plot_word_cloud(
    positive_texts,
    title="Palabras Positivas",
    max_words=50,
    background_color='black',
    colormap='viridis'
)
plt.show()
""")

# ============================================================================
# EJEMPLO 9: Análisis comparativo
# ============================================================================
print("\n" + "="*80)
print("EJEMPLO 9: Crear análisis comparativo personalizado")
print("="*80 + "\n")

print("""
import matplotlib.pyplot as plt
from src.visualizations import (
    plot_model_comparison_bars,
    plot_confusion_matrices
)

# Crear figura con múltiples subplots
fig = plt.figure(figsize=(16, 10))

# Subplot 1: Comparación de métricas
ax1 = fig.add_subplot(2, 2, 1)
fig_bars = plot_model_comparison_bars(eval_results)
# Copiar contenido a subplot

# Subplot 2: Mejor modelo - matriz de confusión
# ... personalizar según necesites

plt.tight_layout()
plt.savefig('results/custom_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
""")

# ============================================================================
# EJEMPLO 10: Pipeline completo
# ============================================================================
print("\n" + "="*80)
print("EJEMPLO 10: Pipeline completo de entrenamiento y visualización")
print("="*80 + "\n")

print("""
# Pipeline completo de ML con visualizaciones automáticas

# Paso 1: Preparar datos
from src.data_preparation import create_balanced_dataset, split_dataset

df = create_balanced_dataset('data/IMDB Dataset.csv', n_samples=50000)
train_df, test_df = split_dataset(df, test_size=0.2)

# Visualizar distribución
from src.visualizations import plot_sentiment_distribution
fig = plot_sentiment_distribution(df)
fig.savefig('results/01_data_distribution.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Paso 2: Entrenar modelos
from src.model import train_from_csv

models = train_from_csv(
    csv_path='data/balanced_reviews.csv',
    output_path='models/review_model.joblib',
    test_size=0.2
)

# Paso 3: Cargar resultados
import joblib
saved_data = joblib.load('models/review_model.joblib')
eval_results = saved_data['eval_results']
models_dict = saved_data['models']

# Paso 4: Generar todas las visualizaciones
from src.visualizations import create_comprehensive_report, save_all_figures

figures = create_comprehensive_report(
    eval_results, models_dict, X_test, y_test, df=df
)

save_all_figures(figures, output_dir='results/final_report', dpi=300)

# Limpiar memoria
for fig in figures.values():
    plt.close(fig)

print("✅ Pipeline completo: Datos → Entrenamiento → Evaluación → Visualización")
print("✅ Resultados guardados en 'results/final_report/'")
""")

# ============================================================================
# Resumen de funciones disponibles
# ============================================================================
print("\n" + "="*80)
print("📚 RESUMEN DE FUNCIONES DISPONIBLES")
print("="*80 + "\n")

print("""
Funciones principales:
  1. plot_model_comparison_bars()     - Comparar métricas de modelos
  2. plot_confusion_matrices()        - Matrices de confusión
  3. plot_roc_curves_comparison()     - Curvas ROC
  4. plot_word_cloud()                - Word cloud simple
  5. plot_sentiment_word_clouds()     - Word clouds comparativos
  6. plot_sentiment_distribution()    - Distribución de sentimientos
  7. plot_text_length_distribution()  - Distribución de longitudes
  8. plot_feature_importance()        - Importancia de features (RF)
  9. create_comprehensive_report()    - Generar todas las visualizaciones
 10. save_all_figures()               - Guardar todas las figuras

Todas las funciones retornan plt.Figure para máxima flexibilidad.
""")

print("="*80)
print("💡 MEJORES PRÁCTICAS")
print("="*80 + "\n")

print("""
1. Guardar figuras:
   fig.savefig('output.png', dpi=300, bbox_inches='tight')
   plt.close(fig)  # Liberar memoria

2. Mostrar en notebooks:
   %matplotlib inline
   fig = plot_model_comparison_bars(eval_results)
   plt.show()  # Se muestra automáticamente

3. Personalizar tamaño:
   fig = plot_model_comparison_bars(eval_results, figsize=(14, 8))

4. Generar reporte completo:
   figures = create_comprehensive_report(...)
   save_all_figures(figures, output_dir='report')

5. Limpiar memoria después de guardar:
   for fig in figures.values():
       plt.close(fig)
""")

print("="*80)
print("✅ Consulta la documentación completa en PASO_5_1_VISUALIZATIONS.md")
print("="*80 + "\n")
