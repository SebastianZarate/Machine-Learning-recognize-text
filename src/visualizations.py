"""
Module for data visualization and result presentation.

This module provides comprehensive visualization functions for:
- Model performance comparison (bar charts, radar plots)
- Confusion matrices (heatmaps)
- ROC curves (multi-model comparison)
- Word clouds (positive/negative reviews)
- Distribution analysis (histograms, box plots)
- Feature importance (for tree-based models)

All functions return matplotlib figures for saving or displaying in notebooks/GUIs.

Author: Machine Learning Workshop
Date: 2025-10-29
"""

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Configuración global de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Colores personalizados para consistencia visual
COLORS = {
    'primary': '#3498db',      # Azul
    'secondary': '#2ecc71',    # Verde
    'warning': '#f39c12',      # Naranja
    'danger': '#e74c3c',       # Rojo
    'info': '#9b59b6',         # Púrpura
    'positive': '#27ae60',     # Verde oscuro
    'negative': '#c0392b',     # Rojo oscuro
    'neutral': '#95a5a6'       # Gris
}


def plot_model_comparison_bars(eval_results: Dict[str, Dict[str, Any]],
                               metrics: Optional[List[str]] = None,
                               figsize: Tuple[int, int] = (14, 8),
                               title: str = "Comparación de Rendimiento de Modelos") -> plt.Figure:
    """Crea gráfico de barras comparando métricas de múltiples modelos.
    
    Genera un gráfico de barras agrupadas para comparar el rendimiento
    de diferentes modelos de ML en múltiples métricas.
    
    Args:
        eval_results: Diccionario de resultados de evaluate_all_models()
            {model_name: {metrics_dict}}
        metrics: Lista de métricas a mostrar (default: todas excepto confusion_matrix)
            Opciones: ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'roc_auc']
        figsize: Tamaño de la figura (ancho, alto) en pulgadas
        title: Título del gráfico
    
    Returns:
        plt.Figure: Figura de matplotlib que puede ser guardada o mostrada
    
    Examples:
        >>> results = evaluate_all_models(models, X_test, y_test)
        >>> fig = plot_model_comparison_bars(results)
        >>> fig.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        >>> plt.show()
        
        >>> # Mostrar solo algunas métricas
        >>> fig = plot_model_comparison_bars(
        ...     results, 
        ...     metrics=['accuracy', 'f1_score', 'roc_auc']
        ... )
    
    Notes:
        - Las barras están agrupadas por modelo
        - Cada color representa una métrica diferente
        - Los valores se muestran encima de cada barra
        - La escala es de 0 a 1 para todas las métricas
        - ROC-AUC None se omite del gráfico
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'roc_auc']
    
    # Preparar datos
    models = list(eval_results.keys())
    data = {metric: [] for metric in metrics}
    
    for model in models:
        for metric in metrics:
            value = eval_results[model].get(metric)
            # Manejar ROC-AUC None
            if value is None:
                value = 0.0
            data[metric].append(value)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Configurar posiciones de las barras
    x = np.arange(len(models))
    width = 0.8 / len(metrics)  # Ancho de cada barra
    
    # Dibujar barras para cada métrica
    for i, metric in enumerate(metrics):
        offset = width * i - (width * len(metrics) / 2) + width / 2
        bars = ax.bar(
            x + offset,
            data[metric],
            width,
            label=metric.replace('_', ' ').title(),
            alpha=0.8
        )
        
        # Agregar valores encima de las barras
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Solo mostrar si no es 0
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    rotation=0
                )
    
    # Configurar ejes y etiquetas
    ax.set_xlabel('Modelo', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0, ha='center')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Ajustar layout
    plt.tight_layout()
    
    return fig


def plot_confusion_matrices(eval_results: Dict[str, Dict[str, Any]],
                            figsize: Tuple[int, int] = (15, 5),
                            cmap: str = 'Blues') -> plt.Figure:
    """Crea matriz de confusión para todos los modelos en un solo gráfico.
    
    Genera subplots con las matrices de confusión de cada modelo,
    mostrando verdaderos positivos, verdaderos negativos, falsos positivos
    y falsos negativos.
    
    Args:
        eval_results: Diccionario de resultados de evaluate_all_models()
        figsize: Tamaño de la figura (ancho, alto) en pulgadas
        cmap: Mapa de colores de seaborn ('Blues', 'Greens', 'Reds', etc.)
    
    Returns:
        plt.Figure: Figura de matplotlib con múltiples subplots
    
    Examples:
        >>> results = evaluate_all_models(models, X_test, y_test)
        >>> fig = plot_confusion_matrices(results)
        >>> fig.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        
        >>> # Usar diferentes colores
        >>> fig = plot_confusion_matrices(results, cmap='Greens')
    
    Notes:
        - Cada subplot muestra la matriz de confusión de un modelo
        - Los valores están normalizados (porcentajes)
        - Los números en las celdas representan el conteo absoluto
        - Diagonal principal = predicciones correctas (más oscuro = mejor)
    """
    n_models = len(eval_results)
    
    # Calcular grid de subplots
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    # Crear figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Asegurar que axes sea siempre un array
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_models > 1 else axes
    
    # Dibujar cada matriz de confusión
    for idx, (model_name, metrics) in enumerate(eval_results.items()):
        ax = axes[idx] if n_models > 1 else axes[0]
        cm = metrics['confusion_matrix']
        
        # Normalizar para mostrar porcentajes
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Crear heatmap
        sns.heatmap(
            cm_normalized,
            annot=cm,  # Mostrar conteos absolutos
            fmt='d',
            cmap=cmap,
            square=True,
            ax=ax,
            cbar=True,
            xticklabels=['Negativo', 'Positivo'],
            yticklabels=['Negativo', 'Positivo'],
            vmin=0,
            vmax=1
        )
        
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel('Clase Real', fontsize=10)
        ax.set_xlabel('Clase Predicha', fontsize=10)
    
    # Ocultar subplots vacíos
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Matrices de Confusión por Modelo', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def plot_roc_curves_comparison(models: Dict[str, Any],
                               X_test: np.ndarray,
                               y_test: np.ndarray,
                               figsize: Tuple[int, int] = (10, 8),
                               title: str = "Curvas ROC - Comparación de Modelos") -> plt.Figure:
    """Crea curvas ROC para todos los modelos en un solo gráfico.
    
    Genera curvas ROC (Receiver Operating Characteristic) para comparar
    el rendimiento de clasificación de múltiples modelos.
    
    Args:
        models: Diccionario {model_name: model_object}
        X_test: Datos de prueba (features)
        y_test: Etiquetas reales de prueba
        figsize: Tamaño de la figura (ancho, alto) en pulgadas
        title: Título del gráfico
    
    Returns:
        plt.Figure: Figura de matplotlib con curvas ROC
    
    Examples:
        >>> models = {
        ...     'Naive Bayes': nb_model,
        ...     'Logistic Regression': lr_model,
        ...     'Random Forest': rf_model
        ... }
        >>> fig = plot_roc_curves_comparison(models, X_test, y_test)
        >>> fig.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    
    Notes:
        - AUC (Area Under Curve) se muestra en la leyenda
        - Línea diagonal = clasificador aleatorio (AUC = 0.5)
        - Más cerca de la esquina superior izquierda = mejor rendimiento
        - Solo funciona con modelos que tienen predict_proba()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colores para las curvas
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for idx, (model_name, model) in enumerate(models.items()):
        # Verificar si el modelo tiene predict_proba
        if not hasattr(model, 'predict_proba'):
            print(f"⚠️  {model_name} no tiene predict_proba(), omitido")
            continue
        
        # Calcular probabilidades
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calcular curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Dibujar curva
        ax.plot(
            fpr,
            tpr,
            color=colors[idx],
            lw=2,
            label=f'{model_name} (AUC = {roc_auc:.3f})',
            alpha=0.8
        )
    
    # Línea diagonal (clasificador aleatorio)
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)', alpha=0.5)
    
    # Configurar ejes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    return fig


def plot_word_cloud(texts: List[str],
                   title: str = "Word Cloud",
                   max_words: int = 100,
                   figsize: Tuple[int, int] = (12, 6),
                   background_color: str = 'white',
                   colormap: str = 'viridis') -> plt.Figure:
    """Crea nube de palabras a partir de lista de textos.
    
    Genera una visualización de las palabras más frecuentes en los textos,
    donde el tamaño de cada palabra representa su frecuencia.
    
    Args:
        texts: Lista de strings (reviews, comentarios, etc.)
        title: Título del gráfico
        max_words: Número máximo de palabras a mostrar
        figsize: Tamaño de la figura (ancho, alto) en pulgadas
        background_color: Color de fondo ('white', 'black', etc.)
        colormap: Mapa de colores ('viridis', 'plasma', 'inferno', etc.)
    
    Returns:
        plt.Figure: Figura de matplotlib con word cloud
    
    Examples:
        >>> positive_reviews = df[df['sentiment'] == 1]['review'].tolist()
        >>> fig = plot_word_cloud(
        ...     positive_reviews,
        ...     title="Palabras más comunes en reseñas positivas"
        ... )
        >>> fig.savefig('positive_wordcloud.png', dpi=300, bbox_inches='tight')
        
        >>> # Word cloud con tema oscuro
        >>> fig = plot_word_cloud(
        ...     negative_reviews,
        ...     title="Palabras más comunes en reseñas negativas",
        ...     background_color='black',
        ...     colormap='Reds'
        ... )
    
    Notes:
        - Las palabras más grandes aparecen con mayor frecuencia
        - Se recomienda preprocesar textos (remover stopwords) antes
        - Para mejores resultados, usar textos en el mismo idioma
    """
    # Combinar todos los textos
    combined_text = ' '.join(texts)
    
    # Crear word cloud
    wordcloud = WordCloud(
        width=1200,
        height=600,
        max_words=max_words,
        background_color=background_color,
        colormap=colormap,
        relative_scaling=0.5,
        min_font_size=10,
        collocations=False  # Evitar bigramas duplicados
    ).generate(combined_text)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    return fig


def plot_sentiment_word_clouds(df: pd.DataFrame,
                               text_column: str = 'review',
                               sentiment_column: str = 'sentiment',
                               figsize: Tuple[int, int] = (16, 6),
                               max_words: int = 100) -> plt.Figure:
    """Crea word clouds comparativos para sentimientos positivos y negativos.
    
    Genera dos word clouds lado a lado: uno para reseñas positivas
    y otro para reseñas negativas, facilitando la comparación visual.
    
    Args:
        df: DataFrame con columnas de texto y sentimiento
        text_column: Nombre de la columna con los textos
        sentiment_column: Nombre de la columna con sentimiento (0=neg, 1=pos)
        figsize: Tamaño de la figura (ancho, alto) en pulgadas
        max_words: Número máximo de palabras por word cloud
    
    Returns:
        plt.Figure: Figura de matplotlib con dos subplots
    
    Examples:
        >>> df = pd.read_csv('reviews.csv')
        >>> fig = plot_sentiment_word_clouds(df)
        >>> fig.savefig('sentiment_comparison.png', dpi=300, bbox_inches='tight')
        
        >>> # Con columnas personalizadas
        >>> fig = plot_sentiment_word_clouds(
        ...     df,
        ...     text_column='comment',
        ...     sentiment_column='rating'
        ... )
    
    Notes:
        - Sentimiento 0 = Negativo (word cloud rojo)
        - Sentimiento 1 = Positivo (word cloud verde)
        - Se recomienda preprocesar textos antes de visualizar
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Word cloud de reseñas positivas
    positive_texts = df[df[sentiment_column] == 1][text_column].tolist()
    if positive_texts:
        combined_positive = ' '.join(positive_texts)
        wordcloud_positive = WordCloud(
            width=800,
            height=600,
            max_words=max_words,
            background_color='white',
            colormap='Greens',
            relative_scaling=0.5,
            min_font_size=10,
            collocations=False
        ).generate(combined_positive)
        
        axes[0].imshow(wordcloud_positive, interpolation='bilinear')
        axes[0].axis('off')
        axes[0].set_title('Reseñas Positivas (Palabras Frecuentes)', 
                         fontsize=12, fontweight='bold', color=COLORS['positive'])
    
    # Word cloud de reseñas negativas
    negative_texts = df[df[sentiment_column] == 0][text_column].tolist()
    if negative_texts:
        combined_negative = ' '.join(negative_texts)
        wordcloud_negative = WordCloud(
            width=800,
            height=600,
            max_words=max_words,
            background_color='white',
            colormap='Reds',
            relative_scaling=0.5,
            min_font_size=10,
            collocations=False
        ).generate(combined_negative)
        
        axes[1].imshow(wordcloud_negative, interpolation='bilinear')
        axes[1].axis('off')
        axes[1].set_title('Reseñas Negativas (Palabras Frecuentes)', 
                         fontsize=12, fontweight='bold', color=COLORS['negative'])
    
    plt.suptitle('Comparación de Palabras por Sentimiento', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig


def plot_sentiment_distribution(df: pd.DataFrame,
                                sentiment_column: str = 'sentiment',
                                figsize: Tuple[int, int] = (10, 6),
                                title: str = "Distribución de Sentimientos") -> plt.Figure:
    """Crea gráfico de barras mostrando distribución de sentimientos.
    
    Genera un gráfico de barras que muestra la cantidad de reseñas
    positivas y negativas en el dataset.
    
    Args:
        df: DataFrame con columna de sentimiento
        sentiment_column: Nombre de la columna con sentimiento (0=neg, 1=pos)
        figsize: Tamaño de la figura (ancho, alto) en pulgadas
        title: Título del gráfico
    
    Returns:
        plt.Figure: Figura de matplotlib con gráfico de barras
    
    Examples:
        >>> df = pd.read_csv('reviews.csv')
        >>> fig = plot_sentiment_distribution(df)
        >>> fig.savefig('sentiment_dist.png', dpi=300, bbox_inches='tight')
    
    Notes:
        - Muestra conteo absoluto y porcentaje
        - Incluye línea horizontal para balanceo 50/50
        - Útil para detectar desbalanceo de clases
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calcular distribución
    sentiment_counts = df[sentiment_column].value_counts().sort_index()
    labels = ['Negativo', 'Positivo']
    colors_list = [COLORS['negative'], COLORS['positive']]
    
    # Crear gráfico de barras
    bars = ax.bar(
        labels,
        sentiment_counts.values,
        color=colors_list,
        alpha=0.7,
        edgecolor='black',
        linewidth=1.5
    )
    
    # Agregar valores y porcentajes
    total = sentiment_counts.sum()
    for bar, count in zip(bars, sentiment_counts.values):
        height = bar.get_height()
        percentage = (count / total) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{count:,}\n({percentage:.1f}%)',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
    
    # Línea de referencia (50%)
    ax.axhline(y=total/2, color='gray', linestyle='--', linewidth=2, alpha=0.5, 
               label='Balanceo perfecto (50%)')
    
    # Configurar ejes
    ax.set_ylabel('Número de Reseñas', fontsize=12, fontweight='bold')
    ax.set_xlabel('Sentimiento', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    return fig


def plot_text_length_distribution(df: pd.DataFrame,
                                  text_column: str = 'review',
                                  sentiment_column: str = 'sentiment',
                                  figsize: Tuple[int, int] = (12, 6),
                                  bins: int = 50) -> plt.Figure:
    """Crea histograma de distribución de longitud de textos por sentimiento.
    
    Genera histogramas superpuestos mostrando la distribución de
    longitud de textos (número de palabras) para cada sentimiento.
    
    Args:
        df: DataFrame con columnas de texto y sentimiento
        text_column: Nombre de la columna con los textos
        sentiment_column: Nombre de la columna con sentimiento (0=neg, 1=pos)
        figsize: Tamaño de la figura (ancho, alto) en pulgadas
        bins: Número de bins para el histograma
    
    Returns:
        plt.Figure: Figura de matplotlib con histogramas superpuestos
    
    Examples:
        >>> df = pd.read_csv('reviews.csv')
        >>> fig = plot_text_length_distribution(df)
        >>> fig.savefig('text_length_dist.png', dpi=300, bbox_inches='tight')
    
    Notes:
        - Verde = Reseñas positivas
        - Rojo = Reseñas negativas
        - Transparencia permite ver superposición
        - Incluye estadísticas (media, mediana) en la leyenda
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calcular longitud de textos (número de palabras)
    df['text_length'] = df[text_column].apply(lambda x: len(str(x).split()))
    
    # Separar por sentimiento
    positive_lengths = df[df[sentiment_column] == 1]['text_length']
    negative_lengths = df[df[sentiment_column] == 0]['text_length']
    
    # Crear histogramas superpuestos
    ax.hist(
        positive_lengths,
        bins=bins,
        alpha=0.6,
        color=COLORS['positive'],
        label=f'Positivo (μ={positive_lengths.mean():.1f}, Med={positive_lengths.median():.1f})',
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.hist(
        negative_lengths,
        bins=bins,
        alpha=0.6,
        color=COLORS['negative'],
        label=f'Negativo (μ={negative_lengths.mean():.1f}, Med={negative_lengths.median():.1f})',
        edgecolor='black',
        linewidth=0.5
    )
    
    # Configurar ejes
    ax.set_xlabel('Longitud del Texto (número de palabras)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
    ax.set_title('Distribución de Longitud de Textos por Sentimiento', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Limpiar columna temporal
    df.drop('text_length', axis=1, inplace=True)
    
    return fig


def plot_feature_importance(model: Any,
                            feature_names: List[str],
                            top_n: int = 20,
                            figsize: Tuple[int, int] = (10, 8),
                            title: str = "Feature Importance") -> plt.Figure:
    """Crea gráfico de barras horizontales con feature importance.
    
    Genera un gráfico mostrando las características más importantes
    del modelo (solo para modelos basados en árboles como Random Forest).
    
    Args:
        model: Modelo entrenado con atributo feature_importances_
        feature_names: Lista de nombres de las características
        top_n: Número de características más importantes a mostrar
        figsize: Tamaño de la figura (ancho, alto) en pulgadas
        title: Título del gráfico
    
    Returns:
        plt.Figure: Figura de matplotlib con barras horizontales
    
    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> model.fit(X_train, y_train)
        >>> 
        >>> # Obtener nombres de features del vectorizador
        >>> feature_names = vectorizer.get_feature_names_out()
        >>> fig = plot_feature_importance(model, feature_names, top_n=30)
        >>> fig.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    
    Notes:
        - Solo funciona con modelos que tienen feature_importances_
        - Características ordenadas de más a menos importante
        - Valores normalizados (suman 1.0)
    
    Raises:
        AttributeError: Si el modelo no tiene feature_importances_
    """
    if not hasattr(model, 'feature_importances_'):
        raise AttributeError(
            f"El modelo {type(model).__name__} no tiene feature_importances_. "
            "Esta visualización solo funciona con modelos basados en árboles "
            "(RandomForest, GradientBoosting, etc.)"
        )
    
    # Obtener importancias
    importances = model.feature_importances_
    
    # Crear DataFrame y ordenar
    feature_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Gráfico de barras horizontales
    bars = ax.barh(
        range(len(feature_df)),
        feature_df['importance'],
        color=COLORS['primary'],
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Configurar ejes
    ax.set_yticks(range(len(feature_df)))
    ax.set_yticklabels(feature_df['feature'])
    ax.set_xlabel('Importancia', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()  # Mayor importancia arriba
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Agregar valores
    for bar, importance in zip(bars, feature_df['importance']):
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2.,
            f'{importance:.4f}',
            ha='left',
            va='center',
            fontsize=8,
            fontweight='bold'
        )
    
    plt.tight_layout()
    
    return fig


def plot_learning_curve(train_scores: List[float],
                       test_scores: List[float],
                       train_sizes: List[int],
                       figsize: Tuple[int, int] = (10, 6),
                       title: str = "Learning Curve") -> plt.Figure:
    """Crea curva de aprendizaje mostrando rendimiento vs tamaño de datos.
    
    Genera un gráfico que muestra cómo varía el rendimiento del modelo
    con diferentes tamaños de conjunto de entrenamiento.
    
    Args:
        train_scores: Lista de scores en conjunto de entrenamiento
        test_scores: Lista de scores en conjunto de prueba
        train_sizes: Lista de tamaños de conjunto de entrenamiento
        figsize: Tamaño de la figura (ancho, alto) en pulgadas
        title: Título del gráfico
    
    Returns:
        plt.Figure: Figura de matplotlib con curvas de aprendizaje
    
    Examples:
        >>> from sklearn.model_selection import learning_curve
        >>> 
        >>> train_sizes, train_scores, test_scores = learning_curve(
        ...     model, X, y, cv=5, 
        ...     train_sizes=np.linspace(0.1, 1.0, 10)
        ... )
        >>> 
        >>> fig = plot_learning_curve(
        ...     train_scores.mean(axis=1),
        ...     test_scores.mean(axis=1),
        ...     train_sizes
        ... )
        >>> fig.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
    
    Notes:
        - Línea azul = Rendimiento en entrenamiento
        - Línea naranja = Rendimiento en prueba
        - Gap grande entre líneas = overfitting
        - Ambas líneas bajas = underfitting
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Dibujar curvas
    ax.plot(
        train_sizes,
        train_scores,
        'o-',
        color=COLORS['primary'],
        linewidth=2,
        markersize=8,
        label='Training Score',
        alpha=0.8
    )
    
    ax.plot(
        train_sizes,
        test_scores,
        'o-',
        color=COLORS['warning'],
        linewidth=2,
        markersize=8,
        label='Validation Score',
        alpha=0.8
    )
    
    # Configurar ejes
    ax.set_xlabel('Training Set Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    return fig


def create_comprehensive_report(eval_results: Dict[str, Dict[str, Any]],
                               models: Dict[str, Any],
                               X_test: np.ndarray,
                               y_test: np.ndarray,
                               df: Optional[pd.DataFrame] = None,
                               output_prefix: str = 'ml_report') -> Dict[str, plt.Figure]:
    """Crea reporte visual completo con todas las visualizaciones.
    
    Genera todas las visualizaciones disponibles y las retorna en un
    diccionario para guardarlas o mostrarlas en notebooks.
    
    Args:
        eval_results: Resultados de evaluate_all_models()
        models: Diccionario con modelos entrenados
        X_test: Datos de prueba
        y_test: Etiquetas de prueba
        df: DataFrame opcional con datos originales para word clouds
        output_prefix: Prefijo para nombres de archivos al guardar
    
    Returns:
        Dict[str, plt.Figure]: Diccionario con todas las figuras generadas
            Keys: 'comparison_bars', 'confusion_matrices', 'roc_curves', etc.
    
    Examples:
        >>> results = evaluate_all_models(models, X_test, y_test)
        >>> df = pd.read_csv('reviews.csv')
        >>> 
        >>> figures = create_comprehensive_report(
        ...     results, models, X_test, y_test, df=df
        ... )
        >>> 
        >>> # Guardar todas las figuras
        >>> for name, fig in figures.items():
        ...     fig.savefig(f'{name}.png', dpi=300, bbox_inches='tight')
        ...     plt.close(fig)
    
    Notes:
        - Genera hasta 6 visualizaciones diferentes
        - Solo genera word clouds si se proporciona df
        - Retorna dict para máxima flexibilidad
        - Las figuras deben cerrarse manualmente con plt.close()
    """
    figures = {}
    
    # 1. Comparación de métricas
    print("📊 Generando gráfico de comparación de modelos...")
    figures['comparison_bars'] = plot_model_comparison_bars(eval_results)
    
    # 2. Matrices de confusión
    print("📊 Generando matrices de confusión...")
    figures['confusion_matrices'] = plot_confusion_matrices(eval_results)
    
    # 3. Curvas ROC
    print("📊 Generando curvas ROC...")
    figures['roc_curves'] = plot_roc_curves_comparison(models, X_test, y_test)
    
    # 4. Distribución de sentimientos (si hay df)
    if df is not None:
        print("📊 Generando visualizaciones de datos...")
        figures['sentiment_distribution'] = plot_sentiment_distribution(df)
        
        # 5. Word clouds comparativos
        if 'review' in df.columns and 'sentiment' in df.columns:
            print("📊 Generando word clouds...")
            figures['word_clouds'] = plot_sentiment_word_clouds(df)
            
            # 6. Distribución de longitud de textos
            print("📊 Generando distribución de longitud de textos...")
            figures['text_length_distribution'] = plot_text_length_distribution(df)
    
    print(f"\n✅ Reporte completo generado: {len(figures)} visualizaciones")
    
    return figures


def save_all_figures(figures: Dict[str, plt.Figure],
                    output_dir: str = 'visualizations',
                    dpi: int = 300,
                    format: str = 'png') -> None:
    """Guarda todas las figuras en un directorio.
    
    Args:
        figures: Diccionario de figuras de create_comprehensive_report()
        output_dir: Directorio donde guardar las figuras
        dpi: Resolución de las imágenes (dots per inch)
        format: Formato de archivo ('png', 'jpg', 'pdf', 'svg')
    
    Examples:
        >>> figures = create_comprehensive_report(results, models, X_test, y_test)
        >>> save_all_figures(figures, output_dir='report_images', dpi=300)
        >>> 
        >>> # Limpiar memoria
        >>> for fig in figures.values():
        ...     plt.close(fig)
    
    Notes:
        - Crea el directorio si no existe
        - Usa bbox_inches='tight' para evitar recortes
        - Cierra las figuras automáticamente después de guardar
    """
    import os
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n💾 Guardando {len(figures)} figuras en '{output_dir}/'...")
    
    for name, fig in figures.items():
        output_path = os.path.join(output_dir, f'{name}.{format}')
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', format=format)
        print(f"   ✅ {output_path}")
        plt.close(fig)
    
    print(f"\n✅ Todas las figuras guardadas exitosamente")


if __name__ == "__main__":
    print("="*80)
    print("📊 MÓDULO DE VISUALIZACIONES - Machine Learning Workshop")
    print("="*80)
    print("\nEste módulo proporciona funciones para crear visualizaciones de:")
    print("  1. Comparación de rendimiento de modelos (barras)")
    print("  2. Matrices de confusión (heatmaps)")
    print("  3. Curvas ROC (multi-modelo)")
    print("  4. Word clouds (positivos/negativos)")
    print("  5. Distribución de sentimientos")
    print("  6. Distribución de longitud de textos")
    print("  7. Feature importance (modelos basados en árboles)")
    print("  8. Learning curves")
    print("\n💡 Todas las funciones retornan figuras de matplotlib para máxima flexibilidad")
    print("="*80)
