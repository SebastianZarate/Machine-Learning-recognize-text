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
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

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


def plot_metrics_comparison(eval_results: Dict[str, Dict[str, Any]],
                           metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
                           figsize: Tuple[int, int] = (12, 6),
                           title: str = 'Model Performance Comparison') -> plt.Figure:
    """Muestra barras comparando accuracy, precision, recall, F1 de todos los modelos.
    
    Una visualización side-by-side de métricas facilita identificar el mejor
    modelo de un vistazo. Es más intuitivo que una tabla de números.
    
    Args:
        eval_results: Diccionario de resultados de evaluate_all_models()
            {model_name: {metrics_dict}}
        metrics: Lista de métricas a comparar
            Default: ['accuracy', 'precision', 'recall', 'f1_score']
        figsize: Tamaño de la figura (ancho, alto) en pulgadas
        title: Título del gráfico
    
    Returns:
        plt.Figure: Figura de matplotlib que puede ser guardada o mostrada
    
    Examples:
        >>> # Después de evaluar modelos
        >>> eval_results = evaluate_all_models(models, X_test, y_test)
        >>> 
        >>> # Comparación básica (4 métricas principales)
        >>> fig = plot_metrics_comparison(eval_results)
        >>> fig.savefig('models/metrics_comparison.png', dpi=300, bbox_inches='tight')
        >>> plt.close()
        
        >>> # Comparación personalizada
        >>> fig = plot_metrics_comparison(
        ...     eval_results,
        ...     metrics=['accuracy', 'recall', 'f1_score'],
        ...     title='Comparación de Modelos - Métricas Clave'
        ... )
        >>> plt.show()
    
    Notes:
        **Punto crítico**: Escala de 0 a 1 para todas las métricas.
        Si incluyes accuracy que es 0.95 y recall que es 0.60, ambas deben
        verse en la misma escala para comparar visualmente. Esta función
        garantiza que todas las métricas se muestren en el rango 0-1.
        
        **Por qué es importante:**
        - Comparación visual más fácil que tablas de números
        - Identifica el mejor modelo de un vistazo
        - Muestra fortalezas y debilidades de cada modelo
        - Útil para presentaciones y reportes
    """
    model_names = list(eval_results.keys())
    n_metrics = len(metrics)
    n_models = len(model_names)
    
    # Preparar datos
    data = {metric: [] for metric in metrics}
    for model_name in model_names:
        for metric in metrics:
            value = eval_results[model_name].get(metric)
            # Manejar valores None (ej: ROC-AUC para algunos modelos)
            if value is None:
                value = 0.0
            data[metric].append(value)
    
    # Configurar barras
    x = np.arange(n_models)
    width = 0.8 / n_metrics
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Dibujar barras para cada métrica
    for idx, metric in enumerate(metrics):
        offset = (idx - n_metrics/2 + 0.5) * width
        bars = ax.bar(
            x + offset, 
            data[metric], 
            width,
            label=metric.replace('_', ' ').title(),
            alpha=0.85
        )
        
        # Agregar valores sobre barras
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Solo mostrar si no es 0
                ax.text(
                    bar.get_x() + bar.get_width()/2., 
                    height,
                    f'{height:.3f}',
                    ha='center', 
                    va='bottom', 
                    fontsize=8
                )
    
    # Configurar ejes y etiquetas
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend(loc='lower right', fontsize=10)
    
    # Punto crítico: Escala 0 a 1 para todas las métricas
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
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


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         labels: List[str] = ['Negative', 'Positive'],
                         title: str = 'Confusion Matrix',
                         normalize: bool = False,
                         figsize: Tuple[int, int] = (8, 6),
                         cmap: str = 'Blues') -> plt.Figure:
    """Dibuja matriz de confusión para un solo modelo con opciones avanzadas.
    
    Esta es la visualización más importante para clasificación binaria.
    Muestra verdaderos positivos, falsos positivos, verdaderos negativos,
    y falsos negativos de forma intuitiva.
    
    Args:
        y_true: Etiquetas reales (ground truth)
        y_pred: Etiquetas predichas por el modelo
        labels: Lista de nombres para las clases ['Clase0', 'Clase1']
        title: Título del gráfico
        normalize: Si True, muestra porcentajes; si False, muestra conteos
        figsize: Tamaño de la figura (ancho, alto) en pulgadas
        cmap: Mapa de colores ('Blues', 'Greens', 'Reds', 'YlOrRd', etc.)
    
    Returns:
        plt.Figure: Figura de matplotlib que puede ser guardada o mostrada
    
    Examples:
        >>> from src.model import predict_text
        >>> # Predecir con el modelo
        >>> y_pred = [model.predict(x) for x in X_test]
        >>> 
        >>> # Matriz con conteos absolutos
        >>> fig = plot_confusion_matrix(y_test, y_pred, normalize=False)
        >>> fig.savefig('confusion_counts.png', dpi=300, bbox_inches='tight')
        >>> 
        >>> # Matriz normalizada (porcentajes)
        >>> fig = plot_confusion_matrix(
        ...     y_test, 
        ...     y_pred,
        ...     labels=['Negativo', 'Positivo'],
        ...     title='Matriz de Confusión Normalizada',
        ...     normalize=True
        ... )
        >>> fig.savefig('confusion_normalized.png', dpi=300)
    
    Notes:
        - **Verdaderos Positivos (TP)**: [1,1] - Correctamente predicho como positivo
        - **Falsos Positivos (FP)**: [0,1] - Incorrectamente predicho como positivo
        - **Verdaderos Negativos (TN)**: [0,0] - Correctamente predicho como negativo
        - **Falsos Negativos (FN)**: [1,0] - Incorrectamente predicho como negativo
        - Normalizar es útil cuando hay desbalanceo de clases
        - La diagonal principal debe tener valores altos (predicciones correctas)
    """
    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalizar si se solicita
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        cbar_label = 'Proporción'
    else:
        cm_display = cm
        fmt = 'd'
        cbar_label = 'Cantidad'
    
    # Crear figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Crear heatmap
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': cbar_label},
        ax=ax,
        square=True,
        linewidths=1,
        linecolor='gray',
        vmin=0,
        vmax=1 if normalize else None
    )
    
    # Configurar etiquetas y título
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Agregar estadísticas en la parte inferior
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    stats_text = f'Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}'
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    return fig


def plot_all_confusion_matrices(models: Dict[str, Any],
                                X_test: np.ndarray,
                                y_test: np.ndarray,
                                labels: List[str] = ['Negative', 'Positive'],
                                figsize: Optional[Tuple[int, int]] = None,
                                cmap: str = 'Blues',
                                normalize: bool = False) -> plt.Figure:
    """Wrapper para dibujar matrices de confusión de todos los modelos.
    
    Crea una figura con múltiples subplots, uno para cada modelo,
    mostrando su matriz de confusión. Alternativa más flexible a
    plot_confusion_matrices() con opciones adicionales.
    
    Args:
        models: Diccionario {model_name: model_object}
        X_test: Datos de prueba (features)
        y_test: Etiquetas reales de prueba
        labels: Lista de nombres para las clases
        figsize: Tamaño de la figura (se calcula automático si None)
        cmap: Mapa de colores para los heatmaps
        normalize: Si True, muestra porcentajes; si False, conteos
    
    Returns:
        plt.Figure: Figura de matplotlib con múltiples matrices
    
    Examples:
        >>> models = {
        ...     'Naive Bayes': nb_model,
        ...     'Logistic Regression': lr_model,
        ...     'Random Forest': rf_model
        ... }
        >>> 
        >>> # Matrices con conteos absolutos
        >>> fig = plot_all_confusion_matrices(models, X_test, y_test)
        >>> fig.savefig('all_confusion_matrices.png', dpi=300)
        >>> 
        >>> # Matrices normalizadas
        >>> fig = plot_all_confusion_matrices(
        ...     models, 
        ...     X_test, 
        ...     y_test,
        ...     labels=['Negativo', 'Positivo'],
        ...     normalize=True,
        ...     cmap='YlOrRd'
        ... )
        >>> plt.show()
    
    Notes:
        - Cada subplot muestra la matriz de un modelo diferente
        - Los modelos se disponen en una fila horizontalmente
        - Si hay más de 4 modelos, se crean múltiples filas
        - Útil para comparar rápidamente el comportamiento de los modelos
    """
    n_models = len(models)
    
    # Calcular tamaño de figura si no se especifica
    if figsize is None:
        width = 6 * min(n_models, 3)  # Máximo 3 columnas
        height = 5 * ((n_models + 2) // 3)  # Filas necesarias
        figsize = (width, height)
    
    # Calcular grid de subplots
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    # Crear figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Asegurar que axes sea siempre un array
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_models > 1 else [axes]
    
    # Dibujar matriz para cada modelo
    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]
        
        # Hacer predicciones
        y_pred = model.predict(X_test)
        
        # Calcular matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        
        # Normalizar si se solicita
        if normalize:
            cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        else:
            cm_display = cm
            fmt = 'd'
        
        # Crear heatmap
        sns.heatmap(
            cm_display,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            ax=ax,
            cbar=True,
            xticklabels=labels,
            yticklabels=labels,
            square=True,
            linewidths=1,
            linecolor='gray',
            vmin=0,
            vmax=1 if normalize else None
        )
        
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
    
    # Ocultar subplots vacíos
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    # Título general
    title = 'Matrices de Confusión ' + ('Normalizadas' if normalize else '(Conteos Absolutos)')
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    
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


def plot_roc_curves(models: Dict[str, Any],
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   title: str = 'ROC Curves Comparison',
                   figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """Grafica curvas ROC para todos los modelos en un solo plot.
    
    La curva ROC (Receiver Operating Characteristic) muestra el trade-off 
    entre tasa de verdaderos positivos (TPR/Recall) y falsos positivos (FPR)
    a diferentes thresholds de clasificación. Es esencial para comparar 
    modelos, especialmente cuando las clases están desbalanceadas.
    
    Args:
        models: Diccionario {model_name: model_object}
        X_test: Datos de prueba (features)
        y_test: Etiquetas reales de prueba
        title: Título del gráfico
        figsize: Tamaño de la figura (ancho, alto) en pulgadas
    
    Returns:
        plt.Figure: Figura de matplotlib con curvas ROC
    
    Examples:
        >>> models = {
        ...     'Naive Bayes': nb_model,
        ...     'Logistic Regression': lr_model,
        ...     'Random Forest': rf_model
        ... }
        >>> fig = plot_roc_curves(models, X_test, y_test)
        >>> fig.savefig('roc_comparison.png', dpi=300, bbox_inches='tight')
        >>> plt.show()
        
        >>> # Con título personalizado
        >>> fig = plot_roc_curves(
        ...     models, 
        ...     X_test, 
        ...     y_test,
        ...     title='Comparación ROC - Clasificación de Sentimientos'
        ... )
    
    Notes:
        **Interpretación de AUC (Area Under Curve):**
        - AUC = 0.5: Rendimiento aleatorio (no mejor que lanzar moneda)
        - AUC = 0.5 - 0.7: Rendimiento pobre
        - AUC = 0.7 - 0.8: Rendimiento aceptable
        - AUC = 0.8 - 0.9: Rendimiento bueno
        - AUC > 0.9: Rendimiento excelente
        - AUC = 1.0: Clasificador perfecto
        
        **Puntos clave:**
        - La línea diagonal representa un clasificador aleatorio
        - Cuanto más cerca de la esquina superior izquierda, mejor
        - Solo funciona con modelos que tienen predict_proba()
        - Útil para comparar modelos cuando hay desbalanceo de clases
        - Permite elegir threshold óptimo según necesidades del negocio
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, model in models.items():
        # Verificar que el modelo tiene predict_proba
        if not hasattr(model, 'predict_proba'):
            print(f"Skipping {name}: no predict_proba method")
            continue
        
        # Obtener probabilidades predichas
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calcular curva ROC
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        
        # Calcular AUC
        auc_score = roc_auc_score(y_test, y_proba)
        
        # Dibujar curva ROC
        ax.plot(
            fpr, 
            tpr, 
            linewidth=2,
            label=f'{name} (AUC = {auc_score:.3f})',
            alpha=0.8
        )
    
    # Línea diagonal (clasificador aleatorio)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, 
            label='Random Classifier (AUC = 0.500)',
            alpha=0.6)
    
    # Configurar ejes y etiquetas
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Configurar límites
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
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


def generate_wordcloud(texts: List[str],
                      title: str = 'Word Cloud',
                      max_words: int = 100,
                      figsize: Tuple[int, int] = (12, 6),
                      background_color: str = 'white',
                      colormap: str = 'viridis') -> plt.Figure:
    """Visualiza términos más frecuentes en forma de nube de palabras.
    
    Esta función genera un WordCloud que es una forma intuitiva de visualizar
    qué palabras dominan en un conjunto de textos. El tamaño de cada palabra
    representa su frecuencia de aparición.
    
    Args:
        texts: Lista de strings (textos sin procesar o mínimamente procesados)
        title: Título del gráfico
        max_words: Número máximo de palabras a mostrar (default: 100)
        figsize: Tamaño de la figura (ancho, alto) en pulgadas
        background_color: Color de fondo ('white', 'black', etc.)
        colormap: Mapa de colores ('viridis', 'plasma', 'inferno', 'cool', etc.)
    
    Returns:
        plt.Figure: Figura de matplotlib que puede ser guardada o mostrada
    
    Examples:
        >>> # Generar WordCloud básico
        >>> texts = ["Este es un texto", "Otro texto más", "Más ejemplos"]
        >>> fig = generate_wordcloud(texts, title='Términos Frecuentes')
        >>> fig.savefig('wordcloud.png', dpi=300, bbox_inches='tight')
        >>> plt.show()
        
        >>> # WordCloud con configuración personalizada
        >>> fig = generate_wordcloud(
        ...     reviews_list,
        ...     title='Palabras Clave en Reseñas',
        ...     max_words=150,
        ...     background_color='black',
        ...     colormap='plasma'
        ... )
    
    Notes:
        **Punto crítico**: Generar WordClouds sobre textos SIN preprocesar
        (o con preprocesamiento mínimo) hace más legibles los resultados.
        Si usas textos lemmatizados, las palabras pueden verse extrañas
        ("movi" en lugar de "movie").
        
        - Las palabras más grandes = mayor frecuencia
        - Útil para entender qué aprendió el modelo
        - Ayuda a validar que los patrones tienen sentido
    """
    # Concatenar todos los textos
    all_text = ' '.join(texts)
    
    # Configurar WordCloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color=background_color,
        max_words=max_words,
        colormap=colormap,
        relative_scaling=0.5,
        min_font_size=10,
        collocations=False  # Evitar duplicados de bigramas
    ).generate(all_text)
    
    # Plotear
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.axis('off')
    
    plt.tight_layout()
    
    return fig


def plot_wordclouds_by_class(texts: List[str],
                             labels: List[int],
                             class_names: List[str] = ['Not Review', 'Review'],
                             figsize: Tuple[int, int] = (16, 6),
                             max_words: int = 100) -> plt.Figure:
    """Genera dos WordClouds: uno para cada clase (reseñas vs no-reseñas).
    
    Esta función visualiza los términos más frecuentes para cada clase por
    separado, ayudando a entender qué palabras dominan cada categoría y
    validar que el modelo captura patrones con sentido.
    
    Args:
        texts: Lista de strings (textos originales)
        labels: Lista de etiquetas (0 o 1)
        class_names: Nombres para las clases [clase_0, clase_1]
        figsize: Tamaño de la figura (ancho, alto) en pulgadas
        max_words: Número máximo de palabras por WordCloud
    
    Returns:
        plt.Figure: Figura de matplotlib con dos subplots lado a lado
    
    Examples:
        >>> # Uso básico con nombres de clase por defecto
        >>> texts = df['text'].tolist()
        >>> labels = df['label'].tolist()
        >>> fig = plot_wordclouds_by_class(texts, labels)
        >>> fig.savefig('models/wordclouds.png', dpi=300, bbox_inches='tight')
        >>> plt.close()
        
        >>> # Con nombres de clase personalizados
        >>> fig = plot_wordclouds_by_class(
        ...     texts,
        ...     labels,
        ...     class_names=['Negative', 'Positive'],
        ...     max_words=150
        ... )
        >>> plt.show()
        
        >>> # Integración en flujo de entrenamiento
        >>> # En train_from_csv(), después de cargar datos:
        >>> from visualizations import plot_wordclouds_by_class
        >>> 
        >>> # Obtener textos originales (sin procesar)
        >>> fig = plot_wordclouds_by_class(
        ...     df['text'].tolist(),
        ...     df['label'].tolist(),
        ...     class_names=['Negative', 'Positive']
        ... )
        >>> fig.savefig('models/wordclouds.png', dpi=300, bbox_inches='tight')
        >>> plt.close()
    
    Notes:
        **Punto crítico**: Usar textos SIN preprocesar o con preprocesamiento
        mínimo para mejor legibilidad. Los textos lemmatizados pueden verse
        extraños ("movi" en lugar de "movie", "plai" en lugar de "play").
        
        **Por qué es importante:**
        - Ayuda a entender qué aprendió el modelo
        - Valida que los patrones tienen sentido semántico
        - Detecta problemas como stopwords no removidas
        - Visualiza diferencias clave entre clases
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    for idx, class_label in enumerate([0, 1]):
        # Filtrar textos de esta clase
        class_texts = [texts[i] for i in range(len(texts)) 
                      if labels[i] == class_label]
        
        if not class_texts:
            axes[idx].text(0.5, 0.5, 'No hay datos para esta clase',
                          ha='center', va='center', fontsize=14)
            axes[idx].axis('off')
            continue
        
        # Generar WordCloud
        all_text = ' '.join(class_texts)
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=max_words,
            colormap='viridis' if idx == 0 else 'plasma',
            relative_scaling=0.5,
            min_font_size=10,
            collocations=False
        ).generate(all_text)
        
        # Plotear
        axes[idx].imshow(wordcloud, interpolation='bilinear')
        axes[idx].set_title(f'WordCloud: {class_names[idx]}',
                           fontsize=14, fontweight='bold')
        axes[idx].axis('off')
    
    plt.suptitle('Comparación de Términos por Clase',
                 fontsize=16, fontweight='bold', y=1.00)
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


def plot_class_distribution(labels: List[int],
                           class_names: List[str] = ['Not Review', 'Review'],
                           figsize: Tuple[int, int] = (8, 6),
                           title: str = 'Class Distribution') -> plt.Figure:
    """Visualiza la distribución de clases en el dataset.
    
    Entender el dataset es fundamental. Si hay desbalanceo de clases,
    esto afecta el modelo y debe documentarse. Esta función muestra
    el conteo y porcentaje de cada clase de forma clara.
    
    Args:
        labels: Lista de etiquetas (0 o 1)
        class_names: Nombres para las clases [clase_0, clase_1]
        figsize: Tamaño de la figura (ancho, alto) en pulgadas
        title: Título del gráfico
    
    Returns:
        plt.Figure: Figura de matplotlib con gráfico de barras
    
    Examples:
        >>> # Visualizar distribución de clases
        >>> labels = df['label'].tolist()
        >>> fig = plot_class_distribution(labels)
        >>> fig.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        >>> plt.show()
        
        >>> # Con nombres personalizados
        >>> fig = plot_class_distribution(
        ...     labels,
        ...     class_names=['Negative', 'Positive'],
        ...     title='Sentiment Distribution'
        ... )
    
    Notes:
        **Por qué es importante:**
        - Detecta desbalanceo de clases (ej: 90% clase 1, 10% clase 0)
        - Desbalanceo afecta métricas (accuracy puede ser engañoso)
        - Ayuda a decidir si usar técnicas de balanceo (SMOTE, undersampling)
        - Documenta características del dataset
        
        **Interpretación:**
        - 50/50: Dataset balanceado (ideal)
        - 60/40: Leve desbalanceo (aceptable)
        - 70/30: Desbalanceo moderado (considerar técnicas de balanceo)
        - 80/20+: Desbalanceo severo (requiere atención especial)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calcular conteos
    counts = pd.Series(labels).value_counts().sort_index()
    colors = ['#e74c3c', '#27ae60']  # Rojo para clase 0, verde para clase 1
    
    # Crear barras
    bars = ax.bar(class_names, counts, color=colors, alpha=0.7, 
                  edgecolor='black', linewidth=1.5)
    
    # Agregar porcentajes y conteos
    total = len(labels)
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        percentage = (height / total) * 100
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height,
            f'{int(height)}\n({percentage:.1f}%)',
            ha='center', 
            va='bottom', 
            fontsize=12, 
            fontweight='bold'
        )
    
    # Línea de referencia (50%)
    ax.axhline(y=total/2, color='gray', linestyle='--', linewidth=2, 
               alpha=0.5, label='Perfect Balance (50%)')
    
    # Configurar ejes
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    return fig


def plot_text_length_distribution(texts: List[str],
                                  labels: List[int],
                                  class_names: List[str] = ['Not Review', 'Review'],
                                  figsize: Tuple[int, int] = (14, 5),
                                  bins: int = 50,
                                  title: str = 'Text Length Distribution by Class') -> plt.Figure:
    """Visualiza distribución de longitud de textos por clase.
    
    Si las longitudes promedio son muy diferentes entre clases, el modelo
    podría estar "haciendo trampa" clasificando solo por longitud en lugar
    de contenido semántico. Esta función documenta estas diferencias.
    
    Args:
        texts: Lista de strings (textos)
        labels: Lista de etiquetas (0 o 1)
        class_names: Nombres para las clases [clase_0, clase_1]
        figsize: Tamaño de la figura (ancho, alto) en pulgadas
        bins: Número de bins para histogramas
        title: Título general del gráfico
    
    Returns:
        plt.Figure: Figura de matplotlib con dos subplots (histogramas)
    
    Examples:
        >>> # Análisis básico de longitudes
        >>> texts = df['text'].tolist()
        >>> labels = df['label'].tolist()
        >>> fig = plot_text_length_distribution(texts, labels)
        >>> fig.savefig('text_lengths.png', dpi=300, bbox_inches='tight')
        >>> plt.show()
        
        >>> # Con nombres personalizados
        >>> fig = plot_text_length_distribution(
        ...     texts,
        ...     labels,
        ...     class_names=['Negative', 'Positive'],
        ...     bins=30
        ... )
    
    Notes:
        **Punto crítico**: Si las longitudes promedio son muy diferentes
        entre clases (ej: reseñas 200 palabras, no-reseñas 50 palabras),
        el modelo podría estar "haciendo trampa" clasificando solo por
        longitud en lugar de contenido.
        
        **Interpretación:**
        - Medias similares (±10%): Bueno, modelo usa contenido
        - Medias diferentes (±30%): Precaución, verificar feature importance
        - Medias muy diferentes (±50%+): Problema, modelo usa longitud como atajo
        
        **Qué hacer si hay diferencias grandes:**
        - Normalizar longitudes en preprocesamiento
        - Verificar que vectorizador no use features relacionadas con longitud
        - Analizar feature importance para confirmar
        - Considerar balancear longitudes entre clases
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Calcular longitudes (número de palabras)
    lengths_by_class = {0: [], 1: []}
    for text, label in zip(texts, labels):
        word_count = len(str(text).split())
        lengths_by_class[label].append(word_count)
    
    # Histogramas para cada clase
    for idx, class_label in enumerate([0, 1]):
        lengths = lengths_by_class[class_label]
        
        # Crear histograma
        axes[idx].hist(
            lengths, 
            bins=bins,
            color='#3498db' if idx == 1 else '#e74c3c',
            alpha=0.7, 
            edgecolor='black',
            linewidth=0.5
        )
        
        # Calcular estadísticas
        mean_len = np.mean(lengths)
        median_len = np.median(lengths)
        std_len = np.std(lengths)
        
        # Líneas de referencia
        axes[idx].axvline(
            mean_len, 
            color='red', 
            linestyle='--',
            linewidth=2, 
            label=f'Mean: {mean_len:.0f}'
        )
        axes[idx].axvline(
            median_len, 
            color='green', 
            linestyle='--',
            linewidth=2, 
            label=f'Median: {median_len:.0f}'
        )
        
        # Configurar subplot
        axes[idx].set_xlabel('Number of Words', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[idx].set_title(
            f'Text Length: {class_names[idx]}\n(σ={std_len:.1f})',
            fontsize=12, 
            fontweight='bold'
        )
        axes[idx].legend(loc='upper right', fontsize=10)
        axes[idx].grid(alpha=0.3, linestyle='--')
    
    # Título general
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Advertencia si diferencias son grandes
    mean_0 = np.mean(lengths_by_class[0])
    mean_1 = np.mean(lengths_by_class[1])
    diff_percentage = abs(mean_0 - mean_1) / max(mean_0, mean_1) * 100
    
    if diff_percentage > 30:
        warning_text = (f'⚠️ WARNING: Large difference in mean lengths '
                       f'({diff_percentage:.1f}%). Model may use length as shortcut!')
        fig.text(0.5, 0.01, warning_text, ha='center', fontsize=10, 
                color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
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
