"""
Module for comprehensive model evaluation and performance metrics.

This module provides functions to evaluate classification models using
standard ML metrics: accuracy, precision, recall, F1-score, ROC-AUC, etc.

Includes visualization utilities for confusion matrices and ROC curves.

Author: Machine Learning Workshop
Date: 2025-10-29
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)


def evaluate_model(model: Any,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   model_name: str = 'Model') -> Dict[str, Any]:
    """Evalúa un modelo y retorna todas las métricas de rendimiento.
    
    Calcula métricas completas para un clasificador binario:
    - Accuracy: Proporción de predicciones correctas
    - Precision: Proporción de positivos predichos que son correctos
    - Recall: Proporción de positivos reales que se detectaron
    - F1-Score: Media armónica de precision y recall
    - ROC-AUC: Área bajo la curva ROC (si predict_proba disponible)
    - Confusion Matrix: Tabla de TP, TN, FP, FN
    - Classification Report: Resumen detallado por clase
    
    Args:
        model: Modelo entrenado con métodos predict() y opcionalmente predict_proba()
        X_test: Features de test (matriz TF-IDF), shape (n_samples, n_features)
        y_test: Etiquetas verdaderas de test (0/1), shape (n_samples,)
        model_name: Nombre del modelo para identificación (default: 'Model')
    
    Returns:
        Diccionario con todas las métricas:
        {
            'model_name': str,
            'accuracy': float,          # [0, 1]
            'precision': float,         # [0, 1]
            'recall': float,            # [0, 1]
            'f1_score': float,          # [0, 1]
            'specificity': float,       # [0, 1] - True Negative Rate
            'report': str,              # Classification report formateado
            'confusion_matrix': ndarray,  # [[TN, FP], [FN, TP]]
            'roc_auc': float or None,   # [0, 1] si predict_proba disponible
            'roc_curve': tuple or None, # (fpr, tpr, thresholds) si disponible
            'support': dict             # Número de muestras por clase
        }
    
    Examples:
        >>> from sklearn.linear_model import LogisticRegression
        >>> model = LogisticRegression().fit(X_train, y_train)
        >>> metrics = evaluate_model(model, X_test, y_test, "Logistic Regression")
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
        >>> print(f"F1-Score: {metrics['f1_score']:.4f}")
        >>> print(metrics['report'])
        
        >>> # Verificar si tiene ROC-AUC
        >>> if metrics['roc_auc']:
        ...     print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    Notes:
        - Precision: TP / (TP + FP) - Importante cuando FP son costosos
        - Recall: TP / (TP + FN) - Importante cuando FN son costosos
        - F1-Score: 2 * (precision * recall) / (precision + recall)
        - Specificity: TN / (TN + FP) - True Negative Rate
        - ROC-AUC requiere predict_proba (no todos los modelos lo tienen)
        - Confusion Matrix: [[TN, FP], [FN, TP]]
    
    Performance:
        - Random Forest SÍ tiene predict_proba (probabilidad por árbol)
        - Naive Bayes y Logistic Regression también lo tienen
        - Si un modelo no tiene predict_proba, ROC-AUC será None
    """
    # ========== 1. PREDICCIONES ==========
    y_pred = model.predict(X_test)
    
    # ========== 2. MÉTRICAS BÁSICAS ==========
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # ========== 3. CONFUSION MATRIX ==========
    cm = confusion_matrix(y_test, y_pred)
    # cm = [[TN, FP],
    #       [FN, TP]]
    
    # Calcular Specificity (True Negative Rate)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # ========== 4. CLASSIFICATION REPORT ==========
    report = classification_report(
        y_test, 
        y_pred,
        target_names=['Not Review (0)', 'Review (1)'],
        digits=4,
        zero_division=0
    )
    
    # ========== 5. SUPPORT (muestras por clase) ==========
    unique, counts = np.unique(y_test, return_counts=True)
    support = {int(label): int(count) for label, count in zip(unique, counts)}
    
    # ========== 6. ROC-AUC Y ROC CURVE (si predict_proba disponible) ==========
    roc_auc = None
    roc_curve_data = None
    
    if hasattr(model, 'predict_proba'):
        try:
            # predict_proba retorna [[prob_class_0, prob_class_1], ...]
            y_proba = model.predict_proba(X_test)[:, 1]  # Probabilidades clase 1
            
            # Calcular ROC-AUC
            roc_auc = roc_auc_score(y_test, y_proba)
            
            # Calcular curva ROC
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            roc_curve_data = (fpr, tpr, thresholds)
            
        except Exception as e:
            print(f"⚠️  Warning: No se pudo calcular ROC-AUC para {model_name}: {e}")
            roc_auc = None
            roc_curve_data = None
    
    # ========== 7. RETORNAR RESULTADOS ==========
    return {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'report': report,
        'confusion_matrix': cm,
        'roc_auc': float(roc_auc) if roc_auc is not None else None,
        'roc_curve': roc_curve_data,
        'support': support
    }


def evaluate_all_models(models: Dict[str, Any],
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """Evalúa múltiples modelos y retorna métricas para cada uno.
    
    Itera sobre un diccionario de modelos entrenados y calcula métricas
    completas para cada uno. Útil para comparar rendimiento de diferentes
    algoritmos.
    
    Args:
        models: Diccionario {nombre_modelo: modelo_entrenado}
        X_test: Features de test (matriz TF-IDF)
        y_test: Etiquetas verdaderas de test
        verbose: Si True, imprime tabla comparativa (default: True)
    
    Returns:
        Diccionario anidado con métricas por modelo:
        {
            'Naive Bayes': {
                'accuracy': 0.85,
                'precision': 0.86,
                'recall': 0.84,
                'f1_score': 0.85,
                'roc_auc': 0.92,
                ...
            },
            'Logistic Regression': {...},
            'Random Forest': {...}
        }
    
    Examples:
        >>> models = {
        ...     'Naive Bayes': nb_model,
        ...     'Logistic Regression': lr_model,
        ...     'Random Forest': rf_model
        ... }
        >>> results = evaluate_all_models(models, X_test, y_test)
        >>> 
        >>> # Comparar F1-scores
        >>> for name, metrics in results.items():
        ...     print(f"{name}: F1={metrics['f1_score']:.4f}")
        >>> 
        >>> # Encontrar mejor modelo
        >>> best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
        >>> print(f"Mejor modelo: {best_model[0]}")
    
    Notes:
        - Todos los modelos se evalúan en el mismo test set
        - La tabla comparativa se ordena por F1-score (descendente)
        - Modelos sin predict_proba tendrán ROC-AUC = None
    """
    results = {}
    
    print("\n" + "="*80)
    print("📊 EVALUACIÓN DE MODELOS")
    print("="*80 + "\n")
    
    for model_name, model in models.items():
        if verbose:
            print(f"Evaluando {model_name}...")
        
        metrics = evaluate_model(model, X_test, y_test, model_name)
        results[model_name] = metrics
        
        if verbose:
            print(f"  ✓ Accuracy: {metrics['accuracy']:.4f}")
            print(f"  ✓ F1-Score: {metrics['f1_score']:.4f}")
            if metrics['roc_auc']:
                print(f"  ✓ ROC-AUC:  {metrics['roc_auc']:.4f}")
            print()
    
    # ========== TABLA COMPARATIVA ==========
    if verbose:
        print("="*80)
        print("📈 TABLA COMPARATIVA DE MODELOS")
        print("="*80)
        print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
        print("-"*80)
        
        # Ordenar por F1-score (descendente)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        for model_name, metrics in sorted_results:
            roc_str = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "N/A"
            print(f"{model_name:<25} "
                  f"{metrics['accuracy']:<10.4f} "
                  f"{metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} "
                  f"{metrics['f1_score']:<10.4f} "
                  f"{roc_str:<10}")
        
        # Identificar mejor modelo
        best_model = sorted_results[0]
        print("\n" + "="*80)
        print(f"🏆 MEJOR MODELO: {best_model[0]}")
        print(f"   F1-Score: {best_model[1]['f1_score']:.4f}")
        print(f"   Accuracy: {best_model[1]['accuracy']:.4f}")
        if best_model[1]['roc_auc']:
            print(f"   ROC-AUC:  {best_model[1]['roc_auc']:.4f}")
        print("="*80 + "\n")
    
    return results


def print_detailed_metrics(metrics: Dict[str, Any]) -> None:
    """Imprime métricas detalladas de un modelo en formato legible.
    
    Args:
        metrics: Diccionario de métricas retornado por evaluate_model()
    
    Examples:
        >>> metrics = evaluate_model(model, X_test, y_test, "My Model")
        >>> print_detailed_metrics(metrics)
    """
    print("\n" + "="*80)
    print(f"📊 MÉTRICAS DETALLADAS: {metrics['model_name']}")
    print("="*80)
    
    # Métricas principales
    print("\n🎯 Métricas principales:")
    print(f"   Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   Precision:   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"   Recall:      {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"   F1-Score:    {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"   Specificity: {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
    
    if metrics['roc_auc']:
        print(f"   ROC-AUC:     {metrics['roc_auc']:.4f} ({metrics['roc_auc']*100:.2f}%)")
    
    # Support
    print(f"\n📦 Muestras por clase:")
    for label, count in metrics['support'].items():
        label_name = "Not Review" if label == 0 else "Review"
        print(f"   {label_name} ({label}): {count:,} muestras")
    
    # Confusion Matrix
    cm = metrics['confusion_matrix']
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n🔍 Matriz de Confusión:")
    print(f"                    Predicted Negative  Predicted Positive")
    print(f"   Actual Negative       {tn:6d}             {fp:6d}")
    print(f"   Actual Positive       {fn:6d}             {tp:6d}")
    
    # Interpretación
    print(f"\n💡 Interpretación:")
    print(f"   True Negatives (TN):  {tn:,} - Correctamente identificados como Negative (sentimiento negativo)")
    print(f"   True Positives (TP):  {tp:,} - Correctamente identificados como Positive (sentimiento positivo)")
    print(f"   False Positives (FP): {fp:,} - Incorrectamente identificados como Positive")
    print(f"   False Negatives (FN): {fn:,} - Incorrectamente identificados como Negative")
    
    # Classification Report
    print(f"\n📋 Classification Report:")
    print(metrics['report'])
    
    print("="*80 + "\n")


def plot_confusion_matrix(cm: np.ndarray,
                         model_name: str,
                         save_path: Optional[str] = None) -> None:
    """Visualiza la matriz de confusión con un heatmap.
    
    Args:
        cm: Matriz de confusión (2x2 array)
        model_name: Nombre del modelo para el título
        save_path: Ruta opcional para guardar la figura (default: None)
    
    Examples:
        >>> metrics = evaluate_model(model, X_test, y_test, "My Model")
        >>> plot_confusion_matrix(metrics['confusion_matrix'], "My Model")
        >>> plt.show()
        
        >>> # Guardar figura
        >>> plot_confusion_matrix(cm, "My Model", "confusion_matrix.png")
    """
    plt.figure(figsize=(8, 6))
    
    # Crear heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Not Review (0)', 'Review (1)'],
        yticklabels=['Not Review (0)', 'Review (1)'],
        cbar_kws={'label': 'Count'}
    )
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figura guardada en: {save_path}")
    
    plt.show()


def plot_roc_curves(results: Dict[str, Dict[str, Any]],
                   save_path: Optional[str] = None) -> None:
    """Visualiza curvas ROC de múltiples modelos en una sola figura.
    
    Args:
        results: Diccionario de resultados de evaluate_all_models()
        save_path: Ruta opcional para guardar la figura (default: None)
    
    Examples:
        >>> results = evaluate_all_models(models, X_test, y_test)
        >>> plot_roc_curves(results)
        >>> plt.show()
        
        >>> # Guardar figura
        >>> plot_roc_curves(results, "roc_curves.png")
    
    Notes:
        - Solo incluye modelos que tienen predict_proba
        - La línea diagonal representa un clasificador aleatorio (AUC=0.5)
    """
    plt.figure(figsize=(10, 8))
    
    # Colores para cada modelo
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    models_with_roc = []
    
    for idx, (model_name, metrics) in enumerate(results.items()):
        if metrics['roc_auc'] and metrics['roc_curve']:
            fpr, tpr, _ = metrics['roc_curve']
            auc = metrics['roc_auc']
            
            color = colors[idx % len(colors)]
            plt.plot(
                fpr, tpr,
                color=color,
                lw=2,
                label=f"{model_name} (AUC = {auc:.4f})"
            )
            models_with_roc.append(model_name)
    
    # Línea diagonal (clasificador aleatorio)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5000)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figura guardada en: {save_path}")
    
    if not models_with_roc:
        print("⚠️  Warning: Ningún modelo tiene predict_proba. No se puede graficar ROC.")
        plt.close()
        return
    
    plt.show()


def compare_models_table(results: Dict[str, Dict[str, Any]]) -> None:
    """Imprime tabla comparativa detallada de todos los modelos.
    
    Args:
        results: Diccionario de resultados de evaluate_all_models()
    
    Examples:
        >>> results = evaluate_all_models(models, X_test, y_test, verbose=False)
        >>> compare_models_table(results)
    """
    print("\n" + "="*100)
    print("📊 TABLA COMPARATIVA DETALLADA DE MODELOS")
    print("="*100)
    
    # Header
    print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} "
          f"{'F1-Score':<10} {'Specificity':<12} {'ROC-AUC':<10}")
    print("-"*100)
    
    # Ordenar por F1-score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    
    for model_name, metrics in sorted_results:
        roc_str = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "N/A"
        
        print(f"{model_name:<25} "
              f"{metrics['accuracy']:<10.4f} "
              f"{metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} "
              f"{metrics['f1_score']:<10.4f} "
              f"{metrics['specificity']:<12.4f} "
              f"{roc_str:<10}")
    
    print("="*100)
    
    # Mejor modelo por métrica
    print("\n🏆 MEJORES MODELOS POR MÉTRICA:")
    
    best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_precision = max(results.items(), key=lambda x: x[1]['precision'])
    best_recall = max(results.items(), key=lambda x: x[1]['recall'])
    best_f1 = max(results.items(), key=lambda x: x[1]['f1_score'])
    
    print(f"   Accuracy:    {best_accuracy[0]:<25} ({best_accuracy[1]['accuracy']:.4f})")
    print(f"   Precision:   {best_precision[0]:<25} ({best_precision[1]['precision']:.4f})")
    print(f"   Recall:      {best_recall[0]:<25} ({best_recall[1]['recall']:.4f})")
    print(f"   F1-Score:    {best_f1[0]:<25} ({best_f1[1]['f1_score']:.4f})")
    
    # ROC-AUC (solo modelos con predict_proba)
    models_with_roc = {k: v for k, v in results.items() if v['roc_auc']}
    if models_with_roc:
        best_roc = max(models_with_roc.items(), key=lambda x: x[1]['roc_auc'])
        print(f"   ROC-AUC:     {best_roc[0]:<25} ({best_roc[1]['roc_auc']:.4f})")
    
    print("="*100 + "\n")


def compare_models(eval_results: Dict[str, Dict[str, Any]],
                   sort_by: str = 'f1_score',
                   ascending: bool = False) -> 'pd.DataFrame':
    """Crea una tabla comparativa de todos los modelos evaluados.
    
    Genera un DataFrame de pandas con todas las métricas de rendimiento
    para facilitar la comparación lado a lado de múltiples modelos.
    
    Args:
        eval_results: Diccionario de resultados de evaluate_all_models()
            {model_name: {metrics_dict}}
        sort_by: Métrica por la cual ordenar (default: 'f1_score')
            Opciones: 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'
        ascending: Si True, ordena ascendente; si False, descendente (default: False)
    
    Returns:
        DataFrame de pandas con columnas:
        - Model: Nombre del modelo
        - Accuracy: Exactitud
        - Precision: Precisión
        - Recall: Sensibilidad
        - F1-Score: Media armónica
        - Specificity: Tasa de verdaderos negativos
        - ROC-AUC: Área bajo curva ROC (0.0 si no disponible)
    
    Examples:
        >>> results = evaluate_all_models(models, X_test, y_test)
        >>> df = compare_models(results)
        >>> print(df)
                          Model  Accuracy  Precision    Recall  F1-Score  Specificity  ROC-AUC
        0  Logistic Regression    0.8723     0.8801    0.8642    0.8715       0.8804   0.9456
        1          Naive Bayes    0.8542     0.8621    0.8453    0.8536       0.8631   0.9234
        2        Random Forest    0.8401     0.8489    0.8305    0.8392       0.8497   0.9123
        
        >>> # Ordenar por accuracy
        >>> df = compare_models(results, sort_by='accuracy')
        
        >>> # Guardar a CSV
        >>> df.to_csv('model_comparison.csv', index=False)
        
        >>> # Encontrar mejor modelo
        >>> best_model = df.iloc[0]['Model']
        >>> print(f"Mejor modelo: {best_model}")
    
    Notes:
        - F1-Score es la mejor métrica cuando las clases están balanceadas
        - Si hay desbalanceo, considera F1 ponderado o analizar precision/recall por separado
        - ROC-AUC se establece en 0.0 si el modelo no tiene predict_proba
        - La tabla se ordena por defecto de mejor a peor rendimiento
    
    Performance Tips:
        - Para datasets balanceados: Usa F1-Score (default)
        - Para minimizar falsos positivos: Usa Precision
        - Para minimizar falsos negativos: Usa Recall
        - Para evaluar discriminación: Usa ROC-AUC
    """
    import pandas as pd
    
    # Validar sort_by
    valid_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'roc_auc']
    if sort_by not in valid_metrics:
        raise ValueError(
            f"sort_by debe ser uno de {valid_metrics}. "
            f"Recibido: '{sort_by}'"
        )
    
    # Construir lista de diccionarios para DataFrame
    comparison = []
    
    for model_name, metrics in eval_results.items():
        comparison.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'Specificity': metrics['specificity'],
            'ROC-AUC': metrics['roc_auc'] if metrics['roc_auc'] is not None else 0.0
        })
    
    # Crear DataFrame
    df = pd.DataFrame(comparison)
    
    # Mapeo de nombres de columnas para ordenamiento
    column_mapping = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1-Score',
        'specificity': 'Specificity',
        'roc_auc': 'ROC-AUC'
    }
    
    sort_column = column_mapping[sort_by]
    
    # Ordenar DataFrame
    df = df.sort_values(sort_column, ascending=ascending)
    
    # Resetear índice para que sea secuencial
    df = df.reset_index(drop=True)
    
    return df


def print_comparison_table(eval_results: Dict[str, Dict[str, Any]],
                          sort_by: str = 'f1_score',
                          show_best_by_metric: bool = True) -> None:
    """Imprime tabla comparativa de modelos en formato legible.
    
    Wrapper conveniente de compare_models() que imprime directamente
    en lugar de retornar DataFrame.
    
    Args:
        eval_results: Diccionario de resultados de evaluate_all_models()
        sort_by: Métrica por la cual ordenar (default: 'f1_score')
        show_best_by_metric: Si True, muestra mejor modelo por cada métrica
    
    Examples:
        >>> results = evaluate_all_models(models, X_test, y_test)
        >>> print_comparison_table(results)
        
        ==================================================================
        📊 COMPARACIÓN DE MODELOS
        ==================================================================
        
                          Model  Accuracy  Precision    Recall  F1-Score  Specificity  ROC-AUC
        0  Logistic Regression    0.8723     0.8801    0.8642    0.8715       0.8804   0.9456
        1          Naive Bayes    0.8542     0.8621    0.8453    0.8536       0.8631   0.9234
        2        Random Forest    0.8401     0.8489    0.8305    0.8392       0.8497   0.9123
        
        🏆 Mejor modelo (por F1-Score): Logistic Regression (0.8715)
    """
    df = compare_models(eval_results, sort_by=sort_by)
    
    print("\n" + "="*80)
    print("📊 COMPARACIÓN DE MODELOS")
    print("="*80 + "\n")
    
    # Imprimir DataFrame con formato
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    # Mostrar mejor modelo
    best_model = df.iloc[0]['Model']
    
    # Mapeo de sort_by a nombre de columna
    column_mapping = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1-Score',
        'specificity': 'Specificity',
        'roc_auc': 'ROC-AUC'
    }
    
    sort_metric_name = column_mapping[sort_by]
    best_value = df.iloc[0][sort_metric_name]
    
    print(f"\n🏆 Mejor modelo (por {sort_metric_name}): {best_model} ({best_value:.4f})")
    
    # Mostrar mejor por cada métrica si se solicita
    if show_best_by_metric:
        print("\n📈 Mejor modelo por métrica:")
        
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'ROC-AUC']:
            if metric == 'ROC-AUC':
                # Excluir modelos sin ROC-AUC (0.0)
                df_with_roc = df[df[metric] > 0.0]
                if len(df_with_roc) > 0:
                    best_idx = df_with_roc[metric].idxmax()
                    best_name = df_with_roc.loc[best_idx, 'Model']
                    best_val = df_with_roc.loc[best_idx, metric]
                    print(f"   • {metric:12s}: {best_name:<25s} ({best_val:.4f})")
            else:
                best_idx = df[metric].idxmax()
                best_name = df.loc[best_idx, 'Model']
                best_val = df.loc[best_idx, metric]
                print(f"   • {metric:12s}: {best_name:<25s} ({best_val:.4f})")
    
    print("\n" + "="*80)
    
    # Interpretación
    print("\n💡 Interpretación:")
    print(f"   • Dataset balanceado → F1-Score es la mejor métrica general")
    print(f"   • Minimizar falsos positivos → Priorizar Precision")
    print(f"   • Minimizar falsos negativos → Priorizar Recall")
    print(f"   • Evaluar poder discriminatorio → Usar ROC-AUC")
    print("="*80 + "\n")


if __name__ == "__main__":
    """
    Ejemplo de uso del módulo de evaluación.
    """
    print("\n" + "="*80)
    print("📊 MÓDULO DE EVALUACIÓN - Documentación")
    print("="*80)
    print("\nEste módulo proporciona funciones completas para evaluar modelos ML:")
    print("\n1. evaluate_model(model, X_test, y_test, model_name)")
    print("   - Calcula todas las métricas para un modelo individual")
    print("   - Retorna: accuracy, precision, recall, F1, ROC-AUC, confusion matrix")
    print("\n2. evaluate_all_models(models, X_test, y_test)")
    print("   - Evalúa múltiples modelos y compara resultados")
    print("   - Retorna diccionario con métricas de cada modelo")
    print("\n3. print_detailed_metrics(metrics)")
    print("   - Imprime métricas detalladas en formato legible")
    print("\n4. plot_confusion_matrix(cm, model_name)")
    print("   - Visualiza matriz de confusión con heatmap")
    print("\n5. plot_roc_curves(results)")
    print("   - Grafica curvas ROC de múltiples modelos")
    print("\n6. compare_models_table(results)")
    print("   - Tabla comparativa detallada de todos los modelos")
    print("\n" + "="*80)
    print("💡 Para ejemplos de uso, ver: test_evaluation.py")
    print("="*80 + "\n")
