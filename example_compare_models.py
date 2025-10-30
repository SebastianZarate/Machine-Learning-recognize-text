"""
Ejemplo de uso de compare_models() - Comparación de modelos con DataFrame

Este script demuestra cómo usar la nueva función compare_models() para:
1. Obtener un DataFrame con las métricas de todos los modelos
2. Ordenar por diferentes métricas según el objetivo
3. Exportar resultados a CSV para reportes
4. Realizar análisis programático de los resultados

Author: Machine Learning Workshop
Date: 2025-10-29
"""

import sys
from pathlib import Path

# Agregar src al path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

import pandas as pd
from evaluation import compare_models, print_comparison_table


def example_basic_comparison():
    """Ejemplo básico: Comparar modelos y mostrar resultados."""
    print("="*80)
    print("EJEMPLO 1: Comparación básica de modelos")
    print("="*80 + "\n")
    
    # Simular resultados de evaluate_all_models()
    eval_results = {
        'Naive Bayes': {
            'accuracy': 0.8542,
            'precision': 0.8621,
            'recall': 0.8453,
            'f1_score': 0.8536,
            'specificity': 0.8631,
            'roc_auc': 0.9234,
            'confusion_matrix': [[4312, 688], [774, 4226]]
        },
        'Logistic Regression': {
            'accuracy': 0.8723,
            'precision': 0.8801,
            'recall': 0.8642,
            'f1_score': 0.8715,
            'specificity': 0.8804,
            'roc_auc': 0.9456,
            'confusion_matrix': [[4402, 598], [679, 4321]]
        },
        'Random Forest': {
            'accuracy': 0.8401,
            'precision': 0.8489,
            'recall': 0.8305,
            'f1_score': 0.8392,
            'specificity': 0.8497,
            'roc_auc': 0.9123,
            'confusion_matrix': [[4247, 753], [847, 4153]]
        }
    }
    
    # Obtener DataFrame de comparación
    df = compare_models(eval_results)
    
    print("📊 Comparación de modelos (ordenado por F1-Score):\n")
    print(df.to_string(index=False))
    
    # Identificar mejor modelo
    best_model = df.iloc[0]['Model']
    best_f1 = df.iloc[0]['F1-Score']
    
    print(f"\n🏆 Mejor modelo: {best_model}")
    print(f"   • F1-Score: {best_f1:.4f}")
    print(f"   • Accuracy: {df.iloc[0]['Accuracy']:.4f}")
    print(f"   • ROC-AUC: {df.iloc[0]['ROC-AUC']:.4f}")


def example_sort_by_different_metrics():
    """Ejemplo: Ordenar por diferentes métricas según objetivo."""
    print("\n" + "="*80)
    print("EJEMPLO 2: Ordenar por diferentes métricas")
    print("="*80 + "\n")
    
    eval_results = {
        'Model A': {
            'accuracy': 0.85,
            'precision': 0.92,  # Alta precision
            'recall': 0.78,     # Baja recall
            'f1_score': 0.84,
            'specificity': 0.92,
            'roc_auc': 0.91
        },
        'Model B': {
            'accuracy': 0.87,
            'precision': 0.84,
            'recall': 0.90,     # Alta recall
            'f1_score': 0.87,
            'specificity': 0.84,
            'roc_auc': 0.95     # Mejor ROC-AUC
        },
        'Model C': {
            'accuracy': 0.88,   # Mejor accuracy
            'precision': 0.88,
            'recall': 0.88,
            'f1_score': 0.88,   # Mejor F1
            'specificity': 0.88,
            'roc_auc': 0.94
        }
    }
    
    print("📌 ESCENARIO 1: Minimizar falsos positivos (spam detection)")
    print("   → Ordenar por Precision\n")
    df_precision = compare_models(eval_results, sort_by='precision')
    print(df_precision[['Model', 'Precision', 'Recall', 'F1-Score']])
    print(f"\n   💡 Mejor modelo: {df_precision.iloc[0]['Model']} (Precision={df_precision.iloc[0]['Precision']:.4f})")
    
    print("\n📌 ESCENARIO 2: Minimizar falsos negativos (detección de fraude)")
    print("   → Ordenar por Recall\n")
    df_recall = compare_models(eval_results, sort_by='recall')
    print(df_recall[['Model', 'Precision', 'Recall', 'F1-Score']])
    print(f"\n   💡 Mejor modelo: {df_recall.iloc[0]['Model']} (Recall={df_recall.iloc[0]['Recall']:.4f})")
    
    print("\n📌 ESCENARIO 3: Dataset balanceado (movie reviews)")
    print("   → Ordenar por F1-Score\n")
    df_f1 = compare_models(eval_results, sort_by='f1_score')
    print(df_f1[['Model', 'Precision', 'Recall', 'F1-Score']])
    print(f"\n   💡 Mejor modelo: {df_f1.iloc[0]['Model']} (F1={df_f1.iloc[0]['F1-Score']:.4f})")
    
    print("\n📌 ESCENARIO 4: Evaluar poder discriminatorio general")
    print("   → Ordenar por ROC-AUC\n")
    df_roc = compare_models(eval_results, sort_by='roc_auc')
    print(df_roc[['Model', 'F1-Score', 'ROC-AUC']])
    print(f"\n   💡 Mejor modelo: {df_roc.iloc[0]['Model']} (ROC-AUC={df_roc.iloc[0]['ROC-AUC']:.4f})")


def example_export_to_csv():
    """Ejemplo: Exportar resultados a CSV."""
    print("\n" + "="*80)
    print("EJEMPLO 3: Exportar resultados a CSV")
    print("="*80 + "\n")
    
    eval_results = {
        'Naive Bayes': {
            'accuracy': 0.8542, 'precision': 0.8621, 'recall': 0.8453,
            'f1_score': 0.8536, 'specificity': 0.8631, 'roc_auc': 0.9234
        },
        'Logistic Regression': {
            'accuracy': 0.8723, 'precision': 0.8801, 'recall': 0.8642,
            'f1_score': 0.8715, 'specificity': 0.8804, 'roc_auc': 0.9456
        },
        'Random Forest': {
            'accuracy': 0.8401, 'precision': 0.8489, 'recall': 0.8305,
            'f1_score': 0.8392, 'specificity': 0.8497, 'roc_auc': 0.9123
        }
    }
    
    df = compare_models(eval_results)
    
    # Exportar a CSV
    output_file = 'model_comparison_report.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✅ Resultados exportados a: {output_file}")
    print("\n📄 Contenido del CSV:")
    print(df.to_string(index=False))
    
    # Limpiar
    from pathlib import Path
    Path(output_file).unlink()
    
    print(f"\n💡 Casos de uso para CSV:")
    print("   • Incluir en reportes de Jupyter Notebooks")
    print("   • Compartir con stakeholders no técnicos")
    print("   • Documentar experimentos de ML")
    print("   • Generar gráficos en Excel/Google Sheets")


def example_programmatic_analysis():
    """Ejemplo: Análisis programático de resultados."""
    print("\n" + "="*80)
    print("EJEMPLO 4: Análisis programático")
    print("="*80 + "\n")
    
    eval_results = {
        'Naive Bayes': {
            'accuracy': 0.8542, 'precision': 0.8621, 'recall': 0.8453,
            'f1_score': 0.8536, 'specificity': 0.8631, 'roc_auc': 0.9234
        },
        'Logistic Regression': {
            'accuracy': 0.8723, 'precision': 0.8801, 'recall': 0.8642,
            'f1_score': 0.8715, 'specificity': 0.8804, 'roc_auc': 0.9456
        },
        'Random Forest': {
            'accuracy': 0.8401, 'precision': 0.8489, 'recall': 0.8305,
            'f1_score': 0.8392, 'specificity': 0.8497, 'roc_auc': 0.9123
        }
    }
    
    df = compare_models(eval_results)
    
    print("🔍 Análisis estadístico:")
    print(f"   • F1-Score promedio: {df['F1-Score'].mean():.4f}")
    print(f"   • F1-Score máximo: {df['F1-Score'].max():.4f}")
    print(f"   • F1-Score mínimo: {df['F1-Score'].min():.4f}")
    print(f"   • Desviación estándar: {df['F1-Score'].std():.4f}")
    print(f"   • Rango: {df['F1-Score'].max() - df['F1-Score'].min():.4f}")
    
    print("\n🎯 Filtrar modelos con criterio:")
    # Modelos con F1 > 0.85
    good_models = df[df['F1-Score'] > 0.85]
    print(f"\n   Modelos con F1-Score > 0.85:")
    print(good_models[['Model', 'F1-Score', 'Accuracy']].to_string(index=False))
    
    # Modelos con ROC-AUC > 0.92
    high_roc = df[df['ROC-AUC'] > 0.92]
    print(f"\n   Modelos con ROC-AUC > 0.92:")
    print(high_roc[['Model', 'ROC-AUC', 'F1-Score']].to_string(index=False))
    
    print("\n🏆 Top N modelos:")
    top2 = df.head(2)
    print("\n   Top 2 modelos por F1-Score:")
    print(top2[['Model', 'F1-Score', 'Accuracy', 'ROC-AUC']].to_string(index=False))
    
    print("\n📊 Diferencia de rendimiento:")
    best_f1 = df.iloc[0]['F1-Score']
    worst_f1 = df.iloc[-1]['F1-Score']
    diff = best_f1 - worst_f1
    improvement_pct = (diff / worst_f1) * 100
    print(f"   • Mejor modelo vs peor: +{diff:.4f} ({improvement_pct:.2f}% mejora)")
    
    print("\n💡 Recomendación automática:")
    if diff < 0.01:
        print("   ⚠️ Modelos tienen rendimiento muy similar")
        print("   → Priorizar modelo más simple (Naive Bayes) por eficiencia")
    elif diff < 0.03:
        print("   ⚖️ Diferencia moderada de rendimiento")
        print("   → Considerar trade-off entre accuracy e interpretabilidad")
    else:
        print(f"   ✅ {df.iloc[0]['Model']} supera significativamente a otros")
        print("   → Recomendado usar este modelo en producción")


def example_print_function():
    """Ejemplo: Usar función print_comparison_table()."""
    print("\n" + "="*80)
    print("EJEMPLO 5: Función print_comparison_table()")
    print("="*80 + "\n")
    
    eval_results = {
        'Naive Bayes': {
            'accuracy': 0.8542, 'precision': 0.8621, 'recall': 0.8453,
            'f1_score': 0.8536, 'specificity': 0.8631, 'roc_auc': 0.9234
        },
        'Logistic Regression': {
            'accuracy': 0.8723, 'precision': 0.8801, 'recall': 0.8642,
            'f1_score': 0.8715, 'specificity': 0.8804, 'roc_auc': 0.9456
        },
        'Random Forest': {
            'accuracy': 0.8401, 'precision': 0.8489, 'recall': 0.8305,
            'f1_score': 0.8392, 'specificity': 0.8497, 'roc_auc': None
        }
    }
    
    print("💡 Para visualización rápida, usa print_comparison_table():\n")
    print_comparison_table(eval_results, show_best_by_metric=True)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎓 EJEMPLOS DE USO: compare_models()")
    print("="*80 + "\n")
    
    print("Esta suite demuestra diferentes formas de usar compare_models():")
    print("1. Comparación básica con DataFrame")
    print("2. Ordenar por diferentes métricas según objetivo")
    print("3. Exportar resultados a CSV")
    print("4. Análisis programático de resultados")
    print("5. Función print_comparison_table() para visualización rápida")
    print("\n" + "="*80 + "\n")
    
    try:
        example_basic_comparison()
        example_sort_by_different_metrics()
        example_export_to_csv()
        example_programmatic_analysis()
        example_print_function()
        
        print("\n" + "="*80)
        print("✅ TODOS LOS EJEMPLOS EJECUTADOS EXITOSAMENTE")
        print("="*80)
        print("\n📚 Resumen de funcionalidades:")
        print("   ✅ compare_models() retorna DataFrame de pandas")
        print("   ✅ Ordenamiento flexible por cualquier métrica")
        print("   ✅ Exportación a CSV para reportes")
        print("   ✅ Análisis programático con pandas")
        print("   ✅ print_comparison_table() para visualización rápida")
        print("\n💡 Casos de uso recomendados:")
        print("   • Comparar múltiples modelos después del entrenamiento")
        print("   • Seleccionar mejor modelo según objetivo del negocio")
        print("   • Generar reportes automáticos de rendimiento")
        print("   • Documentar experimentos de ML")
        print("   • Análisis exploratorio de modelos")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
