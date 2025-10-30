"""
Test script para la función compare_models() del módulo evaluation.

Este script verifica que la función de comparación de modelos:
1. Retorna un DataFrame de pandas correctamente formateado
2. Ordena por la métrica especificada
3. Maneja valores None en ROC-AUC
4. Proporciona una tabla legible para comparación

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


def test_compare_models_basic():
    """Prueba básica de compare_models con resultados simulados."""
    print("="*80)
    print("TEST 1: Comparación básica de modelos")
    print("="*80)
    
    # Simular resultados de evaluación
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
    
    # Llamar función
    df = compare_models(eval_results)
    
    print("\n📊 DataFrame retornado por compare_models():")
    print(df)
    
    print("\n✅ Verificaciones:")
    print(f"   • Tipo retornado: {type(df)}")
    print(f"   • Número de filas: {len(df)}")
    print(f"   • Columnas: {list(df.columns)}")
    print(f"   • Ordenado por: F1-Score (descendente)")
    print(f"   • Mejor modelo: {df.iloc[0]['Model']} (F1={df.iloc[0]['F1-Score']:.4f})")
    
    # Verificar ordenamiento
    assert df.iloc[0]['Model'] == 'Logistic Regression', "Error: Mejor modelo no es el esperado"
    assert df.iloc[0]['F1-Score'] > df.iloc[1]['F1-Score'], "Error: No está ordenado correctamente"
    
    print("\n✅ Test 1 PASADO: DataFrame generado correctamente\n")


def test_compare_models_sorting():
    """Prueba ordenamiento por diferentes métricas."""
    print("="*80)
    print("TEST 2: Ordenamiento por diferentes métricas")
    print("="*80)
    
    eval_results = {
        'Model A': {
            'accuracy': 0.85,
            'precision': 0.88,
            'recall': 0.82,
            'f1_score': 0.85,
            'specificity': 0.88,
            'roc_auc': 0.92
        },
        'Model B': {
            'accuracy': 0.87,
            'precision': 0.84,
            'recall': 0.90,
            'f1_score': 0.87,
            'specificity': 0.84,
            'roc_auc': 0.95
        },
        'Model C': {
            'accuracy': 0.82,
            'precision': 0.90,
            'recall': 0.74,
            'f1_score': 0.81,
            'specificity': 0.90,
            'roc_auc': 0.89
        }
    }
    
    # Ordenar por diferentes métricas
    metrics_to_test = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    for metric in metrics_to_test:
        df = compare_models(eval_results, sort_by=metric)
        print(f"\n📊 Ordenado por {metric}:")
        print(df[['Model', metric.replace('_', '-').title() if metric != 'f1_score' and metric != 'roc_auc' else ('F1-Score' if metric == 'f1_score' else 'ROC-AUC')]])
        
        # Verificar que está ordenado descendente
        col_name = metric.replace('_', '-').title() if metric not in ['f1_score', 'roc_auc'] else ('F1-Score' if metric == 'f1_score' else 'ROC-AUC')
        values = df[col_name].tolist()
        assert values == sorted(values, reverse=True), f"Error: No ordenado correctamente por {metric}"
        print(f"   ✅ Ordenamiento correcto")
    
    print("\n✅ Test 2 PASADO: Ordenamiento por todas las métricas funciona\n")


def test_compare_models_none_roc():
    """Prueba manejo de ROC-AUC None."""
    print("="*80)
    print("TEST 3: Manejo de ROC-AUC = None")
    print("="*80)
    
    eval_results = {
        'Model with ROC': {
            'accuracy': 0.85,
            'precision': 0.86,
            'recall': 0.84,
            'f1_score': 0.85,
            'specificity': 0.86,
            'roc_auc': 0.92
        },
        'Model without ROC': {
            'accuracy': 0.87,
            'precision': 0.88,
            'recall': 0.86,
            'f1_score': 0.87,
            'specificity': 0.88,
            'roc_auc': None  # Modelo sin predict_proba
        }
    }
    
    df = compare_models(eval_results)
    
    print("\n📊 DataFrame con ROC-AUC None:")
    print(df)
    
    # Verificar que None se convierte a 0.0
    roc_values = df['ROC-AUC'].tolist()
    print(f"\n✅ Valores ROC-AUC: {roc_values}")
    assert 0.0 in roc_values, "Error: None no se convirtió a 0.0"
    assert None not in roc_values, "Error: Queda None en el DataFrame"
    
    print("\n✅ Test 3 PASADO: ROC-AUC None manejado correctamente\n")


def test_print_comparison_table():
    """Prueba función print_comparison_table."""
    print("="*80)
    print("TEST 4: Función print_comparison_table()")
    print("="*80)
    
    eval_results = {
        'Naive Bayes': {
            'accuracy': 0.8542,
            'precision': 0.8621,
            'recall': 0.8453,
            'f1_score': 0.8536,
            'specificity': 0.8631,
            'roc_auc': 0.9234
        },
        'Logistic Regression': {
            'accuracy': 0.8723,
            'precision': 0.8801,
            'recall': 0.8642,
            'f1_score': 0.8715,
            'specificity': 0.8804,
            'roc_auc': 0.9456
        },
        'Random Forest': {
            'accuracy': 0.8401,
            'precision': 0.8489,
            'recall': 0.8305,
            'f1_score': 0.8392,
            'specificity': 0.8497,
            'roc_auc': None  # Sin predict_proba
        }
    }
    
    # Llamar función de impresión
    print_comparison_table(eval_results, show_best_by_metric=True)
    
    print("\n✅ Test 4 PASADO: Tabla impresa correctamente\n")


def test_dataframe_export():
    """Prueba exportación del DataFrame a CSV."""
    print("="*80)
    print("TEST 5: Exportación a CSV")
    print("="*80)
    
    eval_results = {
        'Model A': {
            'accuracy': 0.85, 'precision': 0.86, 'recall': 0.84,
            'f1_score': 0.85, 'specificity': 0.86, 'roc_auc': 0.92
        },
        'Model B': {
            'accuracy': 0.87, 'precision': 0.88, 'recall': 0.86,
            'f1_score': 0.87, 'specificity': 0.88, 'roc_auc': 0.95
        }
    }
    
    df = compare_models(eval_results)
    
    # Exportar a CSV
    output_file = 'test_model_comparison.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ DataFrame exportado a: {output_file}")
    
    # Verificar que se creó
    from pathlib import Path
    assert Path(output_file).exists(), "Error: CSV no creado"
    
    # Leer y verificar
    df_read = pd.read_csv(output_file)
    print(f"\n📄 CSV leído correctamente:")
    print(df_read)
    
    assert len(df_read) == len(df), "Error: Número de filas diferente"
    
    # Limpiar archivo de prueba
    Path(output_file).unlink()
    
    print("\n✅ Test 5 PASADO: Exportación a CSV funciona\n")


def test_programmatic_usage():
    """Prueba uso programático del DataFrame."""
    print("="*80)
    print("TEST 6: Uso programático del DataFrame")
    print("="*80)
    
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
    
    # Análisis programático
    print("\n🔍 Análisis programático:")
    
    # Encontrar mejor modelo
    best_model = df.iloc[0]['Model']
    best_f1 = df.iloc[0]['F1-Score']
    print(f"   • Mejor modelo: {best_model} (F1={best_f1:.4f})")
    
    # Estadísticas
    print(f"\n📊 Estadísticas:")
    print(f"   • F1-Score promedio: {df['F1-Score'].mean():.4f}")
    print(f"   • F1-Score máximo: {df['F1-Score'].max():.4f}")
    print(f"   • F1-Score mínimo: {df['F1-Score'].min():.4f}")
    print(f"   • Desviación estándar: {df['F1-Score'].std():.4f}")
    
    # Filtrar modelos con F1 > 0.85
    good_models = df[df['F1-Score'] > 0.85]
    print(f"\n✨ Modelos con F1-Score > 0.85:")
    print(good_models[['Model', 'F1-Score']])
    
    # Top 2 modelos
    top2 = df.head(2)
    print(f"\n🏆 Top 2 modelos:")
    print(top2[['Model', 'F1-Score', 'ROC-AUC']])
    
    print("\n✅ Test 6 PASADO: Uso programático exitoso\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 SUITE DE PRUEBAS: compare_models()")
    print("="*80 + "\n")
    
    try:
        test_compare_models_basic()
        test_compare_models_sorting()
        test_compare_models_none_roc()
        test_print_comparison_table()
        test_dataframe_export()
        test_programmatic_usage()
        
        print("="*80)
        print("✅ TODOS LOS TESTS PASARON EXITOSAMENTE")
        print("="*80)
        print("\n📚 La función compare_models() está lista para usar:")
        print("   1. Retorna DataFrame de pandas para análisis programático")
        print("   2. Permite ordenar por cualquier métrica")
        print("   3. Maneja ROC-AUC None correctamente")
        print("   4. Se puede exportar a CSV fácilmente")
        print("   5. Incluye función print_comparison_table() para visualización rápida")
        print("\n💡 Próximos pasos:")
        print("   • Integrar en train_from_csv() para comparación automática")
        print("   • Usar en análisis exploratorio de modelos")
        print("   • Exportar resultados para reportes")
        print("="*80 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FALLIDO: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
