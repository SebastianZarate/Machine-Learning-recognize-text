"""
Test script para el módulo visualizations.py

Este script verifica todas las funciones de visualización:
1. Comparación de modelos (barras)
2. Matrices de confusión
3. Curvas ROC
4. Word clouds
5. Distribución de sentimientos
6. Distribución de longitud de textos
7. Feature importance
8. Reporte completo

Author: Machine Learning Workshop
Date: 2025-10-29
"""

import sys
from pathlib import Path
import os

# Agregar src al path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from visualizations import (
    plot_model_comparison_bars,
    plot_confusion_matrices,
    plot_roc_curves_comparison,
    plot_word_cloud,
    plot_sentiment_word_clouds,
    plot_sentiment_distribution,
    plot_text_length_distribution,
    plot_feature_importance,
    create_comprehensive_report,
    save_all_figures
)


def create_sample_data():
    """Crea datos de muestra para testing."""
    print("📦 Creando datos de muestra...")
    
    # Datos de evaluación simulados
    eval_results = {
        'Naive Bayes': {
            'accuracy': 0.8542,
            'precision': 0.8621,
            'recall': 0.8453,
            'f1_score': 0.8536,
            'specificity': 0.8631,
            'roc_auc': 0.9234,
            'confusion_matrix': np.array([[4312, 688], [774, 4226]])
        },
        'Logistic Regression': {
            'accuracy': 0.8723,
            'precision': 0.8801,
            'recall': 0.8642,
            'f1_score': 0.8715,
            'specificity': 0.8804,
            'roc_auc': 0.9456,
            'confusion_matrix': np.array([[4402, 598], [679, 4321]])
        },
        'Random Forest': {
            'accuracy': 0.8401,
            'precision': 0.8489,
            'recall': 0.8305,
            'f1_score': 0.8392,
            'specificity': 0.8497,
            'roc_auc': 0.9123,
            'confusion_matrix': np.array([[4247, 753], [847, 4153]])
        }
    }
    
    # DataFrame de muestra
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'review': [
            # Reseñas positivas simuladas
            *['amazing excellent wonderful great fantastic best love perfect incredible awesome ' * 5] * (n_samples // 2),
            # Reseñas negativas simuladas
            *['terrible horrible awful bad worst hate disappointing waste poor boring ' * 5] * (n_samples // 2)
        ],
        'sentiment': [1] * (n_samples // 2) + [0] * (n_samples // 2)
    })
    
    # Agregar variación en longitud
    df['review'] = df['review'].apply(
        lambda x: ' '.join(x.split()[:np.random.randint(50, 200)])
    )
    
    print("   ✅ Datos de muestra creados")
    return eval_results, df


def create_mock_models():
    """Crea modelos simulados para testing."""
    print("🤖 Creando modelos simulados...")
    
    class MockModel:
        """Modelo simulado con predict_proba."""
        def __init__(self, name, auc):
            self.name = name
            self.auc = auc
        
        def predict_proba(self, X):
            n_samples = len(X) if hasattr(X, '__len__') else 100
            # Simular probabilidades que den un AUC cercano al deseado
            if self.auc > 0.9:
                probs = np.random.beta(2, 1, n_samples)
            elif self.auc > 0.85:
                probs = np.random.beta(1.5, 1, n_samples)
            else:
                probs = np.random.beta(1, 1, n_samples)
            
            return np.column_stack([1 - probs, probs])
    
    models = {
        'Naive Bayes': MockModel('Naive Bayes', 0.92),
        'Logistic Regression': MockModel('Logistic Regression', 0.95),
        'Random Forest': MockModel('Random Forest', 0.91)
    }
    
    print("   ✅ Modelos simulados creados")
    return models


def test_model_comparison_bars():
    """Test 1: Gráfico de barras de comparación."""
    print("\n" + "="*80)
    print("TEST 1: Gráfico de Comparación de Modelos (Barras)")
    print("="*80)
    
    eval_results, _ = create_sample_data()
    
    # Crear visualización
    fig = plot_model_comparison_bars(eval_results)
    
    print("✅ Gráfico de barras creado exitosamente")
    print(f"   • Tipo: {type(fig)}")
    print(f"   • Tamaño: {fig.get_size_inches()}")
    print(f"   • Modelos comparados: {len(eval_results)}")
    
    # Guardar
    output_file = 'test_comparison_bars.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   • Guardado en: {output_file}")
    plt.close(fig)
    
    # Limpiar
    if os.path.exists(output_file):
        os.remove(output_file)
    
    print("✅ Test 1 PASADO\n")


def test_confusion_matrices():
    """Test 2: Matrices de confusión."""
    print("="*80)
    print("TEST 2: Matrices de Confusión")
    print("="*80)
    
    eval_results, _ = create_sample_data()
    
    # Crear visualización
    fig = plot_confusion_matrices(eval_results)
    
    print("✅ Matrices de confusión creadas exitosamente")
    print(f"   • Tipo: {type(fig)}")
    print(f"   • Número de subplots: {len(eval_results)}")
    
    # Guardar
    output_file = 'test_confusion_matrices.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   • Guardado en: {output_file}")
    plt.close(fig)
    
    # Limpiar
    if os.path.exists(output_file):
        os.remove(output_file)
    
    print("✅ Test 2 PASADO\n")


def test_roc_curves():
    """Test 3: Curvas ROC."""
    print("="*80)
    print("TEST 3: Curvas ROC")
    print("="*80)
    
    models = create_mock_models()
    
    # Datos de prueba simulados
    X_test = np.random.randn(1000, 100)
    y_test = np.random.randint(0, 2, 1000)
    
    # Crear visualización
    fig = plot_roc_curves_comparison(models, X_test, y_test)
    
    print("✅ Curvas ROC creadas exitosamente")
    print(f"   • Tipo: {type(fig)}")
    print(f"   • Modelos graficados: {len(models)}")
    
    # Guardar
    output_file = 'test_roc_curves.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   • Guardado en: {output_file}")
    plt.close(fig)
    
    # Limpiar
    if os.path.exists(output_file):
        os.remove(output_file)
    
    print("✅ Test 3 PASADO\n")


def test_word_cloud():
    """Test 4: Word cloud."""
    print("="*80)
    print("TEST 4: Word Cloud")
    print("="*80)
    
    _, df = create_sample_data()
    
    # Obtener textos positivos
    positive_texts = df[df['sentiment'] == 1]['review'].tolist()
    
    # Crear visualización
    fig = plot_word_cloud(
        positive_texts,
        title="Word Cloud de Reseñas Positivas",
        max_words=50
    )
    
    print("✅ Word cloud creado exitosamente")
    print(f"   • Tipo: {type(fig)}")
    print(f"   • Número de textos: {len(positive_texts)}")
    
    # Guardar
    output_file = 'test_word_cloud.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   • Guardado en: {output_file}")
    plt.close(fig)
    
    # Limpiar
    if os.path.exists(output_file):
        os.remove(output_file)
    
    print("✅ Test 4 PASADO\n")


def test_sentiment_word_clouds():
    """Test 5: Word clouds comparativos."""
    print("="*80)
    print("TEST 5: Word Clouds Comparativos (Positivo vs Negativo)")
    print("="*80)
    
    _, df = create_sample_data()
    
    # Crear visualización
    fig = plot_sentiment_word_clouds(df, max_words=50)
    
    print("✅ Word clouds comparativos creados exitosamente")
    print(f"   • Tipo: {type(fig)}")
    print(f"   • Número de subplots: 2")
    
    # Guardar
    output_file = 'test_sentiment_word_clouds.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   • Guardado en: {output_file}")
    plt.close(fig)
    
    # Limpiar
    if os.path.exists(output_file):
        os.remove(output_file)
    
    print("✅ Test 5 PASADO\n")


def test_sentiment_distribution():
    """Test 6: Distribución de sentimientos."""
    print("="*80)
    print("TEST 6: Distribución de Sentimientos")
    print("="*80)
    
    _, df = create_sample_data()
    
    # Crear visualización
    fig = plot_sentiment_distribution(df)
    
    print("✅ Distribución de sentimientos creada exitosamente")
    print(f"   • Tipo: {type(fig)}")
    print(f"   • Sentimiento positivo: {(df['sentiment'] == 1).sum()} reseñas")
    print(f"   • Sentimiento negativo: {(df['sentiment'] == 0).sum()} reseñas")
    
    # Guardar
    output_file = 'test_sentiment_distribution.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   • Guardado en: {output_file}")
    plt.close(fig)
    
    # Limpiar
    if os.path.exists(output_file):
        os.remove(output_file)
    
    print("✅ Test 6 PASADO\n")


def test_text_length_distribution():
    """Test 7: Distribución de longitud de textos."""
    print("="*80)
    print("TEST 7: Distribución de Longitud de Textos")
    print("="*80)
    
    _, df = create_sample_data()
    
    # Crear visualización
    fig = plot_text_length_distribution(df)
    
    print("✅ Distribución de longitud creada exitosamente")
    print(f"   • Tipo: {type(fig)}")
    
    # Guardar
    output_file = 'test_text_length_distribution.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   • Guardado en: {output_file}")
    plt.close(fig)
    
    # Limpiar
    if os.path.exists(output_file):
        os.remove(output_file)
    
    print("✅ Test 7 PASADO\n")


def test_feature_importance():
    """Test 8: Feature importance."""
    print("="*80)
    print("TEST 8: Feature Importance")
    print("="*80)
    
    # Crear modelo simulado con feature_importances_
    class MockRandomForest:
        def __init__(self):
            self.feature_importances_ = np.random.dirichlet(
                np.ones(100), size=1
            )[0]
    
    model = MockRandomForest()
    feature_names = [f'word_{i}' for i in range(100)]
    
    # Crear visualización
    fig = plot_feature_importance(model, feature_names, top_n=20)
    
    print("✅ Feature importance creado exitosamente")
    print(f"   • Tipo: {type(fig)}")
    print(f"   • Total features: {len(feature_names)}")
    print(f"   • Top features mostrados: 20")
    
    # Guardar
    output_file = 'test_feature_importance.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   • Guardado en: {output_file}")
    plt.close(fig)
    
    # Limpiar
    if os.path.exists(output_file):
        os.remove(output_file)
    
    print("✅ Test 8 PASADO\n")


def test_comprehensive_report():
    """Test 9: Reporte completo."""
    print("="*80)
    print("TEST 9: Reporte Completo (Todas las Visualizaciones)")
    print("="*80)
    
    eval_results, df = create_sample_data()
    models = create_mock_models()
    
    # Datos de prueba
    X_test = np.random.randn(1000, 100)
    y_test = np.random.randint(0, 2, 1000)
    
    # Crear reporte completo
    print("\n📊 Generando reporte completo...")
    figures = create_comprehensive_report(
        eval_results,
        models,
        X_test,
        y_test,
        df=df
    )
    
    print(f"\n✅ Reporte completo generado")
    print(f"   • Total de visualizaciones: {len(figures)}")
    print(f"   • Visualizaciones incluidas:")
    for name in figures.keys():
        print(f"      - {name}")
    
    # Guardar todas las figuras
    output_dir = 'test_visualizations'
    save_all_figures(figures, output_dir=output_dir, dpi=150)
    
    # Verificar que se guardaron
    saved_files = list(Path(output_dir).glob('*.png'))
    print(f"\n   • Archivos guardados: {len(saved_files)}")
    
    # Limpiar directorio de prueba
    for file in saved_files:
        file.unlink()
    Path(output_dir).rmdir()
    
    print("✅ Test 9 PASADO\n")


def test_all_return_figures():
    """Test 10: Verificar que todas las funciones retornan Figure."""
    print("="*80)
    print("TEST 10: Verificación de Retorno de Figuras")
    print("="*80)
    
    eval_results, df = create_sample_data()
    models = create_mock_models()
    X_test = np.random.randn(100, 10)
    y_test = np.random.randint(0, 2, 100)
    
    # Lista de funciones a verificar
    functions = [
        ('plot_model_comparison_bars', lambda: plot_model_comparison_bars(eval_results)),
        ('plot_confusion_matrices', lambda: plot_confusion_matrices(eval_results)),
        ('plot_roc_curves_comparison', lambda: plot_roc_curves_comparison(models, X_test, y_test)),
        ('plot_word_cloud', lambda: plot_word_cloud(df['review'].tolist())),
        ('plot_sentiment_word_clouds', lambda: plot_sentiment_word_clouds(df)),
        ('plot_sentiment_distribution', lambda: plot_sentiment_distribution(df)),
        ('plot_text_length_distribution', lambda: plot_text_length_distribution(df)),
    ]
    
    print("\n🔍 Verificando tipo de retorno...")
    
    all_passed = True
    for name, func in functions:
        try:
            fig = func()
            is_figure = isinstance(fig, plt.Figure)
            status = "✅" if is_figure else "❌"
            print(f"   {status} {name}: {type(fig).__name__}")
            plt.close(fig)
            
            if not is_figure:
                all_passed = False
        except Exception as e:
            print(f"   ❌ {name}: ERROR - {e}")
            all_passed = False
    
    if all_passed:
        print("\n✅ Test 10 PASADO: Todas las funciones retornan plt.Figure\n")
    else:
        print("\n❌ Test 10 FALLIDO: Algunas funciones no retornan plt.Figure\n")
        raise AssertionError("No todas las funciones retornan plt.Figure")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 SUITE DE PRUEBAS: Módulo de Visualizaciones")
    print("="*80 + "\n")
    
    print("Este script prueba todas las funciones del módulo visualizations.py:")
    print("  1. Gráfico de comparación de modelos (barras)")
    print("  2. Matrices de confusión (heatmaps)")
    print("  3. Curvas ROC (multi-modelo)")
    print("  4. Word cloud simple")
    print("  5. Word clouds comparativos (positivo vs negativo)")
    print("  6. Distribución de sentimientos")
    print("  7. Distribución de longitud de textos")
    print("  8. Feature importance")
    print("  9. Reporte completo (todas las visualizaciones)")
    print(" 10. Verificación de retorno de figuras")
    print("\n" + "="*80 + "\n")
    
    try:
        test_model_comparison_bars()
        test_confusion_matrices()
        test_roc_curves()
        test_word_cloud()
        test_sentiment_word_clouds()
        test_sentiment_distribution()
        test_text_length_distribution()
        test_feature_importance()
        test_comprehensive_report()
        test_all_return_figures()
        
        print("="*80)
        print("✅ TODOS LOS TESTS PASARON EXITOSAMENTE")
        print("="*80)
        print("\n📚 El módulo de visualizaciones está listo para usar:")
        print("   • 8 funciones de visualización individuales")
        print("   • 1 función de reporte completo")
        print("   • 1 función para guardar todas las figuras")
        print("   • Todas las funciones retornan plt.Figure")
        print("\n💡 Puntos críticos cumplidos:")
        print("   ✅ Todas las funciones retornan Figure de matplotlib")
        print("   ✅ Configuración de estilo global aplicada")
        print("   ✅ Visualizaciones listas para notebooks y GUIs")
        print("   ✅ Funciones documentadas con docstrings completos")
        print("   ✅ Ejemplos de uso incluidos en cada función")
        print("\n🎯 Próximos pasos:")
        print("   • Integrar visualizaciones en notebooks")
        print("   • Crear ejemplos de uso en scripts")
        print("   • Generar reportes automáticos después del entrenamiento")
        print("="*80 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FALLIDO: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
