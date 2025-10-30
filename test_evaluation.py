"""
Script de prueba para el módulo de evaluación.

Demuestra cómo usar las funciones de evaluación para analizar
el rendimiento de los modelos entrenados.

Author: Machine Learning Workshop
Date: 2025-10-29
"""

import os
import sys

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import joblib
from evaluation import (
    evaluate_model,
    evaluate_all_models,
    print_detailed_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    compare_models_table
)


def main():
    """Pipeline completo de evaluación de modelos."""
    
    print("\n" + "="*80)
    print("📊 EVALUACIÓN COMPLETA DE MODELOS - CLASIFICACIÓN DE RESEÑAS")
    print("="*80 + "\n")
    
    model_path = "models/review_model.joblib"
    
    # ========== VERIFICAR MODELO EXISTE ==========
    if not os.path.exists(model_path):
        print(f"❌ ERROR: No se encontró el modelo en '{model_path}'")
        print(f"   Ejecuta primero: python test_train_supervised.py")
        return
    
    # ========== CARGAR MODELO Y TEST SET ==========
    print("📂 Cargando modelo entrenado...")
    model_data = joblib.load(model_path)
    
    trained_models = model_data['models']
    X_test = model_data['X_test']
    y_test = model_data['y_test']
    metadata = model_data.get('metadata', {})
    
    print(f"✓ Modelo cargado exitosamente")
    print(f"   • Modelos entrenados: {len(trained_models)}")
    print(f"   • Test samples: {X_test.shape[0]:,}")
    print(f"   • Features: {X_test.shape[1]:,}")
    
    if metadata:
        print(f"   • Entrenado: {metadata.get('trained_at', 'N/A')}")
        print(f"   • Train samples: {metadata.get('train_samples', 'N/A'):,}")
    
    # ========== EVALUACIÓN 1: TODOS LOS MODELOS ==========
    print("\n" + "🔵"*40)
    print("EVALUACIÓN 1: Comparación de todos los modelos")
    print("🔵"*40 + "\n")
    
    results = evaluate_all_models(trained_models, X_test, y_test, verbose=True)
    
    # ========== EVALUACIÓN 2: MÉTRICAS DETALLADAS POR MODELO ==========
    print("\n" + "🟢"*40)
    print("EVALUACIÓN 2: Métricas detalladas de cada modelo")
    print("🟢"*40)
    
    for model_name, metrics in results.items():
        print_detailed_metrics(metrics)
        input("\nPresiona Enter para continuar al siguiente modelo...")
    
    # ========== EVALUACIÓN 3: TABLA COMPARATIVA ==========
    print("\n" + "🟡"*40)
    print("EVALUACIÓN 3: Tabla comparativa completa")
    print("🟡"*40)
    
    compare_models_table(results)
    
    # ========== EVALUACIÓN 4: ANÁLISIS DE CONFUSION MATRICES ==========
    print("\n" + "🟣"*40)
    print("EVALUACIÓN 4: Análisis de confusion matrices")
    print("🟣"*40 + "\n")
    
    print("📊 Analizando matrices de confusión...")
    print("\nInterpretación:")
    print("   • True Negatives (TN):  NO reseña correctamente identificada")
    print("   • True Positives (TP):  Reseña correctamente identificada")
    print("   • False Positives (FP): NO reseña clasificada como reseña (Error Tipo I)")
    print("   • False Negatives (FN): Reseña clasificada como NO reseña (Error Tipo II)")
    print()
    
    for model_name, metrics in results.items():
        cm = metrics['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n{model_name}:")
        print(f"   TN: {tn:6,}  |  FP: {fp:6,}")
        print(f"   FN: {fn:6,}  |  TP: {tp:6,}")
        
        # Calcular error rates
        total = tn + fp + fn + tp
        error_rate = (fp + fn) / total
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        print(f"   Error Rate:       {error_rate:.2%}")
        print(f"   False Pos Rate:   {fp_rate:.2%}")
        print(f"   False Neg Rate:   {fn_rate:.2%}")
    
    # ========== EVALUACIÓN 5: ANÁLISIS ROC-AUC ==========
    print("\n" + "🔴"*40)
    print("EVALUACIÓN 5: Análisis ROC-AUC")
    print("🔴"*40 + "\n")
    
    models_with_roc = {k: v for k, v in results.items() if v['roc_auc']}
    
    if models_with_roc:
        print("📈 Modelos con ROC-AUC (predict_proba disponible):\n")
        
        # Ordenar por ROC-AUC
        sorted_roc = sorted(models_with_roc.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
        
        for model_name, metrics in sorted_roc:
            roc_auc = metrics['roc_auc']
            
            # Interpretación de ROC-AUC
            if roc_auc >= 0.95:
                interpretation = "Excelente 🏆"
            elif roc_auc >= 0.90:
                interpretation = "Muy Bueno ✅"
            elif roc_auc >= 0.80:
                interpretation = "Bueno ✓"
            elif roc_auc >= 0.70:
                interpretation = "Aceptable ⚠️"
            else:
                interpretation = "Necesita mejora ❌"
            
            print(f"   {model_name:25s}: {roc_auc:.4f} - {interpretation}")
        
        print("\n💡 Interpretación ROC-AUC:")
        print("   • 1.00:      Clasificador perfecto")
        print("   • 0.90-0.99: Excelente")
        print("   • 0.80-0.89: Muy bueno")
        print("   • 0.70-0.79: Bueno")
        print("   • 0.60-0.69: Regular")
        print("   • 0.50:      Clasificador aleatorio (sin poder predictivo)")
    else:
        print("⚠️  Ningún modelo tiene predict_proba disponible.")
        print("   ROC-AUC no puede ser calculado.")
    
    # ========== VISUALIZACIONES ==========
    visualize = input("\n\n¿Deseas generar visualizaciones? (s/n): ").strip().lower()
    
    if visualize in ['s', 'si', 'yes', 'y']:
        print("\n" + "🎨"*40)
        print("VISUALIZACIONES")
        print("🎨"*40 + "\n")
        
        try:
            # Crear directorio para guardar figuras
            os.makedirs("evaluation_results", exist_ok=True)
            
            # 1. Confusion matrices
            print("📊 Generando confusion matrices...")
            for model_name, metrics in results.items():
                filename = f"evaluation_results/confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
                plot_confusion_matrix(metrics['confusion_matrix'], model_name, filename)
            
            # 2. ROC curves
            if models_with_roc:
                print("\n📈 Generando curvas ROC...")
                plot_roc_curves(results, "evaluation_results/roc_curves.png")
            
            print("\n✓ Visualizaciones guardadas en: evaluation_results/")
            
        except Exception as e:
            print(f"\n⚠️  Error al generar visualizaciones: {e}")
            print("   Asegúrate de tener matplotlib y seaborn instalados:")
            print("   pip install matplotlib seaborn")
    
    # ========== RESUMEN FINAL ==========
    print("\n" + "="*80)
    print("✅ EVALUACIÓN COMPLETADA")
    print("="*80)
    
    # Encontrar mejor modelo
    best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
    
    print("\n🏆 MEJOR MODELO (por F1-Score):")
    print(f"   Modelo:      {best_model[0]}")
    print(f"   F1-Score:    {best_model[1]['f1_score']:.4f}")
    print(f"   Accuracy:    {best_model[1]['accuracy']:.4f}")
    print(f"   Precision:   {best_model[1]['precision']:.4f}")
    print(f"   Recall:      {best_model[1]['recall']:.4f}")
    if best_model[1]['roc_auc']:
        print(f"   ROC-AUC:     {best_model[1]['roc_auc']:.4f}")
    
    print("\n💡 RECOMENDACIONES:")
    
    # Análisis de rendimiento
    f1_score = best_model[1]['f1_score']
    if f1_score >= 0.90:
        print("   ✅ Excelente rendimiento! El modelo está listo para producción.")
    elif f1_score >= 0.85:
        print("   ✓ Muy buen rendimiento. Considerar ajustes finos opcionales.")
    elif f1_score >= 0.80:
        print("   ✓ Buen rendimiento. Considerar:")
        print("      - Ajustar hiperparámetros (grid search)")
        print("      - Aumentar tamaño del dataset")
    else:
        print("   ⚠️  Rendimiento por debajo del objetivo. Acciones sugeridas:")
        print("      - Revisar preprocesamiento de texto")
        print("      - Aumentar features TF-IDF (max_features)")
        print("      - Probar n-gramas más grandes (trigrams)")
        print("      - Balancear mejor el dataset")
        print("      - Ajustar hiperparámetros")
    
    # Análisis de balance precision-recall
    precision = best_model[1]['precision']
    recall = best_model[1]['recall']
    diff = abs(precision - recall)
    
    print(f"\n📊 Balance Precision-Recall:")
    print(f"   Diferencia: {diff:.4f}")
    
    if diff < 0.05:
        print("   ✅ Excelente balance entre precision y recall")
    elif diff < 0.10:
        print("   ✓ Buen balance")
    else:
        if precision > recall:
            print("   ⚠️  Precision > Recall:")
            print("      - El modelo es conservador (pocos falsos positivos)")
            print("      - Pero pierde algunas reseñas reales (falsos negativos)")
        else:
            print("   ⚠️  Recall > Precision:")
            print("      - El modelo detecta muchas reseñas (pocos falsos negativos)")
            print("      - Pero tiene falsos positivos (clasifica no-reseñas como reseñas)")
    
    print("\n" + "="*80)
    print("📁 Archivos generados:")
    print("   • Métricas detalladas: Impresas en consola")
    if visualize in ['s', 'si', 'yes', 'y']:
        print("   • Visualizaciones: evaluation_results/")
    print("\n🚀 Siguiente paso: Usar el mejor modelo para predicciones")
    print("   Ver: python example_predict.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido por el usuario.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
