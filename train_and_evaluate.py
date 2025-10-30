"""
Script completo: Entrenamiento + Evaluación integrados.

Este script ejecuta el pipeline completo de ML:
1. Entrena los modelos (si no existen)
2. Evalúa el rendimiento
3. Muestra análisis comparativo

Author: Machine Learning Workshop
Date: 2025-10-29
"""

import os
import sys

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import joblib
from model import train_from_csv
from evaluation import (
    evaluate_all_models,
    compare_models_table,
    print_detailed_metrics
)


def main():
    """Pipeline completo: Train + Evaluate."""
    
    print("\n" + "="*80)
    print("🎬 PIPELINE COMPLETO: ENTRENAMIENTO + EVALUACIÓN")
    print("="*80 + "\n")
    
    # Rutas
    balanced_dataset = "balanced_dataset.csv"
    model_path = "models/review_model.joblib"
    
    # ========== PASO 1: VERIFICAR DATASET ==========
    if not os.path.exists(balanced_dataset):
        print(f"❌ ERROR: No se encontró '{balanced_dataset}'")
        print(f"   Ejecuta primero:")
        print(f"   python test_train_supervised.py")
        return
    
    # ========== PASO 2: ENTRENAR O CARGAR MODELO ==========
    if not os.path.exists(model_path):
        print("🤖 PASO 1: Entrenando modelos...")
        print("-"*80 + "\n")
        
        model_path = train_from_csv(
            csv_path=balanced_dataset,
            model_path=model_path,
            test_size=0.2,
            random_state=42
        )
        
        print("\n✓ Entrenamiento completado\n")
    else:
        print("🤖 PASO 1: Modelo ya entrenado")
        print("-"*80)
        print(f"✓ Usando modelo existente: {model_path}\n")
    
    # ========== PASO 3: CARGAR MODELO Y TEST SET ==========
    print("\n📂 PASO 2: Cargando modelo y test set")
    print("-"*80)
    
    model_data = joblib.load(model_path)
    trained_models = model_data['models']
    X_test = model_data['X_test']
    y_test = model_data['y_test']
    metadata = model_data.get('metadata', {})
    
    print(f"✓ Modelo cargado")
    print(f"   • Modelos: {', '.join(trained_models.keys())}")
    print(f"   • Test samples: {X_test.shape[0]:,}")
    print(f"   • Features: {X_test.shape[1]:,}")
    
    if metadata:
        print(f"   • Entrenado: {metadata.get('trained_at', 'N/A')}")
    
    # ========== PASO 4: EVALUAR MODELOS ==========
    print("\n\n📊 PASO 3: Evaluando modelos")
    print("-"*80 + "\n")
    
    results = evaluate_all_models(trained_models, X_test, y_test, verbose=True)
    
    # ========== PASO 5: ANÁLISIS DETALLADO DEL MEJOR MODELO ==========
    print("\n\n🏆 PASO 4: Análisis detallado del mejor modelo")
    print("-"*80)
    
    # Encontrar mejor modelo por F1-score
    best_model_name, best_metrics = max(results.items(), key=lambda x: x[1]['f1_score'])
    
    print(f"\n🥇 Mejor modelo: {best_model_name}")
    print_detailed_metrics(best_metrics)
    
    # ========== PASO 6: COMPARACIÓN FINAL ==========
    print("\n\n📈 PASO 5: Comparación final de modelos")
    print("-"*80)
    
    compare_models_table(results)
    
    # ========== RECOMENDACIONES ==========
    print("\n" + "="*80)
    print("💡 RECOMENDACIONES")
    print("="*80)
    
    f1_score = best_metrics['f1_score']
    accuracy = best_metrics['accuracy']
    
    print(f"\n📊 Rendimiento del mejor modelo ({best_model_name}):")
    print(f"   • F1-Score: {f1_score:.4f} ({f1_score*100:.2f}%)")
    print(f"   • Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    if f1_score >= 0.90:
        print("\n✅ EXCELENTE! El modelo está listo para producción.")
        print("   Acciones sugeridas:")
        print("   1. Guardar el mejor modelo para deployment")
        print("   2. Implementar API REST para predicciones")
        print("   3. Monitorear rendimiento en producción")
    elif f1_score >= 0.85:
        print("\n✓ MUY BIEN! Rendimiento sólido para la mayoría de casos.")
        print("   Mejoras opcionales:")
        print("   1. Ajuste fino de hiperparámetros (GridSearchCV)")
        print("   2. Feature engineering adicional")
        print("   3. Ensemble de modelos (voting/stacking)")
    elif f1_score >= 0.80:
        print("\n✓ BUENO. Rendimiento aceptable pero mejorable.")
        print("   Acciones recomendadas:")
        print("   1. Aumentar tamaño del dataset de entrenamiento")
        print("   2. Ajustar hiperparámetros (GridSearchCV/RandomSearchCV)")
        print("   3. Probar n-gramas más grandes (trigrams)")
        print("   4. Considerar word embeddings (Word2Vec, GloVe)")
    else:
        print("\n⚠️  NECESITA MEJORA. El rendimiento está por debajo del objetivo.")
        print("   Acciones críticas:")
        print("   1. Revisar calidad del dataset (balance, ruido)")
        print("   2. Mejorar preprocesamiento de texto")
        print("   3. Aumentar features TF-IDF (max_features > 10000)")
        print("   4. Probar modelos más complejos (XGBoost, Neural Networks)")
        print("   5. Realizar análisis de errores detallado")
    
    # Balance precision-recall
    precision = best_metrics['precision']
    recall = best_metrics['recall']
    
    print(f"\n⚖️  Balance Precision-Recall:")
    print(f"   • Precision: {precision:.4f}")
    print(f"   • Recall:    {recall:.4f}")
    print(f"   • Diferencia: {abs(precision - recall):.4f}")
    
    if abs(precision - recall) < 0.05:
        print("   ✅ Excelente balance!")
    elif precision > recall + 0.10:
        print("   ⚠️  Modelo conservador:")
        print("      - Pocos falsos positivos (bueno)")
        print("      - Pero pierde reseñas reales (malo)")
        print("      → Considerar bajar threshold de decisión")
    elif recall > precision + 0.10:
        print("   ⚠️  Modelo agresivo:")
        print("      - Detecta muchas reseñas (bueno)")
        print("      - Pero tiene falsos positivos (malo)")
        print("      → Considerar subir threshold de decisión")
    
    # ROC-AUC analysis
    if best_metrics['roc_auc']:
        roc_auc = best_metrics['roc_auc']
        print(f"\n📈 ROC-AUC: {roc_auc:.4f}")
        
        if roc_auc >= 0.95:
            print("   🏆 Excelente poder de discriminación!")
        elif roc_auc >= 0.90:
            print("   ✅ Muy buen poder de discriminación")
        elif roc_auc >= 0.80:
            print("   ✓ Buen poder de discriminación")
        else:
            print("   ⚠️  Poder de discriminación mejorable")
    
    # ========== PRÓXIMOS PASOS ==========
    print("\n" + "="*80)
    print("🚀 PRÓXIMOS PASOS")
    print("="*80)
    print("\n1. Hacer predicciones:")
    print("   python example_predict.py")
    print("\n2. Evaluación detallada con visualizaciones:")
    print("   python test_evaluation.py")
    print("\n3. Probar modo interactivo:")
    print("   python test_predict_supervised.py")
    print("\n4. Integrar en aplicación (src/app.py)")
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
