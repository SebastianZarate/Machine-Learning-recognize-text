"""
Ejemplo simple: Entrenamiento con evaluación automática integrada.

Demuestra cómo train_from_csv() ahora incluye evaluación automática,
eliminando la necesidad de un paso manual separado.

Author: Machine Learning Workshop
Date: 2025-10-29
"""

import os
import sys

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import train_from_csv
import joblib


def main():
    """Ejemplo de entrenamiento con evaluación integrada."""
    
    print("\n" + "="*80)
    print("🎬 ENTRENAMIENTO CON EVALUACIÓN AUTOMÁTICA INTEGRADA")
    print("="*80 + "\n")
    
    balanced_dataset = "balanced_dataset.csv"
    model_path = "models/review_model.joblib"
    
    # Verificar dataset
    if not os.path.exists(balanced_dataset):
        print(f"❌ ERROR: No se encontró '{balanced_dataset}'")
        print(f"   Ejecuta primero:")
        print(f"   from data_preparation import create_balanced_dataset")
        print(f"   create_balanced_dataset('IMDB Dataset.csv', 'balanced_dataset.csv')")
        return
    
    print("🚀 PASO 1: Entrenar modelos")
    print("-"*80)
    print("La evaluación se ejecutará AUTOMÁTICAMENTE después del entrenamiento.\n")
    
    # ========== UN SOLO PASO: TRAIN + EVALUATE ==========
    # ANTES (2 pasos separados):
    #   1. train_from_csv(...)
    #   2. evaluate_all_models(...)  <- Paso manual
    #
    # AHORA (1 paso integrado):
    #   1. train_from_csv(...)  <- Incluye evaluación automática
    
    model_path = train_from_csv(
        csv_path=balanced_dataset,
        model_path=model_path,
        test_size=0.2,
        random_state=42
    )
    
    # ========== PASO 2: VERIFICAR RESULTADOS GUARDADOS ==========
    print("\n\n🔍 PASO 2: Verificar resultados guardados")
    print("-"*80 + "\n")
    
    print("📂 Cargando modelo entrenado...")
    model_data = joblib.load(model_path)
    
    # Verificar contenido
    print("✓ Contenido del modelo guardado:")
    print(f"   • vectorizer:          {'✓' if 'vectorizer' in model_data else '✗'}")
    print(f"   • models:              {'✓' if 'models' in model_data else '✗'} ({len(model_data.get('models', {}))} modelos)")
    print(f"   • evaluation_results:  {'✓' if 'evaluation_results' in model_data else '✗'} (NUEVO!)")
    print(f"   • best_model_name:     {'✓' if 'best_model_name' in model_data else '✗'} (NUEVO!)")
    print(f"   • X_test:              {'✓' if 'X_test' in model_data else '✗'}")
    print(f"   • y_test:              {'✓' if 'y_test' in model_data else '✗'}")
    print(f"   • metadata:            {'✓' if 'metadata' in model_data else '✗'}")
    
    # Mostrar mejor modelo
    if 'best_model_name' in model_data:
        best_model_name = model_data['best_model_name']
        eval_results = model_data['evaluation_results']
        best_metrics = eval_results[best_model_name]
        
        print(f"\n🏆 Mejor modelo identificado automáticamente:")
        print(f"   • Modelo:    {best_model_name}")
        print(f"   • F1-Score:  {best_metrics['f1_score']:.4f}")
        print(f"   • Accuracy:  {best_metrics['accuracy']:.4f}")
        print(f"   • Precision: {best_metrics['precision']:.4f}")
        print(f"   • Recall:    {best_metrics['recall']:.4f}")
        if best_metrics['roc_auc']:
            print(f"   • ROC-AUC:   {best_metrics['roc_auc']:.4f}")
    
    # ========== PASO 3: USAR MODELO PARA PREDICCIÓN ==========
    print("\n\n🎯 PASO 3: Usar modelo para predicción")
    print("-"*80 + "\n")
    
    from model import predict_text
    
    # Ejemplo de texto
    test_text = """
    Esta película es increíble! La actuación es magistral y la dirección 
    impecable. La trama te mantiene en suspenso de principio a fin. 
    Definitivamente la recomiendo. 10/10.
    """
    
    print("📝 Texto de prueba:")
    print(test_text.strip())
    print()
    
    # Predicción con estrategia por defecto (majority voting)
    result = predict_text(test_text, model_path)
    
    emoji = "✅" if result['is_review'] else "❌"
    print(f"{emoji} Resultado: {result['final_decision'].upper()}")
    print(f"🎯 Confianza: {result['confidence']:.2%}")
    print(f"🗳️  Votos: {result['votes']['positive']}/{result['votes']['total']} modelos")
    
    # Si quieres usar específicamente el mejor modelo identificado
    if 'best_model_name' in model_data:
        print(f"\n💡 Usando específicamente el mejor modelo ({best_model_name}):")
        
        result_best = predict_text(
            test_text,
            model_path,
            voting_strategy="preferred",
            preferred_model=best_model_name
        )
        
        emoji_best = "✅" if result_best['is_review'] else "❌"
        print(f"   {emoji_best} Resultado: {result_best['final_decision'].upper()}")
        print(f"   🎯 Confianza: {result_best['confidence']:.2%}")
    
    # ========== RESUMEN ==========
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETADO")
    print("="*80)
    print("\n📊 Ventajas de la evaluación integrada:")
    print("   1. ✓ Ahorra tiempo (no necesitas un paso separado)")
    print("   2. ✓ Resultados automáticos después del entrenamiento")
    print("   3. ✓ Identificación automática del mejor modelo")
    print("   4. ✓ Métricas guardadas para referencia futura")
    print("   5. ✓ No hay riesgo de olvidar evaluar")
    
    print("\n🎯 Punto crítico cumplido:")
    print("   • Evaluación SOLO en test set (nunca en train)")
    print("   • Métricas calculadas automáticamente")
    print("   • Sin inflación artificial del rendimiento")
    
    print("\n💡 Siguiente paso:")
    print("   Usar predict_text() para clasificar nuevos textos")
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
