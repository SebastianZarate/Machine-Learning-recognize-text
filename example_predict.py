"""
Ejemplo rápido de predicción con clasificadores supervisados.

Uso básico de predict_text() con diferentes estrategias.

Author: Machine Learning Workshop
Date: 2025-10-29
"""

import os
import sys

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import predict_text


def main():
    """Ejemplo rápido de uso."""
    
    model_path = "models/review_model.joblib"
    
    # Verificar modelo
    if not os.path.exists(model_path):
        print(f"\n❌ Modelo no encontrado. Ejecuta primero:")
        print(f"   python test_train_supervised.py\n")
        return
    
    print("\n" + "="*70)
    print("🎬 EJEMPLO RÁPIDO - Predicción de Reseñas")
    print("="*70 + "\n")
    
    # Texto de ejemplo
    texto_reseña = """
    Esta película es increíble! La actuación de los protagonistas es 
    magistral y la dirección impecable. La trama te mantiene en suspenso
    de principio a fin. Definitivamente la recomiendo. 10/10.
    """
    
    texto_no_reseña = """
    El ministro de economía anunció hoy nuevas medidas fiscales que
    buscan reducir la inflación. Los expertos consideran que el impacto
    será positivo para el crecimiento económico del país.
    """
    
    # ========== PREDICCIÓN 1: RESEÑA POSITIVA ==========
    print("📝 Texto 1: Reseña de película")
    print("-"*70)
    print(texto_reseña.strip())
    print()
    
    result1 = predict_text(texto_reseña, model_path)
    
    emoji1 = "✅" if result1['is_review'] else "❌"
    print(f"{emoji1} Resultado: {result1['final_decision'].upper()}")
    print(f"🎯 Confianza: {result1['confidence']:.2%}")
    print(f"🗳️  Votos: {result1['votes']['positive']}/{result1['votes']['total']} modelos votaron 'es reseña'")
    
    print("\n🤖 Predicciones individuales:")
    for name, pred in result1['predictions_by_model'].items():
        pred_text = "ES reseña" if pred['prediction'] == 1 else "NO es reseña"
        if pred['probability']:
            print(f"   • {name:20s}: {pred_text} (prob: {pred['probability']:.3f})")
        else:
            print(f"   • {name:20s}: {pred_text}")
    
    # ========== PREDICCIÓN 2: NO RESEÑA ==========
    print("\n" + "="*70)
    print("📝 Texto 2: Noticia de economía")
    print("-"*70)
    print(texto_no_reseña.strip())
    print()
    
    result2 = predict_text(texto_no_reseña, model_path)
    
    emoji2 = "✅" if result2['is_review'] else "❌"
    print(f"{emoji2} Resultado: {result2['final_decision'].upper()}")
    print(f"🎯 Confianza: {result2['confidence']:.2%}")
    print(f"🗳️  Votos: {result2['votes']['positive']}/{result2['votes']['total']} modelos votaron 'es reseña'")
    
    print("\n🤖 Predicciones individuales:")
    for name, pred in result2['predictions_by_model'].items():
        pred_text = "ES reseña" if pred['prediction'] == 1 else "NO es reseña"
        if pred['probability']:
            print(f"   • {name:20s}: {pred_text} (prob: {pred['probability']:.3f})")
        else:
            print(f"   • {name:20s}: {pred_text}")
    
    # ========== COMPARACIÓN DE ESTRATEGIAS ==========
    print("\n" + "="*70)
    print("🔍 COMPARACIÓN DE ESTRATEGIAS (con Texto 1)")
    print("="*70 + "\n")
    
    strategies = {
        "majority": "Voto mayoritario (≥2 de 3)",
        "unanimous": "Unanimidad (3 de 3)",
        "preferred": "Solo Logistic Regression",
        "weighted_avg": "Promedio ponderado"
    }
    
    for strategy, description in strategies.items():
        kwargs = {'voting_strategy': strategy, 'return_details': False}
        if strategy == "preferred":
            kwargs['preferred_model'] = "Logistic Regression"
        
        result = predict_text(texto_reseña, model_path, **kwargs)
        decision = "✅ ES reseña" if result['is_review'] else "❌ NO es reseña"
        
        print(f"{strategy:15s} ({description})")
        print(f"  → {decision} | Confianza: {result['confidence']:.2%}")
        print()
    
    # ========== RESUMEN ==========
    print("="*70)
    print("✅ EJEMPLO COMPLETADO")
    print("="*70)
    print("\n💡 Uso recomendado en producción:")
    print("   result = predict_text(texto, voting_strategy='majority')")
    print("\n📚 Para más ejemplos, ejecuta:")
    print("   python test_predict_supervised.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
