"""
Script de prueba para la función de predicción con clasificadores supervisados.

Demuestra las diferentes estrategias de predicción:
- Majority voting (default)
- Unanimous voting
- Preferred model
- Weighted average

Author: Machine Learning Workshop
Date: 2025-10-29
"""

import os
import sys

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import predict_text
import json


def print_prediction_result(result, title="Predicción"):
    """Imprime resultado de predicción de forma legible."""
    print("\n" + "="*80)
    print(f"📊 {title}")
    print("="*80)
    
    # Decisión final
    decision_emoji = "✅" if result['is_review'] else "❌"
    print(f"\n{decision_emoji} Decisión Final: {result['final_decision'].upper()}")
    print(f"🎯 Confianza: {result['confidence']:.2%}")
    print(f"📝 Estrategia: {result['voting_strategy']}")
    
    # Texto analizado
    print(f"\n📄 Texto analizado:")
    print(f"   {result['text_preview']}")
    print(f"   Longitud: {result['text_length']} caracteres")
    
    # Votos
    print(f"\n🗳️  Votos:")
    print(f"   Positivos (es reseña):  {result['votes']['positive']}/{result['votes']['total']}")
    print(f"   Negativos (no reseña):  {result['votes']['negative']}/{result['votes']['total']}")
    
    # Predicciones individuales
    print(f"\n🤖 Predicciones por modelo:")
    for model_name, pred in result['predictions_by_model'].items():
        pred_emoji = "✅" if pred['prediction'] == 1 else "❌"
        pred_text = "ES reseña" if pred['prediction'] == 1 else "NO es reseña"
        
        if pred['probability'] is not None:
            print(f"   {pred_emoji} {model_name:20s}: {pred_text} (prob: {pred['probability']:.4f})")
        else:
            print(f"   {pred_emoji} {model_name:20s}: {pred_text}")
    
    # Análisis de keywords (si disponible)
    if 'keywords_analysis' in result:
        kw = result['keywords_analysis']
        print(f"\n🔍 Análisis de keywords (heurístico):")
        print(f"   Score de keywords: {kw['keyword_score']:.2%}")
        if kw['matched_keywords']:
            print(f"   Keywords encontradas: {', '.join(kw['matched_keywords'][:5])}")
        print(f"   Score evaluativo: {kw['eval_score']:.2%}")
        if kw['matched_evaluative']:
            print(f"   Palabras evaluativas: {', '.join(kw['matched_evaluative'][:5])}")
    
    print("="*80)


def test_examples():
    """Prueba la predicción con diferentes textos de ejemplo."""
    
    model_path = "models/review_model.joblib"
    
    # Verificar que existe el modelo
    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: No se encontró el modelo en '{model_path}'")
        print(f"   Ejecuta primero: python test_train_supervised.py")
        return
    
    print("\n" + "="*80)
    print("🎬 PRUEBA DE PREDICCIÓN CON CLASIFICADORES SUPERVISADOS")
    print("="*80)
    
    # ========== TEXTOS DE PRUEBA ==========
    test_texts = {
        "Reseña positiva clara": """
            Esta película es simplemente magistral. La dirección de Christopher Nolan
            es impecable, y la actuación de Leonardo DiCaprio me dejó sin palabras.
            La trama es compleja pero fascinante, te mantiene al borde del asiento.
            La fotografía es espectacular y la banda sonora de Hans Zimmer es perfecta.
            Definitivamente la recomiendo, es una obra maestra del cine. 10/10.
        """,
        
        "Reseña negativa clara": """
            Qué decepción de película. El guion es predecible y los personajes son
            planos. La actuación es mediocre y la dirección no tiene chispa. Me aburrí
            a los 20 minutos y solo terminé de verla por compromiso. No la recomiendo,
            hay mejores opciones en el cine. 3/10.
        """,
        
        "No es reseña - Noticia": """
            El presidente anunció hoy una nueva reforma económica que entrará en vigor
            el próximo mes. La medida busca reducir la inflación y mejorar las
            condiciones laborales. Los expertos opinan que tendrá un impacto positivo
            en la economía del país.
        """,
        
        "No es reseña - Receta": """
            Para preparar este delicioso pastel de chocolate, necesitarás 200g de
            harina, 150g de azúcar, 3 huevos y 100g de chocolate negro. Mezcla los
            ingredientes secos, agrega los huevos batidos y hornea a 180°C por 30
            minutos. ¡Disfruta!
        """,
        
        "Caso ambiguo - Descripción de película": """
            Inception es una película de ciencia ficción dirigida por Christopher Nolan.
            Fue estrenada en 2010 y está protagonizada por Leonardo DiCaprio. La película
            trata sobre el robo de secretos a través de los sueños. Ganó varios premios
            Oscar por efectos visuales y sonido.
        """,
        
        "Texto corto ambiguo": """
            Me gustó mucho. Muy buena.
        """
    }
    
    # ========== PRUEBA 1: MAJORITY VOTING (DEFAULT) ==========
    print("\n\n" + "🔵"*40)
    print("PRUEBA 1: MAJORITY VOTING (Default)")
    print("🔵"*40)
    print("Decisión: Voto mayoritario (≥2 de 3 modelos)")
    
    for title, text in test_texts.items():
        result = predict_text(text.strip(), model_path, voting_strategy="majority")
        print_prediction_result(result, f"Texto: {title}")
    
    # ========== PRUEBA 2: UNANIMOUS VOTING ==========
    print("\n\n" + "🟣"*40)
    print("PRUEBA 2: UNANIMOUS VOTING")
    print("🟣"*40)
    print("Decisión: Los 3 modelos deben estar de acuerdo (más conservador)")
    
    for title, text in list(test_texts.items())[:3]:  # Solo primeros 3 ejemplos
        result = predict_text(text.strip(), model_path, voting_strategy="unanimous")
        print_prediction_result(result, f"Texto: {title}")
    
    # ========== PRUEBA 3: PREFERRED MODEL ==========
    print("\n\n" + "🟢"*40)
    print("PRUEBA 3: PREFERRED MODEL - Logistic Regression")
    print("🟢"*40)
    print("Decisión: Solo usa Logistic Regression (mejor modelo en validación)")
    
    for title, text in list(test_texts.items())[:3]:  # Solo primeros 3 ejemplos
        result = predict_text(
            text.strip(), 
            model_path, 
            voting_strategy="preferred",
            preferred_model="Logistic Regression"
        )
        print_prediction_result(result, f"Texto: {title}")
    
    # ========== PRUEBA 4: WEIGHTED AVERAGE ==========
    print("\n\n" + "🟡"*40)
    print("PRUEBA 4: WEIGHTED AVERAGE")
    print("🟡"*40)
    print("Decisión: Promedio ponderado de probabilidades")
    
    for title, text in list(test_texts.items())[:3]:  # Solo primeros 3 ejemplos
        result = predict_text(text.strip(), model_path, voting_strategy="weighted_avg")
        print_prediction_result(result, f"Texto: {title}")
    
    # ========== RESUMEN FINAL ==========
    print("\n\n" + "="*80)
    print("✅ PRUEBAS COMPLETADAS")
    print("="*80)
    print("\n📊 Comparación de estrategias:")
    print("   • Majority Voting:  Recomendado para la mayoría de casos (balanceado)")
    print("   • Unanimous:        Más conservador (solo si todos coinciden)")
    print("   • Preferred Model:  Usa el mejor modelo según validación")
    print("   • Weighted Average: Considera confianza de cada modelo")
    print("\n💡 Recomendación: Usar 'majority' para producción")
    print("="*80 + "\n")


def interactive_mode():
    """Modo interactivo para probar predicciones con textos personalizados."""
    
    model_path = "models/review_model.joblib"
    
    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: No se encontró el modelo en '{model_path}'")
        print(f"   Ejecuta primero: python test_train_supervised.py")
        return
    
    print("\n" + "="*80)
    print("🎮 MODO INTERACTIVO - Predicción de Reseñas")
    print("="*80)
    print("\nEscribe un texto para clasificar (o 'salir' para terminar)")
    print("Estrategia por defecto: majority voting\n")
    
    while True:
        print("-"*80)
        text = input("\n📝 Ingresa tu texto: ").strip()
        
        if text.lower() in ['salir', 'exit', 'quit', 'q']:
            print("\n👋 ¡Hasta luego!")
            break
        
        if not text:
            print("⚠️  Texto vacío. Intenta de nuevo.")
            continue
        
        try:
            result = predict_text(text, model_path)
            print_prediction_result(result, "Resultado de tu texto")
        except Exception as e:
            print(f"\n❌ Error al procesar: {e}")


def main():
    """Menú principal."""
    
    print("\n" + "="*80)
    print("🎬 PRUEBA DE PREDICCIÓN - CLASIFICADORES SUPERVISADOS")
    print("="*80)
    print("\nOpciones:")
    print("  1. Ejecutar pruebas automáticas con textos de ejemplo")
    print("  2. Modo interactivo (ingresa tus propios textos)")
    print("  3. Ambos (pruebas + modo interactivo)")
    print("\n")
    
    choice = input("Selecciona una opción (1-3): ").strip()
    
    if choice == "1":
        test_examples()
    elif choice == "2":
        interactive_mode()
    elif choice == "3":
        test_examples()
        input("\n\nPresiona Enter para continuar al modo interactivo...")
        interactive_mode()
    else:
        print("❌ Opción inválida")


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
