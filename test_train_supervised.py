"""
Script de prueba para entrenar y evaluar clasificadores supervisados.

Este script demuestra el flujo completo:
1. Crear dataset balanceado (si no existe)
2. Entrenar clasificadores supervisados
3. Evaluar rendimiento de los modelos
4. Comparar resultados

Author: Machine Learning Workshop
Date: 2025-10-29
"""

import os
import sys

# Agregar src al path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preparation import create_balanced_dataset
from model import train_from_csv
from train_models import evaluate_all_models
import joblib


def main():
    """Pipeline completo de entrenamiento y evaluación."""
    
    print("\n" + "="*80)
    print("🎬 PIPELINE DE ENTRENAMIENTO - CLASIFICACIÓN DE RESEÑAS DE PELÍCULAS")
    print("="*80 + "\n")
    
    # Rutas de archivos
    imdb_dataset = "IMDB Dataset.csv"
    balanced_dataset = "balanced_dataset.csv"
    model_path = "models/review_model.joblib"
    
    # ========== PASO 1: CREAR DATASET BALANCEADO ==========
    if not os.path.exists(balanced_dataset):
        print("📦 PASO 1: Crear dataset balanceado")
        print("-" * 80)
        
        if not os.path.exists(imdb_dataset):
            print(f"❌ ERROR: No se encontró '{imdb_dataset}'")
            print(f"   Por favor, descarga el dataset IMDB desde Kaggle:")
            print(f"   https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
            return
        
        print(f"✓ Dataset IMDB encontrado: {imdb_dataset}")
        print(f"🔄 Creando dataset balanceado con 50k positivos + 50k negativos...\n")
        
        create_balanced_dataset(
            imdb_path=imdb_dataset,
            output_path=balanced_dataset,
            positive_count=50000,
            negative_count=50000
        )
        
        print(f"✓ Dataset balanceado creado: {balanced_dataset}\n")
    else:
        print("📦 PASO 1: Dataset balanceado ya existe")
        print("-" * 80)
        print(f"✓ Usando dataset existente: {balanced_dataset}\n")
    
    # ========== PASO 2: ENTRENAR CLASIFICADORES ==========
    print("\n🤖 PASO 2: Entrenar clasificadores supervisados")
    print("-" * 80)
    print("Esto puede tomar varios minutos dependiendo del tamaño del dataset...\n")
    
    model_path = train_from_csv(
        csv_path=balanced_dataset,
        model_path=model_path,
        test_size=0.2,
        random_state=42
    )
    
    # ========== PASO 3: EVALUAR MODELOS ==========
    print("\n📊 PASO 3: Evaluar modelos entrenados")
    print("-" * 80 + "\n")
    
    # Cargar modelo guardado
    print(f"📂 Cargando modelo desde: {model_path}")
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
    
    print("\n🔍 Evaluando todos los modelos en test set...\n")
    
    all_metrics = evaluate_all_models(
        models=trained_models,
        X_test=X_test,
        y_test=y_test,
        verbose=True
    )
    
    # ========== PASO 4: ANÁLISIS DE RESULTADOS ==========
    print("\n📈 PASO 4: Análisis de resultados")
    print("-" * 80)
    
    # Encontrar mejor modelo por cada métrica
    best_accuracy = max(all_metrics.items(), key=lambda x: x[1]['accuracy'])
    best_f1 = max(all_metrics.items(), key=lambda x: x[1]['f1_score'])
    best_precision = max(all_metrics.items(), key=lambda x: x[1]['precision'])
    best_recall = max(all_metrics.items(), key=lambda x: x[1]['recall'])
    
    print("\n🏆 Mejores modelos por métrica:")
    print(f"   • Accuracy:  {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.4f})")
    print(f"   • F1-Score:  {best_f1[0]} ({best_f1[1]['f1_score']:.4f})")
    print(f"   • Precision: {best_precision[0]} ({best_precision[1]['precision']:.4f})")
    print(f"   • Recall:    {best_recall[0]} ({best_recall[1]['recall']:.4f})")
    
    # Recomendaciones
    print("\n💡 Recomendaciones:")
    if best_f1[1]['f1_score'] >= 0.85:
        print("   ✅ Excelente rendimiento! F1-Score >= 0.85")
    elif best_f1[1]['f1_score'] >= 0.80:
        print("   ✓ Buen rendimiento. F1-Score >= 0.80")
    else:
        print("   ⚠️  Considerar:")
        print("      - Aumentar tamaño del dataset")
        print("      - Ajustar hiperparámetros (grid search)")
        print("      - Probar diferentes features (n-gramas más grandes)")
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*80)
    print("\n📁 Archivos generados:")
    print(f"   • Dataset balanceado: {balanced_dataset}")
    print(f"   • Modelo entrenado:   {model_path}")
    print("\n🚀 Siguiente paso: Usar los modelos para predicción")
    print("   Ver ejemplos en predict_text() en src/model.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido por el usuario.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
