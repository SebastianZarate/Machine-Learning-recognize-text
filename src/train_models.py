"""Módulo de entrenamiento de modelos."""
import time
import os
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Importar configuración
from config import (
    TFIDF_CONFIG, MODEL_CONFIG, TRAIN_TEST_SPLIT, 
    MODEL_FILES, MODELS_DIR
)

# Configurar logging
logger = logging.getLogger(__name__)

def split_data(df, test_size=0.2, random_state=42):
    print('\n📊 Dividiendo datos...')
    X, y = df['review_clean'], df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    print(f'✓ OK Division: {len(X_train)} train, {len(X_test)} test')
    return X_train, X_test, y_train, y_test

def create_vectorizers():
    print('\n✨ Creando vectorizadores...')
    logger.info("Creando vectorizadores con configuración estandarizada")
    cv = CountVectorizer(max_features=5000, ngram_range=(1,2))
    tfidf = TfidfVectorizer(**TFIDF_CONFIG)
    logger.debug(f"TF-IDF config: {TFIDF_CONFIG}")
    return cv, tfidf

def vectorize_data(X_train, X_test, vectorizer):
    print(f'🔤 Vectorizando con {type(vectorizer).__name__}...')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f'✓ OK Shape: {X_train_vec.shape}')
    return X_train_vec, X_test_vec, vectorizer

def train_naive_bayes(X_train, y_train):
    print('\n🧠 Entrenando Naive Bayes...')
    logger.info("Iniciando entrenamiento de Naive Bayes")
    model = MultinomialNB(**MODEL_CONFIG['naive_bayes'])
    model.fit(X_train, y_train)
    print('✓ OK Naive Bayes entrenado')
    logger.info("Naive Bayes entrenado exitosamente")
    return model

def train_logistic_regression(X_train, y_train):
    print('\n🔄 Entrenando Regresion Logistica...')
    logger.info("Iniciando entrenamiento de Logistic Regression")
    model = LogisticRegression(**MODEL_CONFIG['logistic_regression'])
    model.fit(X_train, y_train)
    print('✓ OK Regresion Logistica entrenada')
    logger.info("Logistic Regression entrenada exitosamente")
    return model

def train_random_forest(X_train, y_train):
    print('\n🌲 Entrenando Random Forest...')
    logger.info("Iniciando entrenamiento de Random Forest")
    model = RandomForestClassifier(**MODEL_CONFIG['random_forest'])
    model.fit(X_train, y_train)
    print('✓ OK Random Forest entrenado')
    logger.info("Random Forest entrenado exitosamente")
    return model

def train_all_models(X_train, y_train):
    print('\n' + '='*50)
    print('ENTRENANDO TODOS LOS MODELOS')
    print('='*50)
    models = {}
    models['naive_bayes'] = train_naive_bayes(X_train, y_train)
    models['logistic_regression'] = train_logistic_regression(X_train, y_train)
    models['random_forest'] = train_random_forest(X_train, y_train)
    print('\n✓ OK TODOS LOS MODELOS ENTRENADOS')
    return models

def save_models(models_dict, vectorizer, output_dir='models/'):
    print(f'💾 Guardando modelos en {output_dir}...')
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models_dict.items():
        path = os.path.join(output_dir, f'{name}.joblib')
        joblib.dump(model, path)
        print(f'✓ OK {name} guardado')
    
    # Guardar vectorizador con nombre estándar
    vec_path = os.path.join(output_dir, 'vectorizer.joblib')
    joblib.dump(vectorizer, vec_path)
    print(f'✓ OK Vectorizador guardado como: {vec_path}')


def train_all_models_from_file(csv_path: str, output_dir: str = 'models/') -> Dict[str, Any]:
    """
    Función wrapper completa para entrenar todos los modelos desde un archivo CSV.
    
    Esta función es usada por la GUI (app.py) para entrenar los modelos.
    
    Args:
        csv_path: Ruta al archivo CSV con columnas 'review' y 'sentiment'
        output_dir: Directorio donde guardar los modelos entrenados
    
    Returns:
        Dict con los resultados del entrenamiento
    """
    print('\n' + '='*70)
    print('INICIANDO PIPELINE COMPLETO DE ENTRENAMIENTO')
    print('='*70)
    
    # 1. Cargar datos
    print(f'\n📂 Cargando datos desde: {csv_path}')
    df = pd.read_csv(csv_path)
    print(f'✓ Datos cargados: {len(df)} muestras')
    
    # 2. Verificar columnas
    if 'review' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError(
            "El CSV debe tener columnas 'review' y 'sentiment'.\n"
            f"Columnas encontradas: {list(df.columns)}"
        )
    
    # 3. Preprocesar
    print('\n🧹 Preprocesando textos...')
    from preprocessing import preprocess_pipeline
    df['review_clean'] = df['review'].apply(preprocess_pipeline)
    print('✓ Preprocesamiento completado')
    
    # 4. Codificar etiquetas
    print('\n🏷️ Codificando etiquetas...')
    if df['sentiment'].dtype == 'object':
        # Si es texto ('positive'/'negative'), convertir a 1/0
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    print('✓ Etiquetas codificadas')
    
    # 5. Dividir datos
    X_train, X_test, y_train, y_test = split_data(df)
    
    # 6. Crear vectorizador TF-IDF
    _, tfidf = create_vectorizers()
    
    # 7. Vectorizar
    X_train_vec, X_test_vec, fitted_vectorizer = vectorize_data(X_train, X_test, tfidf)
    
    # 8. Entrenar modelos
    models = train_all_models(X_train_vec, y_train)
    
    # 9. Guardar modelos
    save_models(models, fitted_vectorizer, output_dir)
    
    print('\n' + '='*70)
    print('✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE')
    print('='*70)
    
    return {
        'models': models,
        'vectorizer': fitted_vectorizer,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }


if __name__ == '__main__':
    print('Modulo train_models OK')
