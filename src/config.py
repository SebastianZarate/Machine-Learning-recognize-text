"""
Configuración centralizada del proyecto de Análisis de Sentimientos.

Este módulo contiene todas las configuraciones y parámetros utilizados
en el proyecto para asegurar consistencia entre los diferentes módulos.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS DEL PROYECTO
# ============================================================================

# Directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent

# Directorios de datos
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Archivo del dataset
IMDB_DATASET_PATH = PROJECT_ROOT / "IMDB Dataset.csv"

# Crear directorios si no existen
for directory in [DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)

# ============================================================================
# CONFIGURACIÓN DE PREPROCESAMIENTO
# ============================================================================

PREPROCESSING_CONFIG = {
    # Stopwords
    'remove_stopwords': True,
    'stopwords_language': 'english',
    
    # Lematización
    'apply_lemmatization': True,
    'use_pos_tagging': True,  # Para lematización más precisa
    
    # Limpieza
    'lowercase': True,
    'remove_html': True,
    'remove_urls': True,
    'remove_emails': True,
    'remove_special_chars': True,
    'remove_numbers': False,  # Los números pueden ser relevantes en reseñas
    
    # Filtros
    'min_word_length': 2,
    'max_word_length': 50,
}

# ============================================================================
# CONFIGURACIÓN DE TF-IDF (VECTORIZACIÓN)
# ============================================================================

TFIDF_CONFIG = {
    'max_features': 5000,  # Top 5000 palabras más frecuentes
    'ngram_range': (1, 2),  # Unigramas y bigramas
    'min_df': 5,  # Palabra debe aparecer en al menos 5 documentos
    'max_df': 0.95,  # Ignorar palabras que aparecen en >95% de documentos
    'sublinear_tf': True,  # Aplicar escalado logarítmico a TF
    'use_idf': True,
    'smooth_idf': True,
    'norm': 'l2',  # Normalización L2
}

# ============================================================================
# CONFIGURACIÓN DE MODELOS
# ============================================================================

MODEL_CONFIG = {
    'naive_bayes': {
        'alpha': 1.0,  # Parámetro de suavizado de Laplace
    },
    
    'logistic_regression': {
        'max_iter': 1000,
        'random_state': 42,
        'n_jobs': -1,  # Usar todos los cores disponibles
        'C': 1.0,  # Regularización inversa
        'solver': 'lbfgs',
    },
    
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 50,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1,
        'max_features': 'sqrt',
    }
}

# ============================================================================
# CONFIGURACIÓN DE TRAIN/TEST SPLIT
# ============================================================================

TRAIN_TEST_SPLIT = {
    'test_size': 0.2,  # 80% entrenamiento, 20% prueba
    'random_state': 42,
    'stratify': True,  # Mantener proporción de clases
    'shuffle': True,
}

# ============================================================================
# CONFIGURACIÓN DE EVALUACIÓN
# ============================================================================

EVALUATION_CONFIG = {
    # Métricas a calcular
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    
    # Configuración de visualizaciones
    'figsize': (10, 6),
    'dpi': 100,
    'style': 'seaborn-v0_8-darkgrid',
    
    # Guardar resultados
    'save_figures': True,
    'save_reports': True,
}

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': str(PROJECT_ROOT / 'ml_project.log'),
            'mode': 'a',
            'encoding': 'utf-8',
        },
    },
    'loggers': {
        '': {  # Root logger
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}

# ============================================================================
# CONFIGURACIÓN DE GUI
# ============================================================================

GUI_CONFIG = {
    'window_title': 'Análisis de Sentimientos - IMDB Reviews',
    'window_size': '900x700',
    'font_family': 'Segoe UI',
    'font_sizes': {
        'title': 16,
        'subtitle': 12,
        'text': 10,
    },
    'colors': {
        'bg': '#f0f0f0',
        'positive': '#4CAF50',
        'negative': '#f44336',
        'neutral': '#9E9E9E',
    },
    'default_model': 'logistic_regression',
}

# ============================================================================
# NOMBRES DE ARCHIVOS DE MODELOS
# ============================================================================

MODEL_FILES = {
    'naive_bayes': MODELS_DIR / 'naive_bayes.joblib',
    'logistic_regression': MODELS_DIR / 'logistic_regression.joblib',
    'random_forest': MODELS_DIR / 'random_forest.joblib',
    'vectorizer': MODELS_DIR / 'vectorizer.joblib',
    'label_encoder': MODELS_DIR / 'label_encoder.joblib',
}

# ============================================================================
# CONSTANTES DEL PROYECTO
# ============================================================================

# Etiquetas de sentimiento
SENTIMENT_LABELS = {
    0: 'negativa',
    1: 'positiva',
}

# Versión del proyecto
PROJECT_VERSION = '1.0.0'

# Semilla aleatoria global
RANDOM_STATE = 42

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def get_model_path(model_name: str) -> Path:
    """
    Obtiene la ruta completa de un modelo guardado.
    
    Args:
        model_name: Nombre del modelo ('naive_bayes', 'logistic_regression', 'random_forest')
    
    Returns:
        Path: Ruta completa al archivo del modelo
    
    Raises:
        ValueError: Si el nombre del modelo no es válido
    """
    if model_name not in MODEL_FILES:
        raise ValueError(f"Modelo desconocido: {model_name}. "
                        f"Modelos válidos: {list(MODEL_FILES.keys())}")
    return MODEL_FILES[model_name]


def model_exists(model_name: str) -> bool:
    """
    Verifica si un modelo entrenado existe en disco.
    
    Args:
        model_name: Nombre del modelo a verificar
    
    Returns:
        bool: True si el modelo existe, False en caso contrario
    """
    try:
        model_path = get_model_path(model_name)
        return model_path.exists()
    except ValueError:
        return False


def all_models_trained() -> bool:
    """
    Verifica si todos los modelos y el vectorizador están entrenados.
    
    Returns:
        bool: True si todos los archivos existen, False en caso contrario
    """
    return all(path.exists() for path in MODEL_FILES.values())


if __name__ == '__main__':
    # Verificar configuración
    print("=" * 70)
    print("CONFIGURACIÓN DEL PROYECTO")
    print("=" * 70)
    print(f"\n📁 Directorio raíz: {PROJECT_ROOT}")
    print(f"📁 Directorio de datos: {DATA_DIR}")
    print(f"📁 Directorio de modelos: {MODELS_DIR}")
    print(f"📁 Directorio de resultados: {RESULTS_DIR}")
    print(f"\n📊 Dataset: {IMDB_DATASET_PATH}")
    print(f"   Existe: {'✅ Sí' if IMDB_DATASET_PATH.exists() else '❌ No'}")
    
    print(f"\n🤖 Modelos configurados:")
    for model_name, model_path in MODEL_FILES.items():
        exists = '✅' if model_path.exists() else '❌'
        print(f"   {exists} {model_name}: {model_path.name}")
    
    print(f"\n🔧 Configuración TF-IDF:")
    for key, value in TFIDF_CONFIG.items():
        print(f"   {key}: {value}")
    
    print(f"\n✅ Estado: {'Todos los modelos entrenados' if all_models_trained() else 'Modelos pendientes de entrenamiento'}")
    print("=" * 70)
