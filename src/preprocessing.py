"""
Módulo de preprocesamiento avanzado para análisis de sentimientos en reseñas IMDB.

Este módulo proporciona funciones para limpiar, normalizar y procesar texto
de manera eficiente para tareas de NLP.
"""

import re
import pandas as pd
from typing import Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Descargar recursos necesarios de NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


def load_imdb_dataset(csv_path: str) -> pd.DataFrame:
    """
    Carga el dataset IMDB desde un archivo CSV y realiza validaciones básicas.
    
    Args:
        csv_path: Ruta al archivo CSV del dataset IMDB
        
    Returns:
        DataFrame limpio con columnas 'review' y 'sentiment'
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si faltan columnas requeridas
    """
    try:
        print(f"📂 Cargando dataset desde: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Verificar columnas requeridas
        required_columns = ['review', 'sentiment']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Faltan columnas requeridas: {missing_columns}")
        
        # Eliminar filas con valores nulos
        initial_rows = len(df)
        df = df.dropna(subset=['review', 'sentiment'])
        removed_rows = initial_rows - len(df)
        
        if removed_rows > 0:
            print(f"⚠️  Se eliminaron {removed_rows} filas con valores nulos")
        
        print(f"✅ Dataset cargado: {len(df)} reseñas")
        return df
        
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {csv_path}")
        raise
    except Exception as e:
        print(f"❌ Error al cargar dataset: {str(e)}")
        raise


def clean_text_advanced(text: str) -> str:
    """
    Limpia y normaliza texto eliminando elementos no deseados.
    
    Args:
        text: Texto a limpiar
        
    Returns:
        Texto limpio y normalizado
    """
    if not isinstance(text, str):
        return ""
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar tags HTML
    text = re.sub(r'<[^>]+>', '', text)
    
    # Eliminar URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Eliminar emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # Eliminar menciones (@usuario)
    text = re.sub(r'@\w+', '', text)
    
    # Eliminar números
    text = re.sub(r'\d+', '', text)
    
    # Eliminar caracteres especiales (mantener solo letras y espacios)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Normalizar espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    
    # Eliminar espacios al inicio y final
    text = text.strip()
    
    return text


def remove_stopwords(text: str, language: str = 'english') -> str:
    """
    Elimina stopwords (palabras comunes sin valor semántico) del texto.
    
    Args:
        text: Texto a procesar
        language: Idioma de las stopwords (default: 'english')
        
    Returns:
        Texto sin stopwords
    """
    try:
        stop_words = set(stopwords.words(language))
        
        # Tokenizar por espacios
        words = text.split()
        
        # Filtrar stopwords
        filtered_words = [word for word in words if word not in stop_words]
        
        return ' '.join(filtered_words)
        
    except Exception as e:
        print(f"⚠️  Error al eliminar stopwords: {str(e)}")
        return text


def lemmatize_text(text: str) -> str:
    """
    Aplica lematización al texto para reducir palabras a su forma base.
    
    Args:
        text: Texto a lematizar
        
    Returns:
        Texto lematizado
    """
    try:
        lemmatizer = WordNetLemmatizer()
        
        # Tokenizar
        words = text.split()
        
        # Aplicar lematización
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(lemmatized_words)
        
    except Exception as e:
        print(f"⚠️  Error en lematización: {str(e)}")
        return text


def preprocess_pipeline(text: str) -> str:
    """
    Pipeline completo de preprocesamiento de texto.
    
    Aplica en orden:
    1. Limpieza avanzada
    2. Eliminación de stopwords
    3. Lematización
    
    Args:
        text: Texto a preprocesar
        
    Returns:
        Texto completamente preprocesado
    """
    text = clean_text_advanced(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el pipeline de preprocesamiento a todo el DataFrame.
    
    Args:
        df: DataFrame con columnas 'review' y 'sentiment'
        
    Returns:
        DataFrame con columna 'review_clean' y 'sentiment' numérico
    """
    try:
        print("\n🔄 Iniciando preprocesamiento del dataset...")
        
        # Crear copia para no modificar el original
        df_processed = df.copy()
        
        # Aplicar pipeline a cada reseña
        print("📝 Procesando reseñas...")
        df_processed['review_clean'] = df_processed['review'].apply(preprocess_pipeline)
        
        # Convertir sentiment a valores numéricos
        print("🔢 Convirtiendo sentimientos a valores numéricos...")
        sentiment_map = {'positive': 1, 'negative': 0}
        df_processed['sentiment'] = df_processed['sentiment'].map(sentiment_map)
        
        # Verificar si hay valores nulos después del mapeo
        if df_processed['sentiment'].isnull().any():
            print("⚠️  Advertencia: Se encontraron valores de sentimiento no reconocidos")
            df_processed = df_processed.dropna(subset=['sentiment'])
        
        # Eliminar reseñas vacías después del preprocesamiento
        initial_rows = len(df_processed)
        df_processed = df_processed[df_processed['review_clean'].str.len() > 0]
        removed_empty = initial_rows - len(df_processed)
        
        if removed_empty > 0:
            print(f"⚠️  Se eliminaron {removed_empty} reseñas vacías después del preprocesamiento")
        
        print(f"✅ Preprocesamiento completado: {len(df_processed)} reseñas procesadas")
        print(f"   - Reseñas positivas: {(df_processed['sentiment'] == 1).sum()}")
        print(f"   - Reseñas negativas: {(df_processed['sentiment'] == 0).sum()}")
        
        return df_processed
        
    except Exception as e:
        print(f"❌ Error en preprocesamiento del DataFrame: {str(e)}")
        raise


if __name__ == "__main__":
    # Ejemplo de uso básico
    print("Módulo de preprocesamiento cargado correctamente ✅")
