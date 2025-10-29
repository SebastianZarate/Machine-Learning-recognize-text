import os
import re
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans

# NLTK tokenization
import nltk
from nltk.tokenize import word_tokenize

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Descargando recursos NLTK (punkt_tab)...")
    nltk.download('punkt_tab', quiet=True)
    print("✓ Recursos NLTK descargados")


def clean_text(text: str, return_tokens: bool = False) -> str:
    """Clean and tokenize text using NLTK word_tokenize.
    
    This function:
    1. Removes HTML tags
    2. Tokenizes using NLTK (handles contractions like "don't" → ["do", "n't"])
    3. Cleans each token individually (removes non-alphanumeric except accents)
    4. Converts to lowercase
    5. Filters out empty tokens
    
    Args:
        text: Input text to clean
        return_tokens: If True, returns list of tokens; if False, returns joined string
    
    Returns:
        Cleaned text as string (or list of tokens if return_tokens=True)
    
    Examples:
        >>> clean_text("I don't like this movie!")
        "i do n't like this movie"
        
        >>> clean_text("It's great!", return_tokens=True)
        ['it', "'s", 'great']
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Step 1: Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    
    # Step 2: Tokenize with NLTK (handles contractions, punctuation, etc.)
    try:
        tokens = word_tokenize(text)
    except Exception as e:
        # Fallback to simple split if tokenization fails
        print(f"⚠️  Tokenization failed: {e}. Using simple split.")
        tokens = text.split()
    
    # Step 3: Clean each token individually
    cleaned_tokens = []
    for token in tokens:
        # Keep alphanumeric + accented characters
        # This preserves contractions like "n't" but removes standalone punctuation
        cleaned = re.sub(r"[^\w\sáéíóúÁÉÍÓÚñÑüÜ']", "", token)
        cleaned = cleaned.strip()
        if cleaned:  # Only keep non-empty tokens
            cleaned_tokens.append(cleaned.lower())
    
    # Step 4: Return as list or joined string
    if return_tokens:
        return cleaned_tokens
    else:
        return " ".join(cleaned_tokens)


DEFAULT_KEYWORDS = [
    "película",
    "pelicula",
    "reseña",
    "reseña",
    "reseña de",
    "director",
    "actor",
    "actriz",
    "trama",
    "escena",
    "guion",
    "interpretación",
    "fotografía",
    "banda sonora",
    "critica",
    "crítica",
    "cinta",
    "rodaje",
    "festiva",
    "rating",
]


# Palabras/expresiones evaluativas típicas en reseñas (opinión)
EVALUATIVE_WORDS = [
    "recomiend",
    "no recomiendo",
    "me gust",
    "me encant",
    "encant",
    "me aburr",
    "no me gust",
    "excelent",
    "bueno",
    "malo",
    "pésim",
    "magnífic",
    "destac",
    "flojo",
    "promet",
    "impresion",
    "sobresal",
    "interpret",
    "emocion",
    "horrend",
    "terribl",
    "fantást",
    "entreten",
    "brillant",
    "impecabl",
]


# Indicadores de primera persona o frases típicas de reseña
FIRST_PERSON_PATTERNS = [
    "yo ",
    " me ",
    "mi ",
    "mi opinión",
    "vi ",
    "vi la",
    "vi el",
    "vi esta",
    "me parec",
    "salí",
    "salí",
]


def train_from_csv(csv_path: str, model_path: str = "models/review_model.joblib", keywords: List[str] = None, n_clusters: int = 50, sample_limit: int = 20000) -> str:
    """Entrena un vectorizador TF-IDF sobre la columna 'review' del CSV y guarda un artefacto que incluye:
    - vectorizer
    - avg_vector (vector promedio del corpus de reseñas)
    - keywords

    Devuelve la ruta al modelo guardado.
    """
    print("\n" + "="*70)
    print("🚀 INICIO DEL PROCESO DE ENTRENAMIENTO")
    print("="*70 + "\n")
    
    if keywords is None:
        keywords = DEFAULT_KEYWORDS
    
    print(f"📂 Cargando dataset desde: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Dataset cargado: {len(df)} registros encontrados\n")
    
    # Soporte básico: buscar columna 'review' o 'text'
    col = None
    for c in ("review", "text", "texto"):
        if c in df.columns:
            col = c
            break
    if col is None:
        raise ValueError("El CSV debe contener una columna 'review' o 'text' con los textos de reseña.")
    
    print(f"📝 Columna de texto detectada: '{col}'")
    print(f"🧹 Limpiando y procesando {len(df)} textos...")
    
    texts = df[col].astype(str).apply(clean_text).tolist()
    print(f"✓ Textos procesados correctamente\n")
    
    print(f"🔧 Configurando vectorizador TF-IDF...")
    print(f"   • max_features: 10000")
    print(f"   • ngram_range: (1, 2)")
    
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    
    print(f"\n⚙️  Entrenando vectorizador TF-IDF...")
    X = vectorizer.fit_transform(texts)
    print(f"✓ Vectorización completada")
    print(f"   • Matriz generada: {X.shape[0]} documentos x {X.shape[1]} características\n")

    # intentamos calcular centroides por clustering (más robusto que el vector promedio)
    centroids = None
    try:
        print(f"🔍 Calculando centroides mediante clustering...")
        
        # reducir muestra si el dataset es grande
        if X.shape[0] > sample_limit:
            print(f"   • Dataset grande detectado ({X.shape[0]} documentos)")
            print(f"   • Reduciendo muestra a {sample_limit} documentos para clustering")
            idx = np.random.choice(X.shape[0], sample_limit, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X
            print(f"   • Usando dataset completo para clustering ({X.shape[0]} documentos)")

        actual_clusters = min(n_clusters, X_sample.shape[0])
        print(f"   • Número de clusters: {actual_clusters}")
        print(f"   • Ejecutando MiniBatchKMeans...")
        
        kmeans = MiniBatchKMeans(n_clusters=actual_clusters, random_state=123)
        kmeans.fit(X_sample)
        centroids = kmeans.cluster_centers_
        
        print(f"✓ Clustering completado: {centroids.shape[0]} centroides generados")

        # Normalizar centroides (L2) para que la similitud coseno funcione mejor
        try:
            print(f"   • Normalizando centroides (L2)...")
            norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            centroids = centroids / norms
            print(f"✓ Centroides normalizados correctamente\n")
        except Exception as e:
            print(f"⚠️  Advertencia: No se pudieron normalizar centroides: {e}\n")
            
    except Exception as e:
        print(f"⚠️  Advertencia: Clustering falló, usando vector promedio")
        print(f"   Error: {e}")
        centroids = np.asarray(X.mean(axis=0)).ravel().reshape(1, -1)
        print(f"✓ Vector promedio calculado como fallback\n")

    print(f"💾 Guardando modelo en: {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model_data = {
        "vectorizer": vectorizer, 
        "centroids": centroids, 
        "keywords": keywords
    }
    
    joblib.dump(model_data, model_path)
    
    file_size = os.path.getsize(model_path) / 1024 / 1024  # MB
    print(f"✓ Modelo guardado exitosamente")
    print(f"   • Tamaño del archivo: {file_size:.2f} MB")
    print(f"   • Palabras clave incluidas: {len(keywords)}")
    
    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("="*70 + "\n")
    
    return model_path


def predict_text(text: str, model_path: str = "models/review_model.joblib", keyword_weight: float = 0.25, sim_weight: float = 0.15, eval_weight: float = 0.4, first_person_weight: float = 0.2) -> Dict[str, Any]:
    """Predice si `text` es una reseña de cine combinando keywords y similitud coseno.

    Retorna un diccionario con: is_review (bool), probability (0..1), similarity, keyword_score (0..1), matched_keywords (list)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}. Entrene primero con train_from_csv().")

    obj = joblib.load(model_path)
    vectorizer: TfidfVectorizer = obj["vectorizer"]
    centroids = obj.get("centroids", None)
    keywords: List[str] = obj.get("keywords", DEFAULT_KEYWORDS)

    text_clean = clean_text(text)

    # keywords
    matched = [k for k in keywords if k in text_clean]
    keyword_score = len(matched) / len(keywords) if keywords else 0.0

    # evaluative words (señal de opinión)
    eval_matched = [w for w in EVALUATIVE_WORDS if w in text_clean]
    eval_score = min(1.0, len(eval_matched) / 3.0) if EVALUATIVE_WORDS else 0.0

    # primera persona / frases típicas de reseña
    first_person_matched = [p for p in FIRST_PERSON_PATTERNS if p in text_clean]
    first_person_score = 1.0 if len(first_person_matched) > 0 else 0.0

    vec = vectorizer.transform([text_clean])

    sim = 0.0
    try:
        if centroids is None:
            sim = 0.0
        else:
            # centroids puede ser (k, n_features) o vector (1, n_features)
            sims = cosine_similarity(vec, centroids)
            # tomar la media de las top-3 similitudes para evitar depender de un solo centro
            if sims.ndim == 2:
                topk = min(3, sims.shape[1])
                top_vals = np.sort(sims[0])[::-1][:topk]
                sim = float(np.mean(top_vals))
            else:
                sim = float(sims)
    except Exception:
        sim = 0.0

    final = sim_weight * sim + keyword_weight * keyword_score + eval_weight * eval_score + first_person_weight * first_person_score
    # normalizar si centroids es un solo vector promedio que produce valores bajos
    # límite final en [0,1]
    final = max(0.0, min(1.0, final))

    is_review = final >= 0.35

    return {
        "is_review": bool(is_review),
        "probability": float(final),
        "similarity": float(sim),
        "keyword_score": float(keyword_score),
        "eval_score": float(eval_score),
        "first_person_score": float(first_person_score),
        "matched_keywords": matched,
        "matched_evaluative": eval_matched,
        "matched_first_person": first_person_matched,
    }


if __name__ == "__main__":
    print("Módulo de modelo. Use train_from_csv() y predict_text().")
