import os
import re
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    # quitar tags HTML básicos
    text = re.sub(r"<.*?>", " ", text)
    # mantener palabras y espacios
    text = re.sub(r"[^\w\sáéíóúÁÉÍÓÚñÑüÜ]", " ", text)
    return text.lower().strip()


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
    if keywords is None:
        keywords = DEFAULT_KEYWORDS

    df = pd.read_csv(csv_path)
    # Soporte básico: buscar columna 'review' o 'text'
    col = None
    for c in ("review", "text", "texto"):
        if c in df.columns:
            col = c
            break
    if col is None:
        raise ValueError("El CSV debe contener una columna 'review' o 'text' con los textos de reseña.")

    texts = df[col].astype(str).apply(clean_text).tolist()

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)

    # intentamos calcular centroides por clustering (más robusto que el vector promedio)
    centroids = None
    try:
        # reducir muestra si el dataset es grande
        if X.shape[0] > sample_limit:
            # sample indices without replacement
            idx = np.random.choice(X.shape[0], sample_limit, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        kmeans = MiniBatchKMeans(n_clusters=min(n_clusters, X_sample.shape[0]), random_state=123)
        kmeans.fit(X_sample)
        centroids = kmeans.cluster_centers_
        # Normalizar centroides (L2) para que la similitud coseno funcione mejor
        try:
            norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            centroids = centroids / norms
        except Exception:
            # si algo falla, dejamos los centroides tal cual
            pass
    except Exception:
        # fallback: vector promedio
        centroids = np.asarray(X.mean(axis=0)).ravel().reshape(1, -1)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "centroids": centroids, "keywords": keywords}, model_path)
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
