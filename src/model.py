import os
import re
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans

# NLTK tokenization, stopwords, and lemmatization
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Descargando recursos NLTK (punkt_tab)...")
    nltk.download('punkt_tab', quiet=True)
    print("✓ Recursos NLTK descargados")

# Ensure NLTK stopwords are available
try:
    stopwords.words('english')
except LookupError:
    print("Descargando recursos NLTK (stopwords)...")
    nltk.download('stopwords', quiet=True)
    print("✓ Stopwords descargados")

# Ensure NLTK WordNet is available (for lemmatization)
try:
    wordnet.ensure_loaded()
except LookupError:
    print("Descargando recursos NLTK (wordnet)...")
    nltk.download('wordnet', quiet=True)
    print("✓ WordNet descargado")

# Ensure NLTK averaged_perceptron_tagger is available (for POS tagging)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    print("Descargando recursos NLTK (averaged_perceptron_tagger_eng)...")
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    print("✓ POS tagger descargado")

# Load English stopwords as a set for O(1) lookup performance
STOP_WORDS = set(stopwords.words('english'))

# Initialize lemmatizer globally (avoid recreating for each call)
LEMMATIZER = WordNetLemmatizer()


def get_wordnet_pos(treebank_tag: str) -> str:
    """Map Penn Treebank POS tags to WordNet POS tags.
    
    NLTK's pos_tag() returns Penn Treebank tags (e.g., 'NN', 'VBD', 'JJ'),
    but WordNetLemmatizer expects WordNet tags ('n', 'v', 'a', 'r').
    
    This function maps between the two tag systems for accurate lemmatization.
    
    Args:
        treebank_tag: Penn Treebank POS tag from NLTK pos_tag()
    
    Returns:
        WordNet POS tag ('n', 'v', 'a', 'r', or 'n' as default)
    
    Mapping:
        - J* (JJ, JJR, JJS) → 'a' (adjective)
        - V* (VB, VBD, VBG, VBN, VBP, VBZ) → 'v' (verb)
        - N* (NN, NNS, NNP, NNPS) → 'n' (noun)
        - R* (RB, RBR, RBS) → 'r' (adverb)
        - Default → 'n' (noun)
    
    Examples:
        >>> get_wordnet_pos('VBD')  # past tense verb
        'v'
        >>> get_wordnet_pos('JJR')  # comparative adjective
        'a'
        >>> get_wordnet_pos('NNS')  # plural noun
        'n'
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ  # 'a'
    elif treebank_tag.startswith('V'):
        return wordnet.VERB  # 'v'
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN  # 'n'
    elif treebank_tag.startswith('R'):
        return wordnet.ADV  # 'r'
    else:
        return wordnet.NOUN  # 'n' (default)


def clean_text(text: str, return_tokens: bool = False, remove_stopwords: bool = True, lemmatize: bool = True, 
               normalize_numbers: str = 'remove') -> str:
    """Clean and tokenize text using NLTK with advanced preprocessing.
    
    This function performs a complete NLP preprocessing pipeline:
    1. Removes HTML tags
    2. Removes URLs and emails
    3. Processes mentions (@user) and hashtags (#tag)
    4. Normalizes numbers (remove, replace with <NUM>, or keep)
    5. Reduces repeated characters (sooooo → soo)
    6. Tokenizes using NLTK (handles contractions like "don't" → ["do", "n't"])
    7. Cleans each token individually (removes non-alphanumeric except accents)
    8. Converts to lowercase
    9. Filters out empty tokens
    10. Removes stopwords (optional, enabled by default)
    11. Lemmatizes with POS tagging (optional, enabled by default)
    
    Args:
        text: Input text to clean
        return_tokens: If True, returns list of tokens; if False, returns joined string
        remove_stopwords: If True, filters out English stopwords (default: True)
        lemmatize: If True, applies lemmatization with POS tagging (default: True)
        normalize_numbers: How to handle numbers (default: 'remove')
            - 'remove': Delete all numbers
            - 'token': Replace with <NUM> token
            - 'keep': Keep numbers as-is
    
    Returns:
        Cleaned text as string (or list of tokens if return_tokens=True)
    
    Examples:
        >>> clean_text("Check out http://example.com for more info!")
        "check info"  # URL removed
        
        >>> clean_text("Email me at user@example.com")
        "email"  # Email removed
        
        >>> clean_text("@john said #awesome movie!")
        "say awesome movie"  # Mention removed, hashtag preserved
        
        >>> clean_text("Sooooo good! Rated 10/10", normalize_numbers='token')
        "soo good rate <NUM>"  # Repeated chars normalized, numbers → <NUM>
        
        >>> clean_text("I loved the acting in 2024", normalize_numbers='keep')
        "love act 2024"  # Numbers preserved
    
    Note:
        - URLs and emails are removed (unique, non-generalizable)
        - Mentions (@user) removed, hashtags preserved (content signal)
        - Numbers: default 'remove' (reduces vocab size, but loses year/rating info)
        - Character repetition normalized to max 2 (sooooo → soo)
        - Stopwords use set() for O(1) lookup performance
        - Lemmatization uses POS tagging for accurate results
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Step 1: Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    
    # Step 2: Remove URLs (http, https, www)
    text = re.sub(r'http\S+|https\S+|www\.\S+', ' ', text)
    
    # Step 3: Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Step 4: Remove mentions (@user)
    text = re.sub(r'@\w+', ' ', text)
    
    # Step 5: Process hashtags - keep the text content, remove the #
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Step 6: Normalize numbers based on strategy
    if normalize_numbers == 'remove':
        # Remove all numbers completely
        text = re.sub(r'\d+', ' ', text)
    elif normalize_numbers == 'token':
        # Replace numbers with special <NUM> token
        text = re.sub(r'\d+', ' <NUM> ', text)
    # else: 'keep' - do nothing, preserve numbers
    
    # Step 7: Reduce repeated characters (3+ → 2)
    # "sooooo" → "soo", "hellooooo" → "helloo"
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Step 8: Tokenize with NLTK (handles contractions, punctuation, etc.)
    try:
        tokens = word_tokenize(text)
    except Exception as e:
        # Fallback to simple split if tokenization fails
        print(f"⚠️  Tokenization failed: {e}. Using simple split.")
        tokens = text.split()
    
    # Step 9: Clean each token individually
    cleaned_tokens = []
    for token in tokens:
        # Keep alphanumeric + accented characters + <NUM> token
        # This preserves contractions like "n't" but removes standalone punctuation
        if token == '<NUM>':
            cleaned_tokens.append(token)
        else:
            cleaned = re.sub(r"[^\w\sáéíóúÁÉÍÓÚñÑüÜ']", "", token)
            cleaned = cleaned.strip()
            if cleaned:  # Only keep non-empty tokens
                cleaned_tokens.append(cleaned.lower())
    
    # Step 10: Remove stopwords (using set for O(1) lookup)
    if remove_stopwords:
        cleaned_tokens = [token for token in cleaned_tokens if token not in STOP_WORDS]
    
    # Step 11: Lemmatization with POS tagging
    if lemmatize and cleaned_tokens:
        try:
            # Get POS tags for remaining tokens (after stopword removal)
            pos_tagged = pos_tag(cleaned_tokens)
            
            # Lemmatize each token with its corresponding POS tag
            lemmatized_tokens = []
            for word, pos in pos_tagged:
                # Skip special tokens like <NUM>
                if word.startswith('<') and word.endswith('>'):
                    lemmatized_tokens.append(word)
                    continue
                
                # Map Penn Treebank POS tag to WordNet POS tag
                wordnet_pos = get_wordnet_pos(pos)
                # Apply lemmatization with POS context
                lemma = LEMMATIZER.lemmatize(word, pos=wordnet_pos)
                lemmatized_tokens.append(lemma)
            
            cleaned_tokens = lemmatized_tokens
        except Exception as e:
            # If lemmatization fails, continue with non-lemmatized tokens
            print(f"⚠️  Lemmatization failed: {e}. Skipping lemmatization.")
    
    # Step 12: Return as list or joined string
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
