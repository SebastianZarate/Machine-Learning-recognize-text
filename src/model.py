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


# ============================================================================
# TEXT PREPROCESSING PIPELINE - Modular Components
# ============================================================================
# Each function is pure, testable, and performs a single responsibility.
# These can be composed in different orders for experimentation.
# ============================================================================

def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text.
    
    Args:
        text: Input text potentially containing HTML
    
    Returns:
        Text with HTML tags removed
    
    Example:
        >>> remove_html_tags("<p>Great <b>movie</b>!</p>")
        "Great movie!"
    """
    return re.sub(r"<.*?>", " ", text)


def remove_urls(text: str) -> str:
    """Remove URLs (http, https, www) from text.
    
    URLs are unique identifiers that don't generalize and contaminate
    the vocabulary without adding semantic value for classification.
    
    Args:
        text: Input text potentially containing URLs
    
    Returns:
        Text with URLs removed
    
    Example:
        >>> remove_urls("Check http://example.com or www.site.org")
        "Check  or "
    """
    return re.sub(r'http\S+|https\S+|www\.\S+', ' ', text)


def remove_emails(text: str) -> str:
    """Remove email addresses from text.
    
    Email addresses are unique identifiers that don't contribute to
    text classification tasks.
    
    Args:
        text: Input text potentially containing emails
    
    Returns:
        Text with email addresses removed
    
    Example:
        >>> remove_emails("Contact user@example.com for info")
        "Contact  for info"
    """
    return re.sub(r'\S+@\S+', ' ', text)


def remove_mentions(text: str) -> str:
    """Remove social media mentions (@user) from text.
    
    Mentions are user references that don't add semantic content value.
    
    Args:
        text: Input text potentially containing mentions
    
    Returns:
        Text with mentions removed
    
    Example:
        >>> remove_mentions("@john said this movie rocks")
        " said this movie rocks"
    """
    return re.sub(r'@\w+', ' ', text)


def process_hashtags(text: str) -> str:
    """Convert hashtags to plain text (remove # symbol, keep content).
    
    Hashtags are content markers. We preserve the semantic content
    but remove the # symbol since it doesn't add value.
    
    Args:
        text: Input text potentially containing hashtags
    
    Returns:
        Text with hashtags converted to plain words
    
    Example:
        >>> process_hashtags("This #awesome #movie is great!")
        "This awesome movie is great!"
    """
    return re.sub(r'#(\w+)', r'\1', text)


def normalize_numbers(text: str, strategy: str = 'remove') -> str:
    """Normalize numbers in text according to strategy.
    
    Args:
        text: Input text potentially containing numbers
        strategy: How to handle numbers
            - 'remove': Delete all numbers (default, reduces vocab)
            - 'token': Replace with <NUM> special token (balance)
            - 'keep': Preserve numbers as-is (max info, larger vocab)
    
    Returns:
        Text with numbers normalized
    
    Examples:
        >>> normalize_numbers("Movie from 2024 rated 10/10", 'remove')
        "Movie from  rated /"
        
        >>> normalize_numbers("Movie from 2024 rated 10/10", 'token')
        "Movie from <NUM> rated <NUM>/<NUM>"
        
        >>> normalize_numbers("Movie from 2024 rated 10/10", 'keep')
        "Movie from 2024 rated 10/10"
    
    Note:
        - 'remove': Best for reducing vocabulary size
        - 'token': Recommended - maintains numeric signal without vocab explosion
        - 'keep': Only if years/ratings are important features
    """
    if strategy == 'remove':
        return re.sub(r'\d+', ' ', text)
    elif strategy == 'token':
        return re.sub(r'\d+', ' <NUM> ', text)
    else:  # 'keep'
        return text


def reduce_repeated_chars(text: str, max_repetitions: int = 2) -> str:
    """Reduce repeated characters to maximum repetitions.
    
    Handles emphatic expressions in social media text while normalizing
    excessive repetitions (e.g., "sooooo" → "soo").
    
    Args:
        text: Input text potentially containing repeated characters
        max_repetitions: Maximum allowed character repetitions (default: 2)
    
    Returns:
        Text with repeated characters normalized
    
    Examples:
        >>> reduce_repeated_chars("Sooooo good!!!", 2)
        "Soo good!!"
        
        >>> reduce_repeated_chars("Amaaazing", 1)
        "Amazing"
    
    Note:
        max_repetitions=2 balances normalization with preserving emphasis.
        "good" ≠ "goood" (different emotional intensity)
    """
    return re.sub(r'(.)\1{' + str(max_repetitions) + r',}', r'\1' * max_repetitions, text)


def tokenize_text(text: str) -> List[str]:
    """Tokenize text using NLTK word_tokenize.
    
    Handles contractions, punctuation, and special cases better than
    simple split(). Falls back to split() if tokenization fails.
    
    Args:
        text: Input text to tokenize
    
    Returns:
        List of tokens
    
    Examples:
        >>> tokenize_text("I don't like this!")
        ['I', 'do', "n't", 'like', 'this', '!']
        
        >>> tokenize_text("Dr. Smith's work")
        ['Dr.', 'Smith', "'s", 'work']
    
    Note:
        Contractions are split: "don't" → ["do", "n't"]
        This is correct for downstream lemmatization.
    """
    try:
        return word_tokenize(text)
    except Exception as e:
        print(f"⚠️  Tokenization failed: {e}. Using simple split.")
        return text.split()


def clean_tokens(tokens: List[str], preserve_special: bool = True) -> List[str]:
    """Clean individual tokens and convert to lowercase.
    
    Removes non-alphanumeric characters except accents and apostrophes.
    Filters out empty tokens.
    
    Args:
        tokens: List of tokens to clean
        preserve_special: If True, preserves tokens like <NUM>
    
    Returns:
        List of cleaned tokens in lowercase
    
    Examples:
        >>> clean_tokens(['Good', '!', 'movie', '<NUM>'])
        ['good', 'movie', '<NUM>']
        
        >>> clean_tokens(["it's", 'great', '!!!'])
        ["it's", 'great']
    """
    cleaned = []
    for token in tokens:
        # Preserve special tokens like <NUM>
        if preserve_special and token.startswith('<') and token.endswith('>'):
            cleaned.append(token)
        else:
            # Keep alphanumeric + accented chars + apostrophes
            clean = re.sub(r"[^\w\sáéíóúÁÉÍÓÚñÑüÜ']", "", token)
            clean = clean.strip()
            if clean:
                cleaned.append(clean.lower())
    return cleaned


def filter_stopwords(tokens: List[str], stopwords_set: set = None) -> List[str]:
    """Remove stopwords from token list.
    
    Args:
        tokens: List of tokens to filter
        stopwords_set: Set of stopwords (default: STOP_WORDS global)
    
    Returns:
        List of tokens with stopwords removed
    
    Example:
        >>> filter_stopwords(['the', 'movie', 'is', 'great'])
        ['movie', 'great']
    
    Note:
        Uses set for O(1) lookup. With ~200 English stopwords,
        this is significantly faster than list lookup.
    """
    if stopwords_set is None:
        stopwords_set = STOP_WORDS
    return [token for token in tokens if token not in stopwords_set]


def lemmatize_tokens_with_pos(tokens: List[str]) -> List[str]:
    """Lemmatize tokens using POS tagging for accuracy.
    
    Uses NLTK POS tagger to identify word type (noun, verb, adjective, etc.)
    before lemmatization. This significantly improves quality.
    
    Args:
        tokens: List of tokens to lemmatize
    
    Returns:
        List of lemmatized tokens
    
    Examples:
        >>> lemmatize_tokens_with_pos(['movies', 'were', 'amazing'])
        ['movie', 'be', 'amazing']
        
        >>> lemmatize_tokens_with_pos(['better', 'films'])
        ['well', 'film']  # 'better' correctly identified as adjective
    
    Note:
        Without POS tagging: 'better' → 'better' (assumes noun)
        With POS tagging: 'better' → 'good' (correctly as adjective)
        
        Special tokens like <NUM> are preserved.
    """
    if not tokens:
        return tokens
    
    try:
        pos_tagged = pos_tag(tokens)
        lemmatized = []
        
        for word, pos in pos_tagged:
            # Skip special tokens
            if word.startswith('<') and word.endswith('>'):
                lemmatized.append(word)
                continue
            
            # Map POS tag and lemmatize
            wordnet_pos = get_wordnet_pos(pos)
            lemma = LEMMATIZER.lemmatize(word, pos=wordnet_pos)
            lemmatized.append(lemma)
        
        return lemmatized
    except Exception as e:
        print(f"⚠️  Lemmatization failed: {e}. Returning original tokens.")
        return tokens


def preprocess_text(
    text: str,
    remove_stops: bool = True,
    lemmatize: bool = True,
    numbers_strategy: str = 'remove',
    return_tokens: bool = False
) -> Any:
    """Master preprocessing pipeline that orchestrates all cleaning steps.
    
    This function composes all preprocessing steps in the optimal order:
    1. Text-level cleaning (HTML, URLs, emails, mentions, hashtags, numbers, repetitions)
    2. Tokenization
    3. Token-level cleaning (lowercase, filter non-alphanumeric)
    4. Stopword removal (optional)
    5. Lemmatization with POS tagging (optional)
    
    Args:
        text: Input text to preprocess
        remove_stops: If True, removes stopwords (default: True)
        lemmatize: If True, applies lemmatization with POS (default: True)
        numbers_strategy: How to handle numbers - 'remove', 'token', or 'keep' (default: 'remove')
        return_tokens: If True, returns list; if False, returns string (default: False)
    
    Returns:
        Preprocessed text as string or list of tokens
    
    Examples:
        >>> preprocess_text("Check http://site.com! Movie rated 10/10 #awesome")
        "check movie rat awesome"
        
        >>> preprocess_text("The movies were amazing!", return_tokens=True)
        ['movie', 'amazing']
        
        >>> preprocess_text("Rated 8/10", numbers_strategy='token')
        "rat num num"
    
    Note:
        Each step is optional and can be toggled. This design allows for:
        - Easy debugging (disable steps to isolate issues)
        - A/B testing (compare with/without specific steps)
        - Experimentation (try different orderings)
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Phase 1: Text-level cleaning (before tokenization)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_mentions(text)
    text = process_hashtags(text)
    text = normalize_numbers(text, strategy=numbers_strategy)
    text = reduce_repeated_chars(text, max_repetitions=2)
    
    # Phase 2: Tokenization
    tokens = tokenize_text(text)
    
    # Phase 3: Token-level cleaning
    tokens = clean_tokens(tokens, preserve_special=True)
    
    # Phase 4: Stopword removal (optional)
    if remove_stops:
        tokens = filter_stopwords(tokens)
    
    # Phase 5: Lemmatization (optional)
    if lemmatize:
        tokens = lemmatize_tokens_with_pos(tokens)
    
    # Return format
    if return_tokens:
        return tokens
    else:
        return " ".join(tokens)


# Alias for backward compatibility
def clean_text(text: str, return_tokens: bool = False, remove_stopwords: bool = True, lemmatize: bool = True, 
               normalize_numbers: str = 'remove') -> Any:
    """Clean and tokenize text using NLTK with advanced preprocessing.
    
    DEPRECATED: This function is maintained for backward compatibility.
    New code should use preprocess_text() which has a cleaner API.
    
    This function delegates to preprocess_text() with mapped parameters.
    
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
        "check info"
        
        >>> clean_text("@john said #awesome movie!", return_tokens=True)
        ['say', 'awesome', 'movie']
    
    Note:
        For new code, prefer using preprocess_text() which has better
        parameter names and is more maintainable.
    """
    return preprocess_text(
        text=text,
        remove_stops=remove_stopwords,
        lemmatize=lemmatize,
        numbers_strategy=normalize_numbers,
        return_tokens=return_tokens
    )


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
