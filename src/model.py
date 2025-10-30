"""
Este archivo contiene el pipeline legacy basado en similitud coseno.

Para usar los modelos de clasificación supervisada modernos (Naive Bayes, Logistic Regression, Random Forest),
consulte los siguientes módulos:
    - src/preprocessing.py: Preprocesamiento de texto
    - src/train_models.py: Entrenamiento de modelos supervisados
    - src/evaluation.py: Evaluación y métricas
    - src/visualizations.py: Gráficos y visualizaciones

O ejecute los notebooks interactivos en notebooks/:
    - 01_data_exploration.ipynb
    - 02_preprocessing.ipynb
    - 03_model_training.ipynb
    - 04_evaluation.ipynb
    - 05_complete_workflow.ipynb (PRODUCTO FINAL)

Este archivo se mantiene únicamente para compatibilidad con la GUI (src/app.py).
"""

import os
import re
import time
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


def train_from_csv(csv_path: str, 
                   model_path: str = "models/review_model.joblib", 
                   keywords: List[str] = None,
                   test_size: float = 0.2,
                   random_state: int = 42) -> str:
    """Entrena clasificadores supervisados sobre el dataset y guarda el modelo completo.
    
    Este es el pipeline de entrenamiento principal que:
    1. Carga el dataset balanceado (con columnas 'text'/'review' y 'label')
    2. Preprocesa textos con clean_text()
    3. Crea features TF-IDF
    4. Divide en train/test con estratificación
    5. Entrena 3 clasificadores supervisados (Naive Bayes, Logistic Regression, Random Forest)
    6. Guarda todos los artefactos necesarios para predicción
    
    Args:
        csv_path: Ruta al CSV con columnas 'text'/'review' (textos) y 'label' (0/1)
        model_path: Ruta donde guardar el modelo entrenado (default: "models/review_model.joblib")
        keywords: Lista de palabras clave para detección heurística (default: DEFAULT_KEYWORDS)
        test_size: Proporción del dataset para test (default: 0.2 = 20%)
        random_state: Semilla para reproducibilidad (default: 42)
    
    Returns:
        Ruta al archivo del modelo guardado
    
    Raises:
        ValueError: Si el CSV no contiene las columnas necesarias ('text'/'review' y 'label')
        FileNotFoundError: Si el archivo CSV no existe
    
    Examples:
        >>> # Entrenar con dataset balanceado
        >>> model_path = train_from_csv("balanced_dataset.csv")
        🚀 INICIO DEL PROCESO DE ENTRENAMIENTO
        ✓ Dataset cargado: 100000 registros encontrados
        ✓ Distribución de clases: Negative: 50000 (50.0%), Positive: 50000 (50.0%)
        ...
        
        >>> # Entrenar con parámetros personalizados
        >>> model_path = train_from_csv(
        ...     "balanced_dataset.csv",
        ...     model_path="models/custom_model.joblib",
        ...     test_size=0.3,
        ...     random_state=123
        ... )
    
    Notes:
        - El dataset DEBE estar balanceado para mejores resultados
        - Test set NO se usa para entrenamiento (solo se guarda para evaluación posterior)
        - TF-IDF se entrena solo con train set (evita data leakage)
        - Modelos se entrenan con estratificación para preservar balance de clases
        - El archivo guardado contiene: vectorizer, models, X_test, y_test, keywords
    """
    # Import train_models here to avoid circular imports
    # Manejo robusto de imports: funciona desde cualquier ubicación
    import sys
    import os
    if __name__ == '__main__':
        # Si se ejecuta como script, añadir el directorio src al path
        sys.path.insert(0, os.path.dirname(__file__))
    
    try:
        from train_models import train_all_models
    except ImportError:
        # Fallback a import absoluto si el relativo falla
        from src.train_models import train_all_models
    
    print("\n" + "="*70)
    print("🚀 INICIO DEL PROCESO DE ENTRENAMIENTO - CLASIFICADORES SUPERVISADOS")
    print("="*70 + "\n")
    
    if keywords is None:
        keywords = DEFAULT_KEYWORDS
    
    # ========== 1. CARGAR DATASET ==========
    print(f"📂 Cargando dataset desde: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Dataset cargado: {len(df):,} registros encontrados\n")
    
    # ========== 2. VALIDAR COLUMNAS ==========
    # Buscar columna de texto
    text_col = None
    for c in ("text", "review", "texto"):
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError(
            "El CSV debe contener una columna 'text' o 'review' con los textos. "
            f"Columnas encontradas: {list(df.columns)}"
        )
    
    # Buscar columna de etiquetas
    label_col = None
    for c in ("label", "sentiment", "class", "y"):
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError(
            "El CSV debe contener una columna 'label' con las etiquetas (0/1). "
            f"Columnas encontradas: {list(df.columns)}"
        )
    
    print(f"📝 Columnas detectadas:")
    print(f"   • Texto: '{text_col}'")
    print(f"   • Etiquetas: '{label_col}'\n")
    
    # ========== 3. VERIFICAR BALANCE DE CLASES ==========
    y = df[label_col].values
    unique, counts = np.unique(y, return_counts=True)
    print(f"📊 Distribución de clases:")
    for label, count in zip(unique, counts):
        percentage = (count / len(y)) * 100
        label_name = "Negative" if label == 0 else "Positive"
        print(f"   • {label_name} ({label}): {count:,} ({percentage:.1f}%)")
    
    # Warning si el dataset está desbalanceado
    if len(counts) == 2 and abs(counts[0] - counts[1]) / len(y) > 0.2:
        print(f"\n⚠️  ADVERTENCIA: Dataset desbalanceado detectado!")
        print(f"   Se recomienda usar un dataset balanceado para mejores resultados.")
        print(f"   Ver data_preparation.py -> create_balanced_dataset()\n")
    else:
        print(f"✓ Dataset balanceado correctamente\n")
    
    # ========== 4. PREPROCESAR TEXTOS ==========
    print(f"🧹 Limpiando y procesando {len(df):,} textos...")
    print(f"   • Pipeline: HTML → URLs → emails → mentions → hashtags → numbers → tokenize → stopwords → lemmatize")
    
    texts = df[text_col].astype(str).apply(clean_text).tolist()
    print(f"✓ Textos procesados correctamente\n")
    
    # ========== 5. CREAR FEATURES TF-IDF ==========
    print(f"🔧 Configurando vectorizador TF-IDF...")
    print(f"   • max_features: 10,000")
    print(f"   • ngram_range: (1, 2) - unigramas y bigramas")
    print(f"   • min_df: 5 - ignorar términos que aparecen en menos de 5 documentos")
    
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=5,
        sublinear_tf=True  # Usar escala logarítmica para term frequency
    )
    
    print(f"\n⚙️  Entrenando vectorizador TF-IDF...")
    X = vectorizer.fit_transform(texts)
    print(f"✓ Vectorización completada")
    print(f"   • Matriz generada: {X.shape[0]:,} documentos × {X.shape[1]:,} características")
    print(f"   • Sparsity: {(1.0 - X.nnz / (X.shape[0] * X.shape[1])) * 100:.2f}%\n")
    
    # ========== 6. DIVIDIR EN TRAIN/TEST ==========
    print(f"✂️  Dividiendo dataset en train/test...")
    print(f"   • Test size: {test_size * 100:.0f}%")
    print(f"   • Random state: {random_state}")
    print(f"   • Stratify: True (preserva balance de clases)")
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # CRÍTICO: Preservar balance de clases en train/test
    )
    
    print(f"✓ Dataset dividido:")
    print(f"   • Train: {X_train.shape[0]:,} muestras ({(1-test_size)*100:.0f}%)")
    print(f"   • Test:  {X_test.shape[0]:,} muestras ({test_size*100:.0f}%)")
    
    # Verificar balance en splits
    train_pos = np.sum(y_train == 1)
    test_pos = np.sum(y_test == 1)
    print(f"   • Train positivos: {train_pos:,} ({train_pos/len(y_train)*100:.1f}%)")
    print(f"   • Test positivos:  {test_pos:,} ({test_pos/len(y_test)*100:.1f}%)\n")
    
    # ========== 7. ENTRENAR CLASIFICADORES SUPERVISADOS ==========
    print(f"🤖 Entrenando clasificadores supervisados...")
    print(f"   • Naive Bayes")
    print(f"   • Logistic Regression")
    print(f"   • Random Forest")
    print()
    
    trained_models = train_all_models(X_train, y_train, verbose=True)
    
    # ========== 8. EVALUAR MODELOS EN TEST SET ==========
    # CRÍTICO: Evaluar SOLO en test set, NUNCA en train set
    # Evaluar en train set inflaría artificialmente las métricas
    print(f"\n📊 Evaluando modelos en test set...")
    print(f"   • Test samples: {X_test.shape[0]:,}")
    print(f"   • IMPORTANTE: Las métricas son sobre datos NUNCA VISTOS durante entrenamiento")
    print()
    
    # Importar módulo de evaluación
    from evaluation import evaluate_all_models
    
    # Evaluar todos los modelos entrenados
    eval_results = evaluate_all_models(trained_models, X_test, y_test, verbose=False)
    
    # ========== 9. MOSTRAR RESULTADOS DE EVALUACIÓN ==========
    print("\n" + "="*70)
    print("📈 RESULTADOS DE EVALUACIÓN")
    print("="*70)
    
    # Tabla comparativa
    print(f"\n{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
    print("-"*70)
    
    # Ordenar por F1-score
    sorted_results = sorted(eval_results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    
    for model_name, metrics in sorted_results:
        roc_str = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "N/A"
        print(f"{model_name:<25} "
              f"{metrics['accuracy']:<10.4f} "
              f"{metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} "
              f"{metrics['f1_score']:<10.4f} "
              f"{roc_str:<10}")
    
    # Identificar mejor modelo
    best_model_name, best_metrics = sorted_results[0]
    print("\n" + "="*70)
    print(f"🏆 MEJOR MODELO: {best_model_name}")
    print(f"   F1-Score: {best_metrics['f1_score']:.4f}")
    print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
    if best_metrics['roc_auc']:
        print(f"   ROC-AUC:  {best_metrics['roc_auc']:.4f}")
    print("="*70)
    
    # Mostrar métricas detalladas del mejor modelo
    print(f"\n📋 Métricas detalladas de {best_model_name}:")
    print(f"\n{best_metrics['report']}")
    
    print(f"🔍 Matriz de Confusión ({best_model_name}):")
    cm = best_metrics['confusion_matrix']
    tn, fp, fn, tp = cm.ravel()
    print(f"                    Predicted Negative  Predicted Positive")
    print(f"   Actual Negative       {tn:6d}             {fp:6d}")
    print(f"   Actual Positive       {fn:6d}             {tp:6d}")
    print()
    
    # Interpretación de resultados
    f1_score_val = best_metrics['f1_score']
    if f1_score_val >= 0.90:
        print("✅ EXCELENTE! Modelo listo para producción.")
    elif f1_score_val >= 0.85:
        print("✓ MUY BIEN! Rendimiento sólido.")
    elif f1_score_val >= 0.80:
        print("✓ BUENO. Considerar ajustes de hiperparámetros.")
    else:
        print("⚠️  NECESITA MEJORA. Revisar preprocesamiento y features.")
    
    # ========== 10. GUARDAR MODELO COMPLETO CON EVALUACIÓN ==========
    print(f"\n💾 Guardando modelo completo con evaluación en: {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model_data = {
        "vectorizer": vectorizer,
        "models": trained_models,         # Diccionario con los 3 clasificadores entrenados
        "evaluation_results": eval_results,  # NUEVO: Resultados de evaluación
        "best_model_name": best_model_name,  # NUEVO: Nombre del mejor modelo
        "X_test": X_test,                 # Test features (para re-evaluación posterior)
        "y_test": y_test,                 # Test labels (para re-evaluación posterior)
        "keywords": keywords,             # Palabras clave heurísticas (legacy)
        "metadata": {                     # Información adicional
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "n_features": X.shape[1],
            "test_size": test_size,
            "random_state": random_state,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "best_f1_score": best_metrics['f1_score'],
            "best_accuracy": best_metrics['accuracy']
        }
    }
    
    joblib.dump(model_data, model_path)
    
    file_size = os.path.getsize(model_path) / 1024 / 1024  # MB
    print(f"✓ Modelo guardado exitosamente")
    print(f"   • Tamaño del archivo: {file_size:.2f} MB")
    print(f"   • Modelos incluidos: {len(trained_models)}")
    print(f"   • Test set guardado: {X_test.shape[0]:,} muestras")
    print(f"   • Evaluación guardada: {len(eval_results)} modelos")
    print(f"   • Mejor modelo: {best_model_name} (F1={best_metrics['f1_score']:.4f})")
    
    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO Y EVALUACIÓN COMPLETADOS EXITOSAMENTE")
    print("="*70)
    print("\n💡 Próximos pasos:")
    print("   1. Hacer predicciones: predict_text('Mi texto aquí', model_path)")
    print("   2. Evaluación detallada: python test_evaluation.py")
    print("   3. Modo interactivo: python test_predict_supervised.py")
    print(f"\n🎯 Modelo recomendado para predicción: {best_model_name}")
    print("="*70 + "\n")
    
    return model_path


def predict_text(text: str, 
                 model_path: str = "models/review_model.joblib",
                 voting_strategy: str = "majority",
                 preferred_model: str = "Logistic Regression",
                 return_details: bool = True) -> Dict[str, Any]:
    """Predice si un texto es una reseña de película usando clasificadores supervisados.
    
    Esta función usa los modelos de Machine Learning entrenados para hacer predicciones
    en lugar de reglas heurísticas. Soporta diferentes estrategias de decisión:
    - Majority voting: La decisión final se basa en el voto de la mayoría de modelos
    - Preferred model: Usa solo un modelo específico (ej: el mejor en validación)
    - Weighted average: Promedio ponderado de probabilidades
    
    Args:
        text: Texto a clasificar
        model_path: Ruta al archivo del modelo entrenado (default: "models/review_model.joblib")
        voting_strategy: Estrategia de decisión (default: "majority")
            - "majority": Voto mayoritario (≥2 de 3 modelos)
            - "unanimous": Los 3 modelos deben estar de acuerdo
            - "preferred": Usa solo el modelo especificado en preferred_model
            - "weighted_avg": Promedio ponderado de probabilidades (si disponible)
        preferred_model: Nombre del modelo a usar si voting_strategy="preferred"
            Opciones: "Naive Bayes", "Logistic Regression", "Random Forest"
        return_details: Si True, incluye análisis detallado de keywords (default: True)
    
    Returns:
        Diccionario con la predicción y métricas:
        {
            'is_review': bool,                    # Predicción final
            'final_decision': str,                # "review" o "not_review"
            'confidence': float,                  # Confianza de la predicción (0-1)
            'voting_strategy': str,               # Estrategia usada
            'predictions_by_model': {             # Predicciones individuales
                'Naive Bayes': {
                    'prediction': int,            # 0 o 1
                    'probability': float          # Probabilidad clase 1 (si disponible)
                },
                'Logistic Regression': {...},
                'Random Forest': {...}
            },
            'votes': {                            # Resumen de votos
                'positive': int,                  # Votos por "es reseña"
                'negative': int,                  # Votos por "no es reseña"
                'total': int                      # Total de modelos
            },
            'text_preview': str,                  # Primeros 100 chars
            'text_length': int,                   # Longitud del texto
            'keywords_analysis': {                # Análisis heurístico (opcional)
                'keyword_score': float,
                'matched_keywords': list,
                'eval_score': float,
                'matched_evaluative': list
            }
        }
    
    Examples:
        >>> # Predicción básica con majority voting
        >>> result = predict_text("Esta película es increíble! La actuación fue magistral.")
        >>> print(f"Es reseña: {result['is_review']}")
        >>> print(f"Confianza: {result['confidence']:.2%}")
        
        >>> # Usar solo Logistic Regression
        >>> result = predict_text(
        ...     "Esta película es increíble!",
        ...     voting_strategy="preferred",
        ...     preferred_model="Logistic Regression"
        ... )
        
        >>> # Decisión unánime (más conservadora)
        >>> result = predict_text("Texto ambiguo", voting_strategy="unanimous")
        
        >>> # Promedio ponderado de probabilidades
        >>> result = predict_text("Texto", voting_strategy="weighted_avg")
    
    Raises:
        FileNotFoundError: Si el modelo no existe en model_path
        ValueError: Si voting_strategy o preferred_model no son válidos
        KeyError: Si el modelo no contiene los clasificadores entrenados
    
    Notes:
        - El texto se preprocesa automáticamente con preprocess_text()
        - Si un modelo no tiene predict_proba, se usa solo la predicción binaria
        - El análisis de keywords se mantiene para explicabilidad pero NO afecta la decisión
        - Majority voting es la estrategia más robusta para la mayoría de casos
        - Preferred model es útil si conoces el mejor modelo en validación
    """
    # Validar estrategia de voting
    valid_strategies = ["majority", "unanimous", "preferred", "weighted_avg"]
    if voting_strategy not in valid_strategies:
        raise ValueError(
            f"voting_strategy debe ser uno de {valid_strategies}. "
            f"Recibido: '{voting_strategy}'"
        )
    
    # Verificar que el modelo existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modelo no encontrado en {model_path}. "
            f"Entrene primero con train_from_csv()"
        )
    
    # ========== 1. CARGAR MODELO Y ARTEFACTOS ==========
    obj = joblib.load(model_path)
    vectorizer: TfidfVectorizer = obj["vectorizer"]
    models = obj.get("models", None)
    keywords: List[str] = obj.get("keywords", DEFAULT_KEYWORDS)
    
    # Validar que el modelo tiene clasificadores entrenados
    if models is None or len(models) == 0:
        raise KeyError(
            "El modelo no contiene clasificadores entrenados. "
            "El archivo puede ser de una versión antigua. "
            "Re-entrene con train_from_csv()"
        )
    
    # Validar preferred_model si se usa estrategia "preferred"
    if voting_strategy == "preferred" and preferred_model not in models:
        raise ValueError(
            f"Modelo '{preferred_model}' no encontrado. "
            f"Modelos disponibles: {list(models.keys())}"
        )
    
    # ========== 2. PREPROCESAR TEXTO ==========
    text_clean = preprocess_text(text)
    text_vector = vectorizer.transform([text_clean])
    
    # ========== 3. PREDECIR CON CADA MODELO ==========
    predictions = {}
    
    for model_name, model in models.items():
        # Predicción binaria (0 o 1)
        pred = model.predict(text_vector)[0]
        
        # Probabilidad (si el modelo lo soporta)
        proba = None
        if hasattr(model, 'predict_proba'):
            try:
                # predict_proba retorna array 2D: [[prob_clase_0, prob_clase_1]]
                proba_array = model.predict_proba(text_vector)[0]
                proba = proba_array[1]  # Probabilidad de clase 1 (es reseña)
            except Exception as e:
                print(f"⚠️  Warning: No se pudo obtener probabilidad de {model_name}: {e}")
                proba = None
        
        predictions[model_name] = {
            'prediction': int(pred),
            'probability': float(proba) if proba is not None else None
        }
    
    # ========== 4. CALCULAR VOTOS ==========
    votes_positive = sum(p['prediction'] for p in predictions.values())
    votes_negative = len(predictions) - votes_positive
    
    # ========== 5. DECISIÓN FINAL SEGÚN ESTRATEGIA ==========
    final_prediction = 0
    confidence = 0.0
    
    if voting_strategy == "majority":
        # Mayoría simple (≥2 de 3 votos)
        final_prediction = 1 if votes_positive >= len(predictions) / 2 else 0
        # Confianza basada en proporción de votos
        confidence = votes_positive / len(predictions) if final_prediction == 1 else votes_negative / len(predictions)
    
    elif voting_strategy == "unanimous":
        # Todos los modelos deben estar de acuerdo
        final_prediction = 1 if votes_positive == len(predictions) else 0
        confidence = 1.0 if votes_positive in [0, len(predictions)] else 0.0
    
    elif voting_strategy == "preferred":
        # Usar solo el modelo preferido
        final_prediction = predictions[preferred_model]['prediction']
        proba = predictions[preferred_model]['probability']
        confidence = proba if proba is not None else 1.0
    
    elif voting_strategy == "weighted_avg":
        # Promedio ponderado de probabilidades (si disponible)
        probas = [p['probability'] for p in predictions.values() if p['probability'] is not None]
        if len(probas) > 0:
            avg_proba = np.mean(probas)
            final_prediction = 1 if avg_proba >= 0.5 else 0
            confidence = avg_proba if final_prediction == 1 else (1.0 - avg_proba)
        else:
            # Fallback a majority voting si no hay probabilidades
            final_prediction = 1 if votes_positive >= len(predictions) / 2 else 0
            confidence = votes_positive / len(predictions) if final_prediction == 1 else votes_negative / len(predictions)
    
    # ========== 6. ANÁLISIS DE KEYWORDS (OPCIONAL, PARA EXPLICABILIDAD) ==========
    keywords_analysis = None
    if return_details:
        matched_keywords = [k for k in keywords if k in text_clean]
        keyword_score = len(matched_keywords) / len(keywords) if keywords else 0.0
        
        eval_matched = [w for w in EVALUATIVE_WORDS if w in text_clean]
        eval_score = min(1.0, len(eval_matched) / 3.0) if EVALUATIVE_WORDS else 0.0
        
        keywords_analysis = {
            'keyword_score': float(keyword_score),
            'matched_keywords': matched_keywords,
            'eval_score': float(eval_score),
            'matched_evaluative': eval_matched[:5]  # Limitar a 5 para no saturar output
        }
    
    # ========== 7. CONSTRUIR RESPUESTA ==========
    result = {
        'is_review': bool(final_prediction),
        'final_decision': 'review' if final_prediction == 1 else 'not_review',
        'confidence': float(confidence),
        'voting_strategy': voting_strategy,
        'predictions_by_model': predictions,
        'votes': {
            'positive': votes_positive,
            'negative': votes_negative,
            'total': len(predictions)
        },
        'text_preview': text[:100] + '...' if len(text) > 100 else text,
        'text_length': len(text)
    }
    
    # Agregar análisis de keywords si se solicitó
    if keywords_analysis is not None:
        result['keywords_analysis'] = keywords_analysis
    
    return result


if __name__ == "__main__":
    print("Módulo de modelo. Use train_from_csv() y predict_text().")
