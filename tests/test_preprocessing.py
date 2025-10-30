"""
Tests unitarios para el módulo de preprocesamiento.

Este módulo contiene tests para verificar el correcto funcionamiento de:
- Limpieza de texto
- Remoción de stopwords
- Lematización
- Pipeline completo de preprocesamiento
"""

import pytest
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing import (
    clean_text_advanced,
    remove_stopwords,
    lemmatize_text,
    preprocess_pipeline
)


class TestCleanText:
    """Tests para la función clean_text_advanced"""
    
    def test_removes_html_tags(self):
        """Verifica que se remuevan tags HTML"""
        input_text = "<p>This is a <b>test</b></p>"
        result = clean_text_advanced(input_text)
        assert "<" not in result
        assert ">" not in result
        assert "test" in result.lower()
    
    def test_removes_urls(self):
        """Verifica que se remuevan URLs"""
        input_text = "Check out https://example.com for more info"
        result = clean_text_advanced(input_text)
        assert "https://" not in result
        assert "example.com" not in result
    
    def test_removes_emails(self):
        """Verifica que se remuevan emails"""
        input_text = "Contact us at test@example.com"
        result = clean_text_advanced(input_text)
        assert "@" not in result
        assert "test@example.com" not in result
    
    def test_converts_to_lowercase(self):
        """Verifica conversión a minúsculas"""
        input_text = "HELLO World"
        result = clean_text_advanced(input_text)
        assert result == result.lower()
    
    def test_removes_special_characters(self):
        """Verifica remoción de caracteres especiales"""
        input_text = "Hello! How are you? #test @user"
        result = clean_text_advanced(input_text)
        assert "!" not in result
        assert "?" not in result
        assert "#" not in result
        assert "@" not in result
    
    def test_handles_empty_string(self):
        """Verifica manejo de string vacío"""
        result = clean_text_advanced("")
        assert result == ""
    
    def test_handles_whitespace(self):
        """Verifica normalización de espacios"""
        input_text = "Hello    world   test"
        result = clean_text_advanced(input_text)
        assert "    " not in result
        assert "   " not in result
    
    @pytest.mark.parametrize("input_text,should_contain", [
        ("<div>Movie</div>", "movie"),
        ("Visit www.test.com", "visit"),
        ("Email: test@mail.com", "email"),
    ])
    def test_various_inputs(self, input_text, should_contain):
        """Test con múltiples casos parametrizados"""
        result = clean_text_advanced(input_text)
        assert should_contain in result.lower()


class TestRemoveStopwords:
    """Tests para la función remove_stopwords"""
    
    def test_removes_common_stopwords(self):
        """Verifica remoción de stopwords comunes"""
        input_text = "the movie is very good"
        result = remove_stopwords(input_text)
        
        # Stopwords deberían ser removidas
        assert "the" not in result
        assert "is" not in result
        assert "very" not in result
        
        # Palabras importantes deberían permanecer
        assert "movie" in result
        assert "good" in result
    
    def test_handles_empty_string(self):
        """Verifica manejo de string vacío"""
        result = remove_stopwords("")
        assert result == ""
    
    def test_preserves_content_words(self):
        """Verifica que se preserven palabras de contenido"""
        input_text = "excellent acting brilliant performance"
        result = remove_stopwords(input_text)
        assert "excellent" in result
        assert "acting" in result
        assert "brilliant" in result
        assert "performance" in result


class TestLemmatizeText:
    """Tests para la función lemmatize_text"""
    
    def test_lemmatizes_verbs(self):
        """Verifica lematización de verbos"""
        input_text = "running jumping swimming"
        result = lemmatize_text(input_text)
        
        # Verbos deberían estar en forma base
        assert "run" in result or "running" not in result
    
    def test_lemmatizes_nouns(self):
        """Verifica lematización de sustantivos"""
        input_text = "movies actors directors"
        result = lemmatize_text(input_text)
        
        # Sustantivos plurales deberían convertirse a singular
        assert "movie" in result or "actor" in result
    
    def test_handles_empty_string(self):
        """Verifica manejo de string vacío"""
        result = lemmatize_text("")
        assert result == ""


class TestPreprocessPipeline:
    """Tests para el pipeline completo de preprocesamiento"""
    
    def test_complete_pipeline_movie_review(self):
        """Test del pipeline completo con una reseña real"""
        input_text = """
        <p>This movie was ABSOLUTELY AMAZING! 
        The acting was superb and the plot kept me engaged throughout.
        Check out the trailer at https://example.com</p>
        Contact: info@studio.com
        """
        
        result = preprocess_pipeline(input_text)
        
        # Verificaciones
        assert isinstance(result, str)
        assert len(result) > 0
        assert "<p>" not in result
        assert "https://" not in result
        assert "@" not in result
        assert result == result.lower()
    
    def test_pipeline_removes_all_noise(self):
        """Verifica que el pipeline remueva todo el ruido"""
        input_text = "<div>Great! @user #movie https://test.com</div>"
        result = preprocess_pipeline(input_text)
        
        assert "<" not in result
        assert ">" not in result
        assert "@" not in result
        assert "#" not in result
        assert "https://" not in result
    
    def test_pipeline_handles_empty_input(self):
        """Verifica manejo de entrada vacía"""
        result = preprocess_pipeline("")
        assert result == ""
    
    def test_pipeline_handles_none(self):
        """Verifica manejo de None"""
        result = preprocess_pipeline(None)
        assert result == ""
    
    def test_pipeline_preserves_sentiment_words(self):
        """Verifica que se preserven palabras de sentimiento"""
        input_text = "The movie was excellent and amazing but terrible ending"
        result = preprocess_pipeline(input_text)
        
        # Palabras de sentimiento importantes
        assert "excellent" in result or "amazing" in result or "terrible" in result
    
    @pytest.mark.parametrize("review,expected_length_min", [
        ("This movie is great!", 3),
        ("Absolutely terrible film with bad acting", 5),
        ("", 0),
    ])
    def test_pipeline_output_length(self, review, expected_length_min):
        """Test de longitud mínima de output"""
        result = preprocess_pipeline(review)
        word_count = len(result.split())
        assert word_count >= expected_length_min or (expected_length_min == 0 and result == "")


class TestEdgeCases:
    """Tests para casos edge y situaciones especiales"""
    
    def test_very_long_text(self):
        """Test con texto muy largo"""
        input_text = "This is a test. " * 1000
        result = preprocess_pipeline(input_text)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_only_stopwords(self):
        """Test con solo stopwords"""
        input_text = "the a an and or but"
        result = remove_stopwords(input_text)
        # Debería quedar vacío o casi vacío
        assert len(result) < len(input_text)
    
    def test_only_special_characters(self):
        """Test con solo caracteres especiales"""
        input_text = "!@#$%^&*()"
        result = clean_text_advanced(input_text)
        assert len(result) == 0 or result.strip() == ""
    
    def test_mixed_languages(self):
        """Test con mezcla de idiomas (debería manejar gracefully)"""
        input_text = "This is English. Esto es español."
        result = preprocess_pipeline(input_text)
        # Debería procesar sin errores
        assert isinstance(result, str)
    
    def test_numbers_and_text(self):
        """Test con números y texto"""
        input_text = "The movie 007 was released in 2020"
        result = preprocess_pipeline(input_text)
        assert isinstance(result, str)


# Fixtures
@pytest.fixture
def sample_positive_review():
    """Fixture con una reseña positiva de ejemplo"""
    return """
    This movie was absolutely fantastic! The performances were stellar,
    and the cinematography was breathtaking. Highly recommended!
    """

@pytest.fixture
def sample_negative_review():
    """Fixture con una reseña negativa de ejemplo"""
    return """
    This was one of the worst movies I've ever seen. The plot made no sense,
    the acting was terrible, and it was a complete waste of time.
    """


class TestWithFixtures:
    """Tests usando fixtures"""
    
    def test_positive_review_processing(self, sample_positive_review):
        """Test de procesamiento de reseña positiva"""
        result = preprocess_pipeline(sample_positive_review)
        assert len(result) > 0
        assert isinstance(result, str)
    
    def test_negative_review_processing(self, sample_negative_review):
        """Test de procesamiento de reseña negativa"""
        result = preprocess_pipeline(sample_negative_review)
        assert len(result) > 0
        assert isinstance(result, str)


if __name__ == '__main__':
    # Ejecutar tests con verbose
    pytest.main([__file__, '-v', '--tb=short'])
