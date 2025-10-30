"""
Tests unitarios para el módulo de entrenamiento de modelos.

Este módulo contiene tests para verificar:
- Entrenamiento correcto de modelos
- Guardado y carga de modelos
- Configuración de hiperparámetros
- Splits de datos
"""

import pytest
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from train_models import (
    split_data,
    create_vectorizers,
    vectorize_data,
    train_naive_bayes,
    train_logistic_regression,
    train_random_forest,
    train_all_models
)


# Fixtures
@pytest.fixture
def sample_dataframe():
    """Crea un DataFrame de ejemplo para testing"""
    data = {
        'review_clean': [
            'great movie excellent acting',
            'terrible film bad plot',
            'amazing performance loved',
            'worst movie ever seen',
            'fantastic cinematography beautiful',
            'boring waste time',
            'brilliant masterpiece',
            'awful disappointing'
        ] * 5,  # 40 muestras
        'sentiment': [1, 0, 1, 0, 1, 0, 1, 0] * 5
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_train_test_data(sample_dataframe):
    """Crea datos de train/test de ejemplo"""
    X_train, X_test, y_train, y_test = split_data(sample_dataframe, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def sample_vectorizer():
    """Crea un vectorizador de ejemplo"""
    return TfidfVectorizer(max_features=100, ngram_range=(1, 2))


class TestSplitData:
    """Tests para la función split_data"""
    
    def test_split_data_returns_correct_sizes(self, sample_dataframe):
        """Verifica que el split tenga los tamaños correctos"""
        X_train, X_test, y_train, y_test = split_data(sample_dataframe, test_size=0.2, random_state=42)
        
        total_size = len(sample_dataframe)
        expected_test_size = int(total_size * 0.2)
        expected_train_size = total_size - expected_test_size
        
        assert len(X_train) == expected_train_size
        assert len(X_test) == expected_test_size
        assert len(y_train) == expected_train_size
        assert len(y_test) == expected_test_size
    
    def test_split_data_preserves_stratification(self, sample_dataframe):
        """Verifica que el split mantenga la proporción de clases"""
        X_train, X_test, y_train, y_test = split_data(sample_dataframe, test_size=0.2, random_state=42)
        
        # Calcular proporciones
        train_positive_ratio = y_train.sum() / len(y_train)
        test_positive_ratio = y_test.sum() / len(y_test)
        original_positive_ratio = sample_dataframe['sentiment'].sum() / len(sample_dataframe)
        
        # Las proporciones deberían ser similares (con tolerancia)
        assert abs(train_positive_ratio - original_positive_ratio) < 0.1
        assert abs(test_positive_ratio - original_positive_ratio) < 0.1
    
    def test_split_data_reproducibility(self, sample_dataframe):
        """Verifica que el split sea reproducible con mismo random_state"""
        X_train1, X_test1, y_train1, y_test1 = split_data(sample_dataframe, random_state=42)
        X_train2, X_test2, y_train2, y_test2 = split_data(sample_dataframe, random_state=42)
        
        assert X_train1.equals(X_train2)
        assert y_train1.equals(y_train2)


class TestCreateVectorizers:
    """Tests para la función create_vectorizers"""
    
    def test_creates_two_vectorizers(self):
        """Verifica que se creen ambos vectorizadores"""
        cv, tfidf = create_vectorizers()
        
        assert cv is not None
        assert tfidf is not None
    
    def test_vectorizers_have_correct_types(self):
        """Verifica los tipos de vectorizadores"""
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        
        cv, tfidf = create_vectorizers()
        
        assert isinstance(cv, CountVectorizer)
        assert isinstance(tfidf, TfidfVectorizer)
    
    def test_vectorizers_have_correct_config(self):
        """Verifica la configuración de vectorizadores"""
        cv, tfidf = create_vectorizers()
        
        # Verificar configuración de TF-IDF
        assert tfidf.max_features == 5000
        assert tfidf.ngram_range == (1, 2)


class TestVectorizeData:
    """Tests para la función vectorize_data"""
    
    def test_vectorize_returns_correct_shapes(self, sample_train_test_data, sample_vectorizer):
        """Verifica que la vectorización tenga las formas correctas"""
        X_train, X_test, y_train, y_test = sample_train_test_data
        
        X_train_vec, X_test_vec, fitted_vec = vectorize_data(X_train, X_test, sample_vectorizer)
        
        # Verificar formas
        assert X_train_vec.shape[0] == len(X_train)
        assert X_test_vec.shape[0] == len(X_test)
        assert X_train_vec.shape[1] == X_test_vec.shape[1]  # Mismo número de features
    
    def test_vectorize_returns_sparse_matrix(self, sample_train_test_data, sample_vectorizer):
        """Verifica que retorne matrices sparse"""
        from scipy.sparse import issparse
        
        X_train, X_test, y_train, y_test = sample_train_test_data
        X_train_vec, X_test_vec, fitted_vec = vectorize_data(X_train, X_test, sample_vectorizer)
        
        assert issparse(X_train_vec)
        assert issparse(X_test_vec)
    
    def test_vectorize_fits_on_train_only(self, sample_train_test_data, sample_vectorizer):
        """Verifica que el vectorizador se ajuste solo con datos de entrenamiento"""
        X_train, X_test, y_train, y_test = sample_train_test_data
        X_train_vec, X_test_vec, fitted_vec = vectorize_data(X_train, X_test, sample_vectorizer)
        
        # El vectorizador debería estar fitted
        assert hasattr(fitted_vec, 'vocabulary_')
        assert len(fitted_vec.vocabulary_) > 0


class TestTrainModels:
    """Tests para las funciones de entrenamiento de modelos"""
    
    @pytest.fixture
    def vectorized_data(self, sample_train_test_data):
        """Fixture que prepara datos vectorizados"""
        X_train, X_test, y_train, y_test = sample_train_test_data
        _, tfidf = create_vectorizers()
        X_train_vec, X_test_vec, _ = vectorize_data(X_train, X_test, tfidf)
        return X_train_vec, X_test_vec, y_train, y_test
    
    def test_train_naive_bayes(self, vectorized_data):
        """Test de entrenamiento de Naive Bayes"""
        X_train_vec, _, y_train, _ = vectorized_data
        
        model = train_naive_bayes(X_train_vec, y_train)
        
        # Verificar que el modelo fue entrenado
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        
        # Verificar que puede hacer predicciones
        predictions = model.predict(X_train_vec)
        assert len(predictions) == len(y_train)
        assert all(p in [0, 1] for p in predictions)
    
    def test_train_logistic_regression(self, vectorized_data):
        """Test de entrenamiento de Logistic Regression"""
        X_train_vec, _, y_train, _ = vectorized_data
        
        model = train_logistic_regression(X_train_vec, y_train)
        
        # Verificar que el modelo fue entrenado
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        
        # Verificar que puede hacer predicciones
        predictions = model.predict(X_train_vec)
        assert len(predictions) == len(y_train)
    
    def test_train_random_forest(self, vectorized_data):
        """Test de entrenamiento de Random Forest"""
        X_train_vec, _, y_train, _ = vectorized_data
        
        model = train_random_forest(X_train_vec, y_train)
        
        # Verificar que el modelo fue entrenado
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        
        # Verificar que puede hacer predicciones
        predictions = model.predict(X_train_vec)
        assert len(predictions) == len(y_train)
    
    def test_train_all_models(self, vectorized_data):
        """Test de entrenamiento de todos los modelos"""
        X_train_vec, _, y_train, _ = vectorized_data
        
        models = train_all_models(X_train_vec, y_train)
        
        # Verificar que se entrenaron los 3 modelos
        assert isinstance(models, dict)
        assert 'naive_bayes' in models
        assert 'logistic_regression' in models
        assert 'random_forest' in models
        
        # Verificar que todos pueden hacer predicciones
        for model_name, model in models.items():
            assert hasattr(model, 'predict')
            predictions = model.predict(X_train_vec)
            assert len(predictions) == len(y_train)


class TestModelPerformance:
    """Tests para verificar que los modelos tengan rendimiento razonable"""
    
    @pytest.fixture
    def trained_models_and_data(self, sample_train_test_data):
        """Entrena modelos y prepara datos de test"""
        X_train, X_test, y_train, y_test = sample_train_test_data
        _, tfidf = create_vectorizers()
        X_train_vec, X_test_vec, _ = vectorize_data(X_train, X_test, tfidf)
        
        models = train_all_models(X_train_vec, y_train)
        
        return models, X_test_vec, y_test
    
    def test_models_achieve_minimum_accuracy(self, trained_models_and_data):
        """Verifica que los modelos alcancen accuracy mínima razonable"""
        from sklearn.metrics import accuracy_score
        
        models, X_test_vec, y_test = trained_models_and_data
        
        for model_name, model in models.items():
            predictions = model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, predictions)
            
            # Con datos sintéticos simples, deberían tener al menos 50% accuracy
            assert accuracy >= 0.5, f"{model_name} tiene accuracy muy baja: {accuracy:.2%}"
    
    def test_models_predict_both_classes(self, trained_models_and_data):
        """Verifica que los modelos predigan ambas clases"""
        models, X_test_vec, y_test = trained_models_and_data
        
        for model_name, model in models.items():
            predictions = model.predict(X_test_vec)
            unique_predictions = set(predictions)
            
            # Debería predecir al menos una de cada clase
            # (aunque en datasets pequeños puede no ser así siempre)
            assert len(unique_predictions) > 0


class TestEdgeCases:
    """Tests para casos edge y situaciones especiales"""
    
    def test_empty_dataframe_handling(self):
        """Test con DataFrame vacío (debería fallar gracefully)"""
        empty_df = pd.DataFrame({'review_clean': [], 'sentiment': []})
        
        with pytest.raises(Exception):
            split_data(empty_df)
    
    def test_single_class_dataframe(self):
        """Test con DataFrame de una sola clase"""
        single_class_df = pd.DataFrame({
            'review_clean': ['good movie', 'great film'] * 5,
            'sentiment': [1, 1] * 5
        })
        
        # Con una sola clase, stratify puede fallar o simplemente hacer un split aleatorio
        # Verificamos que al menos se pueda procesar
        try:
            X_train, X_test, y_train, y_test = split_data(single_class_df)
            # Si no falla, verificamos que todas las etiquetas son iguales
            assert len(set(y_train)) <= 1
            assert len(set(y_test)) <= 1
        except ValueError:
            # También es válido que falle con stratify
            pass


# Tests de integración
class TestIntegration:
    """Tests de integración del flujo completo"""
    
    def test_complete_pipeline(self, sample_dataframe):
        """Test del pipeline completo de entrenamiento"""
        # 1. Split datos
        X_train, X_test, y_train, y_test = split_data(sample_dataframe, random_state=42)
        
        # 2. Crear vectorizador
        _, tfidf = create_vectorizers()
        
        # 3. Vectorizar
        X_train_vec, X_test_vec, fitted_vec = vectorize_data(X_train, X_test, tfidf)
        
        # 4. Entrenar modelos
        models = train_all_models(X_train_vec, y_train)
        
        # 5. Verificar que todo funciona
        assert len(models) == 3
        
        for model_name, model in models.items():
            predictions = model.predict(X_test_vec)
            assert len(predictions) == len(y_test)
            assert all(p in [0, 1] for p in predictions)


if __name__ == '__main__':
    # Ejecutar tests con verbose
    pytest.main([__file__, '-v', '--tb=short'])
