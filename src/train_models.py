"""
Module for training and evaluating supervised machine learning classifiers.

This module implements the core of the workshop: training and comparing
different ML algorithms for text classification (movie reviews).

Supported Models:
    - Naive Bayes (MultinomialNB): Fast, baseline classifier
    - Logistic Regression: Linear model with regularization
    - Random Forest: Ensemble method, handles non-linearity

Author: Machine Learning Workshop
Date: 2025-10-29
"""

import time
import numpy as np
from typing import Dict, Tuple, Any, Optional
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def train_all_models(X_train: np.ndarray, 
                     y_train: np.ndarray,
                     verbose: bool = True) -> Dict[str, Any]:
    """Train multiple supervised classifiers and return all trained models.
    
    Trains three different classification algorithms optimized for text data:
    1. Naive Bayes: Probabilistic baseline, assumes feature independence
    2. Logistic Regression: Linear model with L2 regularization
    3. Random Forest: Ensemble of decision trees, captures non-linear patterns
    
    Args:
        X_train: Training feature matrix (TF-IDF vectors), shape (n_samples, n_features)
                 MUST be non-negative for Naive Bayes compatibility
        y_train: Training labels (0=negative, 1=positive), shape (n_samples,)
        verbose: If True, prints training time for each model (default: True)
    
    Returns:
        Dictionary mapping model name (str) to trained scikit-learn model object:
        {
            'Naive Bayes': MultinomialNB,
            'Logistic Regression': LogisticRegression,
            'Random Forest': RandomForestClassifier
        }
    
    Examples:
        >>> from sklearn.feature_extraction.text import TfidfVectorizer
        >>> vectorizer = TfidfVectorizer(max_features=5000)
        >>> X_train_tfidf = vectorizer.fit_transform(train_texts)
        >>> models = train_all_models(X_train_tfidf, train_labels)
        Naive Bayes trained in 0.15s
        Logistic Regression trained in 2.34s
        Random Forest trained in 45.67s
        
        >>> # Use trained models for prediction
        >>> X_test_tfidf = vectorizer.transform(test_texts)
        >>> nb_predictions = models['Naive Bayes'].predict(X_test_tfidf)
    
    Model Specifications:
        - Naive Bayes:
            * alpha=1.0: Laplace smoothing to handle unseen features
            * Best for: High-dimensional sparse data (TF-IDF)
            * Speed: Fastest (~0.1-0.5s for 80k samples)
            * Assumption: Features are conditionally independent
        
        - Logistic Regression:
            * C=1.0: Inverse regularization strength (lower = more regularization)
            * max_iter=1000: Sufficient for convergence on balanced datasets
            * solver='lbfgs': Efficient for small datasets, handles L2 penalty
            * Speed: Medium (~2-5s for 80k samples)
            * Best for: Interpretable linear relationships
        
        - Random Forest:
            * n_estimators=100: Number of decision trees in ensemble
            * max_depth=20: Limits tree depth to prevent overfitting
            * random_state=42: Reproducible results
            * n_jobs=-1: Use all CPU cores for parallel training
            * Speed: Slowest (~30-60s for 80k samples)
            * Best for: Capturing complex non-linear patterns
    
    Notes:
        - All models are trained on the same data for fair comparison
        - TF-IDF features are non-negative, satisfying Naive Bayes requirements
        - Training time scales with dataset size and feature dimensionality
        - For production, consider grid search for hyperparameter tuning
    
    Performance Tips:
        - Naive Bayes: Increase alpha if overfitting (rare with text)
        - Logistic Regression: Decrease C if overfitting, increase max_iter if not converging
        - Random Forest: Reduce n_estimators or max_depth if too slow
    
    Raises:
        ValueError: If X_train contains negative values (incompatible with Naive Bayes)
        ValueError: If X_train and y_train have different number of samples
    """
    # Validate input dimensions
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"X_train and y_train must have same number of samples. "
            f"Got X_train: {X_train.shape[0]}, y_train: {y_train.shape[0]}"
        )
    
    # Validate non-negative features for Naive Bayes
    if hasattr(X_train, 'data'):  # Sparse matrix
        if np.any(X_train.data < 0):
            raise ValueError("Naive Bayes requires non-negative features. Use TF-IDF or count vectors.")
    else:  # Dense matrix
        if np.any(X_train < 0):
            raise ValueError("Naive Bayes requires non-negative features. Use TF-IDF or count vectors.")
    
    models = {}
    
    # ========== 1. NAIVE BAYES ==========
    if verbose:
        print("\n" + "="*60)
        print("Training Naive Bayes...")
        print("="*60)
    
    start_time = time.time()
    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train, y_train)
    training_time = time.time() - start_time
    models['Naive Bayes'] = nb
    
    if verbose:
        print(f"✓ Naive Bayes trained in {training_time:.2f}s")
        print(f"  - Features: {X_train.shape[1]:,}")
        print(f"  - Training samples: {X_train.shape[0]:,}")
    
    # ========== 2. LOGISTIC REGRESSION ==========
    if verbose:
        print("\n" + "="*60)
        print("Training Logistic Regression...")
        print("="*60)
    
    start_time = time.time()
    lr = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        random_state=42,
        n_jobs=-1  # Use all cores
    )
    lr.fit(X_train, y_train)
    training_time = time.time() - start_time
    models['Logistic Regression'] = lr
    
    if verbose:
        print(f"✓ Logistic Regression trained in {training_time:.2f}s")
        if not lr.n_iter_[0] < 1000:
            print("  ⚠️  Warning: Model may not have converged. Consider increasing max_iter.")
        else:
            print(f"  - Converged in {lr.n_iter_[0]} iterations")
    
    # ========== 3. RANDOM FOREST ==========
    if verbose:
        print("\n" + "="*60)
        print("Training Random Forest...")
        print("="*60)
    
    start_time = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,  # Parallel training
        verbose=0
    )
    rf.fit(X_train, y_train)
    training_time = time.time() - start_time
    models['Random Forest'] = rf
    
    if verbose:
        print(f"✓ Random Forest trained in {training_time:.2f}s")
        print(f"  - Trees: {rf.n_estimators}")
        print(f"  - Max depth: {rf.max_depth}")
    
    if verbose:
        print("\n" + "="*60)
        print(f"✅ All models trained successfully!")
        print(f"   Total models: {len(models)}")
        print("="*60 + "\n")
    
    return models


def evaluate_model(model: Any,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   model_name: str = "Model",
                   verbose: bool = True) -> Dict[str, float]:
    """Evaluate a trained model on test data and return metrics.
    
    Computes standard classification metrics: accuracy, precision, recall, F1-score.
    Also prints detailed classification report and confusion matrix if verbose=True.
    
    Args:
        model: Trained scikit-learn classifier with predict() method
        X_test: Test feature matrix (TF-IDF vectors), shape (n_samples, n_features)
        y_test: True test labels (0=negative, 1=positive), shape (n_samples,)
        model_name: Name of the model for display purposes (default: "Model")
        verbose: If True, prints detailed metrics and confusion matrix (default: True)
    
    Returns:
        Dictionary with evaluation metrics:
        {
            'accuracy': float,    # Overall correctness
            'precision': float,   # Positive predictive value
            'recall': float,      # True positive rate (sensitivity)
            'f1_score': float     # Harmonic mean of precision and recall
        }
    
    Examples:
        >>> metrics = evaluate_model(nb_model, X_test, y_test, "Naive Bayes")
        
        ============================================================
        Evaluating: Naive Bayes
        ============================================================
        Accuracy:  0.8542
        Precision: 0.8621
        Recall:    0.8453
        F1-Score:  0.8536
        
        Classification Report:
                      precision    recall  f1-score   support
        
               0       0.85      0.86      0.85     10000
               1       0.86      0.85      0.86     10000
        
        >>> print(f"F1-Score: {metrics['f1_score']:.4f}")
        F1-Score: 0.8536
    
    Notes:
        - Accuracy: (TP + TN) / Total - Good when classes are balanced
        - Precision: TP / (TP + FP) - Important when false positives are costly
        - Recall: TP / (TP + FN) - Important when false negatives are costly
        - F1-Score: Harmonic mean - Best overall metric for balanced evaluation
    
    Confusion Matrix Interpretation:
        [[TN  FP]     True Negative  | False Positive
         [FN  TP]]    False Negative | True Positive
    """
    if verbose:
        print("\n" + "="*60)
        print(f"Evaluating: {model_name}")
        print("="*60)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0)
    }
    
    if verbose:
        # Print summary metrics
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        
        # Print detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Negative (0)', 'Positive (1)'],
                                   digits=4))
        
        # Print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(f"                Predicted Negative  Predicted Positive")
        print(f"Actual Negative      {cm[0][0]:6d}            {cm[0][1]:6d}")
        print(f"Actual Positive      {cm[1][0]:6d}            {cm[1][1]:6d}")
        print("="*60 + "\n")
    
    return metrics


def evaluate_all_models(models: Dict[str, Any],
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        verbose: bool = True) -> Dict[str, Dict[str, float]]:
    """Evaluate multiple trained models and compare their performance.
    
    Evaluates each model in the dictionary and returns a summary of metrics
    for easy comparison.
    
    Args:
        models: Dictionary mapping model name to trained model object
                (output from train_all_models)
        X_test: Test feature matrix (TF-IDF vectors)
        y_test: True test labels
        verbose: If True, prints detailed metrics for each model (default: True)
    
    Returns:
        Dictionary mapping model name to its evaluation metrics:
        {
            'Naive Bayes': {'accuracy': 0.85, 'precision': 0.86, ...},
            'Logistic Regression': {'accuracy': 0.87, 'precision': 0.88, ...},
            'Random Forest': {'accuracy': 0.84, 'precision': 0.85, ...}
        }
    
    Examples:
        >>> all_metrics = evaluate_all_models(models, X_test, y_test)
        >>> # Print comparison table
        >>> print("\\nModel Comparison:")
        >>> print(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10}")
        >>> for name, metrics in all_metrics.items():
        ...     print(f"{name:<20} {metrics['accuracy']:<10.4f} {metrics['f1_score']:<10.4f}")
        
        Model Comparison:
        Model                Accuracy   F1-Score  
        Naive Bayes          0.8542     0.8536    
        Logistic Regression  0.8723     0.8715    
        Random Forest        0.8401     0.8392    
    """
    all_metrics = {}
    
    for model_name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name, verbose)
        all_metrics[model_name] = metrics
    
    # Print comparison summary
    if verbose:
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-"*60)
        
        for model_name, metrics in all_metrics.items():
            print(f"{model_name:<25} "
                  f"{metrics['accuracy']:<10.4f} "
                  f"{metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} "
                  f"{metrics['f1_score']:<10.4f}")
        
        # Find best model by F1-score
        best_model = max(all_metrics.items(), key=lambda x: x[1]['f1_score'])
        print("\n" + "="*60)
        print(f"🏆 Best Model: {best_model[0]} (F1-Score: {best_model[1]['f1_score']:.4f})")
        print("="*60 + "\n")
    
    return all_metrics


def save_models(models: Dict[str, Any], 
                output_dir: str = "models") -> None:
    """Save all trained models to disk using joblib.
    
    Args:
        models: Dictionary of trained models from train_all_models()
        output_dir: Directory to save models (default: "models")
    
    Examples:
        >>> save_models(models, "models")
        ✓ Saved Naive Bayes to models/naive_bayes.joblib
        ✓ Saved Logistic Regression to models/logistic_regression.joblib
        ✓ Saved Random Forest to models/random_forest.joblib
    """
    import os
    import joblib
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, model in models.items():
        # Create filename from model name (lowercase, replace spaces with underscores)
        filename = model_name.lower().replace(" ", "_") + ".joblib"
        filepath = os.path.join(output_dir, filename)
        
        joblib.dump(model, filepath)
        print(f"✓ Saved {model_name} to {filepath}")


if __name__ == "__main__":
    """
    Example usage of train_models module.
    
    This demonstrates the complete workflow:
    1. Load and prepare data
    2. Create TF-IDF features
    3. Train multiple models
    4. Evaluate and compare models
    5. Save best models
    """
    print("\n" + "="*60)
    print("SUPERVISED MODEL TRAINING - Example Usage")
    print("="*60)
    print("\nThis module provides functions to train and evaluate")
    print("multiple supervised classifiers for text classification.")
    print("\nTypical workflow:")
    print("  1. Load balanced dataset (from data_preparation.py)")
    print("  2. Split into train/test (80/20)")
    print("  3. Create TF-IDF features (from model.py)")
    print("  4. Train models: train_all_models(X_train, y_train)")
    print("  5. Evaluate: evaluate_all_models(models, X_test, y_test)")
    print("  6. Save best models: save_models(models)")
    print("\nSee README.md for complete examples.")
    print("="*60 + "\n")
