"""Módulo de entrenamiento de modelos."""
import time
import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def split_data(df, test_size=0.2, random_state=42):
    print(chr(10) + chr(128202) + ' Dividiendo datos...')
    X, y = df[chr(39)+'review_clean'+chr(39)], df[chr(39)+'sentiment'+chr(39)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    print(f'+OK Division: {len(X_train)} train, {len(X_test)} test')
    return X_train, X_test, y_train, y_test

def create_vectorizers():
    print(chr(10) + '+Creando vectorizadores...')
    cv = CountVectorizer(max_features=5000, ngram_range=(1,2))
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    return cv, tfidf

def vectorize_data(X_train, X_test, vectorizer):
    print(f'+Vectorizando con {type(vectorizer).__name__}...')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f'+OK Shape: {X_train_vec.shape}')
    return X_train_vec, X_test_vec, vectorizer

def train_naive_bayes(X_train, y_train):
    print(chr(10) + '+Entrenando Naive Bayes...')
    model = MultinomialNB()
    model.fit(X_train, y_train)
    print('+OK Naive Bayes entrenado')
    return model

def train_logistic_regression(X_train, y_train):
    print(chr(10) + '+Entrenando Regresion Logistica...')
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print('+OK Regresion Logistica entrenada')
    return model

def train_random_forest(X_train, y_train):
    print(chr(10) + '+Entrenando Random Forest...')
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=50, n_jobs=-1)
    model.fit(X_train, y_train)
    print('+OK Random Forest entrenado')
    return model

def train_all_models(X_train, y_train):
    print(chr(10) + '='*50)
    print('ENTRENANDO TODOS LOS MODELOS')
    print('='*50)
    models = {}
    models[chr(39)+'naive_bayes'+chr(39)] = train_naive_bayes(X_train, y_train)
    models[chr(39)+'logistic_regression'+chr(39)] = train_logistic_regression(X_train, y_train)
    models[chr(39)+'random_forest'+chr(39)] = train_random_forest(X_train, y_train)
    print(chr(10) + '+OK TODOS LOS MODELOS ENTRENADOS')
    return models

def save_models(models_dict, vectorizer, output_dir=chr(39)+'models/'+chr(39)):
    print(f'+Guardando modelos en {output_dir}...')
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models_dict.items():
        path = os.path.join(output_dir, f'{name}.joblib')
        joblib.dump(model, path)
        print(f'+OK {name} guardado')
    vec_path = os.path.join(output_dir, f'{type(vectorizer).__name__.lower()}.joblib')
    joblib.dump(vectorizer, vec_path)
    print(f'+OK Vectorizador guardado')

if __name__ == chr(39)+'__main__'+chr(39):
    print(chr(39)+'Modulo train_models OK'+chr(39))
