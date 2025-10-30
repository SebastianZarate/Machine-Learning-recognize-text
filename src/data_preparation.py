"""
Este módulo es para casos de uso AVANZADOS y EXPERIMENTALES.
El proyecto principal usa directamente 'IMDB Dataset.csv', que ya está
balanceado (25,000 reseñas positivas + 25,000 negativas).

═══════════════════════════════════════════════════════════════════════
PROPÓSITO DE ESTE MÓDULO:
═══════════════════════════════════════════════════════════════════════

Este módulo genera datos sintéticos para un caso de uso DIFERENTE:
  • Clasificación binaria: "Reseña vs No-Reseña"
  • NO para análisis de sentimientos (positivo/negativo)

Los textos sintéticos generados son noticias, recetas, artículos, etc.
Estos NO son reseñas reales de películas.

═══════════════════════════════════════════════════════════════════════
FLUJO PRINCIPAL DEL PROYECTO (NO USA ESTE MÓDULO):
═══════════════════════════════════════════════════════════════════════

1. Cargar datos:
   from preprocessing import load_imdb_dataset
   df = load_imdb_dataset('IMDB Dataset.csv')

2. Entrenar modelos:
   Ver notebooks/03_model_training.ipynb
   O ejecutar: python -m src.train_models

3. Evaluar:
   Ver notebooks/04_evaluation.ipynb

═══════════════════════════════════════════════════════════════════════

Utilities to create a balanced dataset combining the IMDB reviews
and synthetic non-review examples (negatives).

This module provides create_balanced_dataset() which will:
- read the IMDB dataset (path provided)
- take `positive_count` reviews as positive class (label=1)
- synthesize `negative_count` coherent non-review texts (label=0)
- shuffle and save as a CSV with columns ['text','label']

Notes:
- The IMDB file can be large; the function will sample if more rows are
  available than requested.
- Negative examples are generated from category templates and lightly
  varied to produce many examples while keeping them coherent.
"""

from typing import Optional, Tuple
import random
import csv
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def _read_imdb_reviews(imdb_path: str, positive_count: int, random_state: Optional[int] = None) -> pd.Series:
	"""Read reviews from IMDB CSV and return a Series of text reviews.

	Supports typical IMDB dataset with a column named 'review'. If that
	column isn't present, it will attempt to use the first column.
	"""
	if not os.path.exists(imdb_path):
		raise FileNotFoundError(f"IMDB dataset not found at: {imdb_path}")

	# Try reading with pandas; don't load unnecessary columns if possible.
	df = pd.read_csv(imdb_path, encoding='utf-8')

	if 'review' in df.columns:
		texts = df['review'].astype(str)
	else:
		# fallback to the first column
		texts = df.iloc[:, 0].astype(str)

	# If there are more rows than needed, sample to requested size
	if len(texts) > positive_count:
		texts = texts.sample(n=positive_count, random_state=random_state).reset_index(drop=True)
	else:
		texts = texts.reset_index(drop=True)

	return texts


def _generate_paragraph(sentences: list, target_words: int, rng: random.Random) -> str:
	"""Compose a paragraph close to target_words using provided sentence pool."""
	words = 0
	parts = []
	while words < target_words:
		s = rng.choice(sentences)
		# Insert small variations: change a year, a number, or a product name token
		s = s.replace('{year}', str(rng.randint(2000, 2025)))
		s = s.replace('{num}', str(rng.randint(1, 999)))
		s = s.replace('{ingredient}', rng.choice(['flour', 'sugar', 'butter', 'garlic', 'onion']))
		parts.append(s)
		words = sum(len(p.split()) for p in parts)
	paragraph = ' '.join(parts)
	return paragraph


def _generate_negative_texts(n: int, min_words: int = 100, max_words: int = 300, seed: Optional[int] = None) -> list:
	"""Generate `n` synthetic negative texts from several domain templates.

	The function attempts to create coherent multi-sentence paragraphs
	roughly in the word-length range given.
	"""
	rng = random.Random(seed)

	# Pools of sentences per category. Keep them generic but coherent.
	news_sentences = [
		"The government announced a new policy initiative aimed at improving infrastructure and public services.",
		"Analysts said the move could reshape the political landscape ahead of the upcoming elections.",
		"Reports indicate that economic growth slowed slightly this quarter, driven by weaker exports and lower consumer spending.",
		"Local authorities confirmed that emergency services responded quickly to the incident and are conducting an investigation.",
		"The sports team faced a tough season but showed signs of recovery after the mid-season trades and new signings."
	]

	recipe_sentences = [
		"Start by preheating the oven to 180°C and prepare a baking tray with parchment paper.",
		"In a large bowl, combine flour, {ingredient}, and a pinch of salt before whisking in the wet ingredients.",
		"Simmer the sauce gently for 20 minutes until it thickens and the flavors meld together.",
		"Serve warm with fresh herbs on top and a side of steamed vegetables for a balanced meal.",
		"If you prefer a spicier kick, add chopped chili or a dash of hot sauce to taste."
	]

	tech_sentences = [
		"The API accepts JSON payloads and returns a status code indicating success or failure.",
		"Refer to the configuration file for environment-specific variables and ensure credentials are stored securely.",
		"Performance benchmarks show the new implementation reduces latency by approximately {num} percent under typical loads.",
		"Developers should follow the coding standards and add unit tests for edge cases before merging changes.",
		"This section of the documentation explains how to deploy the service using container orchestration tools."
	]

	product_sentences = [
		"This product features a durable aluminum frame and a battery life of up to {num} hours under standard usage.",
		"Customers cited ease of use and a compact design as key advantages in reviews.",
		"The listing includes detailed specifications, warranty information, and care instructions for long-term maintenance.",
		"Available in multiple colors and sizes, the item is suitable for everyday use and travel.",
		"Accessories are sold separately and include replacement parts and protective cases."
	]

	wiki_sentences = [
		"The region has a long history dating back to the early settlements and played a significant role in trade routes.",
		"Scholars have debated the origins of the movement and its influence on subsequent developments in the field.",
		"The article summarizes key events, notable figures, and the broader cultural context surrounding the topic.",
		"Conservation efforts have increased due to concerns about habitat loss and the impact of human activity.",
		"The entry includes references and suggested readings for further research on the subject."
	]

	social_sentences = [
		"Just had the best coffee of the week—totally made my morning commute better! #coffee",
		"Can't believe how fast this year is going; met up with old friends and reminisced for hours.",
		"Anyone has recommendations for a weekend getaway near the coast? Looking for quiet beaches and good food.",
		"Sharing a quick tip: batch your tasks in the morning and you'll get so much more done before lunch.",
		"I ordered a new gadget online and the delivery was surprisingly fast—unboxing coming soon."
	]

	category_pools = [news_sentences, recipe_sentences, tech_sentences, product_sentences, wiki_sentences, social_sentences]

	outputs = []
	for i in range(n):
		# Choose a category for this sample
		pool = rng.choice(category_pools)
		# target words in range
		target_words = rng.randint(min_words, max_words)
		paragraph = _generate_paragraph(pool, target_words, rng)
		# Add a small unique token to help diversify samples
		paragraph = paragraph + f"\n\nRefID: NEG-{i:06d}"
		outputs.append(paragraph)

	return outputs


def create_balanced_dataset(imdb_path: str = 'IMDB Dataset.csv', output_path: str = 'balanced_dataset.csv',
							positive_count: int = 50000, negative_count: int = 50000, random_state: Optional[int] = None) -> None:
	"""Create a balanced dataset with `positive_count` IMDB reviews (label=1)
	and `negative_count` synthetic non-review texts (label=0).

	The resulting CSV will have two columns: ['text', 'label'].
	"""
	# Read positive examples
	positives = _read_imdb_reviews(imdb_path, positive_count, random_state=random_state)

	# Generate negatives
	negatives = _generate_negative_texts(negative_count, min_words=100, max_words=300, seed=random_state)

	pos_df = pd.DataFrame({'text': positives.tolist(), 'label': 1})
	neg_df = pd.DataFrame({'text': negatives, 'label': 0})

	combined = pd.concat([pos_df, neg_df], ignore_index=True)

	# Shuffle
	combined = combined.sample(frac=1, random_state=random_state).reset_index(drop=True)

	# Save
	combined.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)


def split_dataset(X, y, test_size: float = 0.2, random_state: int = 42, stratify: bool = True) -> Tuple:
	"""Split dataset into training and test sets with stratification.
	
	Args:
		X: Feature data (texts or feature matrix)
		y: Labels (0 or 1)
		test_size: Proportion of dataset to include in test split (default 0.2 = 20%)
		random_state: Random seed for reproducibility (default 42)
		stratify: Whether to maintain class proportions in both sets (default True)
	
	Returns:
		Tuple of (X_train, X_test, y_train, y_test)
	
	Notes:
		- The test set should NEVER be used during training or hyperparameter tuning
		- stratify=True ensures both train and test have the same proportion of classes
		- random_state=42 guarantees reproducible splits across runs
	"""
	stratify_param = y if stratify else None
	
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, 
		test_size=test_size, 
		random_state=random_state, 
		stratify=stratify_param
	)
	
	return X_train, X_test, y_train, y_test


if __name__ == '__main__':
	# When run directly, create the balanced dataset in the repository root.
	# Be careful: this will attempt to read the full IMDB CSV; you can override
	# the counts or paths by editing the variables below.
	import argparse

	parser = argparse.ArgumentParser(description='Create a balanced dataset combining IMDB reviews and synthetic negatives.')
	parser.add_argument('--imdb', default='IMDB Dataset.csv')
	parser.add_argument('--out', default='balanced_dataset.csv')
	parser.add_argument('--pos', type=int, default=50000)
	parser.add_argument('--neg', type=int, default=50000)
	parser.add_argument('--seed', type=int, default=None)
	args = parser.parse_args()

	print(f"Reading IMDB from: {args.imdb}")
	create_balanced_dataset(imdb_path=args.imdb, output_path=args.out, positive_count=args.pos, negative_count=args.neg, random_state=args.seed)

