"""
Movie Recommendation System - Model Training & Saving
Run this ONCE to train and save the model.
After this, use load_and_recommend.py to get recommendations instantly.
"""

import os
import ast
import warnings
import pandas as pd
import numpy as np
import joblib
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────
# CONFIG — Update your path here
# ──────────────────────────────────────────────
DATASET_PATH = r'C:\Users\Syed Faizan Pasha\OneDrive\Desktop\Ml Activity\archive\tmdb_5000_movies.csv'
MODEL_DIR    = r'C:\Users\Syed Faizan Pasha\OneDrive\Desktop\Ml Activity\model'

print("=" * 60)
print("  TRAINING MOVIE RECOMMENDATION MODEL")
print("=" * 60)

# ── Load Data ──
df = pd.read_csv(DATASET_PATH)
print(f"\nDataset loaded: {df.shape[0]} movies\n")

# ── Preprocessing ──
def parse_json_column(text, key='name', limit=None):
    try:
        items = ast.literal_eval(text)
        names = [item[key].replace(" ", "") for item in items]
        return names[:limit] if limit else names
    except Exception:
        return []

df['genres_list']   = df['genres'].apply(lambda x: parse_json_column(x, limit=3))
df['keywords_list'] = df['keywords'].apply(lambda x: parse_json_column(x, limit=5))
df['overview']      = df['overview'].fillna('')

def build_soup(row):
    genres   = ' '.join(row['genres_list'])
    keywords = ' '.join(row['keywords_list'])
    return genres + ' ' + genres + ' ' + keywords + ' ' + row['overview']

df['soup'] = df.apply(build_soup, axis=1)
df = df.reset_index(drop=True)

print("Preprocessing complete")

# ── Train TF-IDF ──
print("Training TF-IDF vectorizer...")
tfidf        = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['soup'])

# ── Compute Cosine Similarity ──
print("Computing cosine similarity matrix...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print(f"Similarity matrix shape: {cosine_sim.shape}")

# ── Save Model ──
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(tfidf,        os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
joblib.dump(cosine_sim,   os.path.join(MODEL_DIR, 'cosine_sim.pkl'))
joblib.dump(tfidf_matrix, os.path.join(MODEL_DIR, 'tfidf_matrix.pkl'))

# Save cleaned dataframe
df[['title', 'genres_list', 'vote_average', 'vote_count', 'overview']].to_pickle(
    os.path.join(MODEL_DIR, 'movies_df.pkl')
)

print(f"\nModel saved to: {MODEL_DIR}")
print("Files saved:")
for f in os.listdir(MODEL_DIR):
    size = os.path.getsize(os.path.join(MODEL_DIR, f)) / (1024 * 1024)
    print(f"  {f}  ({size:.1f} MB)")

print("\nTraining complete! Now run load_and_recommend.py")
