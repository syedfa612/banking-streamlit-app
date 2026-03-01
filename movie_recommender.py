"""
Movie Recommendation System
Based on: TMDB 5000 Movies Dataset
Approach: Content-Based Filtering using TF-IDF + Cosine Similarity
"""

import os
import ast
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ✏️  SET YOUR CSV PATH HERE
# ──────────────────────────────────────────────
DATASET_PATH = r'C:\Users\Syed Faizan Pasha\OneDrive\Desktop\Ml Activity\archive\tmdb_5000_movies.csv'

# Example paths:
# DATASET_PATH = r'C:\Users\YourName\Downloads\tmdb_5000_movies.csv'  # Windows
# DATASET_PATH = r'/home/yourname/Downloads/tmdb_5000_movies.csv'     # Linux/Mac

print("=" * 60)
print("  MOVIE RECOMMENDATION SYSTEM")
print("=" * 60)

if not os.path.exists(DATASET_PATH):
    search_locations = [
        os.path.join(os.path.expanduser("~"), "Downloads", "tmdb_5000_movies.csv"),
        os.path.join(os.path.expanduser("~"), "Desktop", "tmdb_5000_movies.csv"),
        os.path.join(os.getcwd(), "tmdb_5000_movies.csv"),
    ]
    found = False
    for loc in search_locations:
        if os.path.exists(loc):
            DATASET_PATH = loc
            print("Found dataset at:", loc)
            found = True
            break
    if not found:
        print("ERROR: Could not find 'tmdb_5000_movies.csv'")
        print("Please update DATASET_PATH at the top of this script.")
        print("Example: DATASET_PATH = r'C:\\Users\\YourName\\Downloads\\tmdb_5000_movies.csv'")
        raise SystemExit(1)

df = pd.read_csv(DATASET_PATH)
print("\nDataset loaded:", df.shape[0], "movies,", df.shape[1], "features\n")

# ──────────────────────────────────────────────
# 2. DATA PREPROCESSING
# ──────────────────────────────────────────────

def parse_json_column(text, key='name', limit=None):
    """Parse JSON-like columns (genres, keywords, etc.)"""
    try:
        items = ast.literal_eval(text)
        names = [item[key].replace(" ", "") for item in items]
        if limit:
            names = names[:limit]
        return names
    except Exception:
        return []

df['genres_list']   = df['genres'].apply(lambda x: parse_json_column(x, limit=3))
df['keywords_list'] = df['keywords'].apply(lambda x: parse_json_column(x, limit=5))
df['overview']      = df['overview'].fillna('')

def build_soup(row):
    genres   = ' '.join(row['genres_list'])
    keywords = ' '.join(row['keywords_list'])
    overview = row['overview']
    return genres + ' ' + genres + ' ' + keywords + ' ' + overview

df['soup'] = df.apply(build_soup, axis=1)
df = df.reset_index(drop=True)
title_to_idx = pd.Series(df.index, index=df['title'].str.lower())

print("Preprocessing complete")

# ──────────────────────────────────────────────
# 3. BUILD SIMILARITY MATRIX
# ──────────────────────────────────────────────

print("Building TF-IDF matrix & cosine similarity...")

tfidf        = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['soup'])
cosine_sim   = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("Similarity matrix shape:", cosine_sim.shape, "\n")

# ──────────────────────────────────────────────
# 4. RECOMMENDATION FUNCTIONS
# ──────────────────────────────────────────────

def get_recommendations(movie_title, n=10, min_votes=100):
    """
    Get top-N similar movies based on content (genres, keywords, overview).

    Args:
        movie_title (str): Movie name (e.g. 'Inception')
        n (int): Number of recommendations to return
        min_votes (int): Filter out movies with fewer votes than this

    Returns:
        DataFrame of recommended movies
    """
    key = movie_title.lower()

    if key not in title_to_idx:
        matches = [t for t in title_to_idx.index if key in t]
        if matches:
            key = matches[0]
            print("Closest match found:", df.loc[title_to_idx[key], 'title'])
        else:
            print("Movie not found:", movie_title)
            print("Try one of these:", list(df['title'].sample(5).values))
            return None

    idx = title_to_idx[key]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    sim_scores  = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:]
    top_indices = [i[0] for i in sim_scores[:n * 3]]

    candidates = df.iloc[top_indices][['title', 'genres_list', 'vote_average', 'vote_count', 'overview']].copy()
    candidates['similarity_score'] = [sim_scores[j][1] for j in range(len(top_indices))]
    candidates = candidates[candidates['vote_count'] >= min_votes].head(n)
    candidates['genres'] = candidates['genres_list'].apply(lambda x: ', '.join(x))

    return candidates[['title', 'genres', 'vote_average', 'similarity_score', 'overview']].reset_index(drop=True)


def get_genre_recommendations(genre, n=10, min_votes=500):
    """
    Get top-rated movies for a specific genre using a weighted rating formula.

    Args:
        genre (str): Genre name (e.g. 'Action', 'Comedy', 'Horror')
        n (int): Number of results
        min_votes (int): Minimum vote count

    Returns:
        DataFrame of top movies in that genre
    """
    genre_lower = genre.lower().replace(" ", "")
    mask        = df['genres_list'].apply(lambda g: genre_lower in [x.lower() for x in g])
    genre_df    = df[mask & (df['vote_count'] >= min_votes)].copy()

    C = df['vote_average'].mean()
    m = min_votes
    genre_df['weighted_rating'] = (
        (genre_df['vote_count'] / (genre_df['vote_count'] + m)) * genre_df['vote_average'] +
        (m / (genre_df['vote_count'] + m)) * C
    )

    top = genre_df.nlargest(n, 'weighted_rating')[['title', 'vote_average', 'vote_count', 'weighted_rating']]
    return top.reset_index(drop=True)


def evaluate_recommendations(movie_title, n=10):
    """Evaluate recommendation quality using intra-list similarity."""
    recs = get_recommendations(movie_title, n=n)
    if recs is None:
        return

    indices = []
    for t in recs['title'].tolist():
        t_lower = t.lower()
        if t_lower in title_to_idx:
            idx = title_to_idx[t_lower]
            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]
            indices.append(idx)

    if len(indices) < 2:
        print("Not enough recommendations to evaluate.")
        return

    sub_matrix = cosine_sim[np.ix_(indices, indices)]
    np.fill_diagonal(sub_matrix, 0)
    avg_sim = sub_matrix.sum() / (len(indices) * (len(indices) - 1))

    print("\nEvaluation for:", movie_title)
    print("  Intra-list Similarity :", round(avg_sim, 4), " (higher = more consistent)")
    print("  Avg Recommended Rating:", round(recs['vote_average'].mean(), 2), "/ 10")
    print("  Recommendations Found :", len(recs))


# ──────────────────────────────────────────────
# 5. DEMO
# ──────────────────────────────────────────────

print("=" * 60)
print("  DEMO: CONTENT-BASED RECOMMENDATIONS")
print("=" * 60)

for movie in ["Avatar", "The Dark Knight", "Inception"]:
    print("\nMovies similar to:", movie)
    print("-" * 50)
    recs = get_recommendations(movie, n=5)
    if recs is not None:
        print(recs[['title', 'genres', 'vote_average', 'similarity_score']].to_string(index=True))

print("\n" + "=" * 60)
print("  DEMO: TOP MOVIES BY GENRE")
print("=" * 60)

for genre in ["Action", "Comedy", "Horror"]:
    print("\nTop", genre, "Movies:")
    print("-" * 50)
    print(get_genre_recommendations(genre, n=5).to_string(index=True))

print("\n" + "=" * 60)
print("  EVALUATION")
print("=" * 60)

for movie in ["Avatar", "The Dark Knight", "Inception"]:
    evaluate_recommendations(movie)

print("\n" + "=" * 60)
print("  System Ready!")
print("  Use: get_recommendations('Movie Title')")
print("  Use: get_genre_recommendations('Genre')")
print("=" * 60)
