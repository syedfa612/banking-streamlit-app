"""
Movie Recommendation System - Load & Recommend
Run this AFTER train_model.py to get instant recommendations without retraining.
"""

import os
import joblib
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# CONFIG — Same MODEL_DIR as train_model.py
# ──────────────────────────────────────────────
MODEL_DIR = r'C:\Users\Syed Faizan Pasha\OneDrive\Desktop\Ml Activity\model'

print("=" * 60)
print("  MOVIE RECOMMENDATION SYSTEM")
print("=" * 60)

# ── Load Saved Model ──
print("\nLoading saved model...")
tfidf        = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
cosine_sim   = joblib.load(os.path.join(MODEL_DIR, 'cosine_sim.pkl'))
tfidf_matrix = joblib.load(os.path.join(MODEL_DIR, 'tfidf_matrix.pkl'))
df           = pd.read_pickle(os.path.join(MODEL_DIR, 'movies_df.pkl'))

title_to_idx = pd.Series(df.index, index=df['title'].str.lower())
print(f"Model loaded! {len(df)} movies ready.\n")

# ──────────────────────────────────────────────
# RECOMMENDATION FUNCTIONS
# ──────────────────────────────────────────────

def get_recommendations(movie_title, n=10, min_votes=100):
    """
    Get top-N similar movies.

    Args:
        movie_title (str): e.g. 'Inception', 'Avatar'
        n (int): number of results
        min_votes (int): minimum vote count filter

    Returns:
        DataFrame of recommendations
    """
    key = movie_title.lower()

    if key not in title_to_idx:
        matches = [t for t in title_to_idx.index if key in t]
        if matches:
            key = matches[0]
            print("Closest match:", df.loc[title_to_idx[key], 'title'])
        else:
            print(f"Movie '{movie_title}' not found.")
            print("Try:", list(df['title'].sample(5).values))
            return None

    idx = title_to_idx[key]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    sim_scores  = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:]
    top_indices = [i[0] for i in sim_scores[:n * 3]]

    candidates = df.iloc[top_indices].copy()
    candidates['similarity_score'] = [sim_scores[j][1] for j in range(len(top_indices))]
    candidates = candidates[candidates['vote_count'] >= min_votes].head(n)
    candidates['genres'] = candidates['genres_list'].apply(lambda x: ', '.join(x))

    return candidates[['title', 'genres', 'vote_average', 'similarity_score']].reset_index(drop=True)


def get_genre_recommendations(genre, n=10, min_votes=500):
    """
    Get top-rated movies for a specific genre.

    Args:
        genre (str): e.g. 'Action', 'Comedy', 'Horror', 'Romance'
        n (int): number of results
        min_votes (int): minimum vote count filter

    Returns:
        DataFrame of top movies in that genre
    """
    genre_lower = genre.lower().replace(" ", "")
    mask        = df['genres_list'].apply(lambda g: genre_lower in [x.lower() for x in g])
    genre_df    = df[mask & (df['vote_count'] >= min_votes)].copy()

    if genre_df.empty:
        print(f"No movies found for genre: {genre}")
        return None

    C = df['vote_average'].mean()
    m = min_votes
    genre_df['weighted_rating'] = (
        (genre_df['vote_count'] / (genre_df['vote_count'] + m)) * genre_df['vote_average'] +
        (m / (genre_df['vote_count'] + m)) * C
    )

    top = genre_df.nlargest(n, 'weighted_rating')[['title', 'vote_average', 'vote_count', 'weighted_rating']]
    return top.reset_index(drop=True)


def search_movie(query):
    """Search for a movie by partial title."""
    query_lower = query.lower()
    matches = df[df['title'].str.lower().str.contains(query_lower, na=False)]
    if matches.empty:
        print(f"No movies found matching '{query}'")
        return None
    return matches[['title', 'vote_average', 'vote_count']].reset_index(drop=True)


# ──────────────────────────────────────────────
# DEMO — Try it out!
# ──────────────────────────────────────────────

print("=" * 60)
print("  EXAMPLE RECOMMENDATIONS")
print("=" * 60)

# Content-based recommendations
for movie in ["Inception", "The Dark Knight", "Avatar"]:
    print(f"\nMovies similar to '{movie}':")
    print("-" * 50)
    recs = get_recommendations(movie, n=5)
    if recs is not None:
        print(recs.to_string(index=True))

# Genre-based recommendations
print("\n" + "=" * 60)
print("  TOP MOVIES BY GENRE")
print("=" * 60)

for genre in ["Action", "Comedy", "Horror"]:
    print(f"\nTop {genre} movies:")
    print("-" * 50)
    result = get_genre_recommendations(genre, n=5)
    if result is not None:
        print(result.to_string(index=True))

print("\n" + "=" * 60)
print("  SYSTEM READY — Use these functions:")
print("  get_recommendations('Movie Title')")
print("  get_genre_recommendations('Genre')")
print("  search_movie('partial title')")
print("=" * 60)
