# ═══════════════════════════════════════════════════════════════
#  AURUM CINEMA  — Luxury Movie Recommendation System
#  Run:     streamlit run app.py
#  Install: pip install streamlit pandas numpy scikit-learn plotly requests
# ═══════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import ast, requests, urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# ── CONFIG ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Aurum Cinema",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── SESSION STATE ────────────────────────────────────────────
for key, val in {
    'watchlist': [],
    'user_ratings': {},   # {title: {'stars': int, 'review': str}}
    'selected_mood': 'Drama',
    'rec_movies': [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ════════════════════════════════════════════════════════════
#  LUXURY CINEMA CSS  — Art Deco Gold & Black
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;0,700;1,300;1,400&family=EB+Garamond:ital,wght@0,400;0,500;1,400&family=Space+Mono:wght@400;700&display=swap');

:root {
  --black:      #080808;
  --deep:       #0F0F0F;
  --surface:    #161616;
  --surface2:   #1E1E1E;
  --gold:       #C9A84C;
  --gold-light: #E8C96A;
  --gold-dim:   #7A6330;
  --cream:      #F0E8D5;
  --cream-dim:  #B8A898;
  --red:        #8B1A1A;
  --red-bright: #C0392B;
  --border:     #2A2A2A;
  --border-gold:#3D3020;
}

/* ── GLOBAL ── */
html, body, [class*="css"] {
  font-family: 'EB Garamond', Georgia, serif;
  background: var(--black) !important;
  color: var(--cream);
}
.stApp { background: var(--black) !important; }
* { box-sizing: border-box; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--deep); }
::-webkit-scrollbar-thumb { background: var(--gold-dim); border-radius: 3px; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
  background: var(--deep) !important;
  border-right: 1px solid var(--border-gold) !important;
}
[data-testid="stSidebar"] * { color: var(--cream) !important; }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] .stTextInput input {
  background: var(--surface) !important;
  border: 1px solid var(--border-gold) !important;
  color: var(--cream) !important;
  border-radius: 0 !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.72rem !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
  background: var(--surface) !important;
  border: 1px solid var(--border-gold) !important;
  color: var(--cream) !important;
  border-radius: 0 !important;
}
[data-testid="stSidebar"] hr { border-color: var(--border-gold) !important; }
[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] { color: var(--gold) !important; }

/* ── GRAND HEADER ── */
.grand-header {
  background: var(--deep);
  border-bottom: 1px solid var(--border-gold);
  padding: 3rem 4rem 2.5rem;
  position: relative;
  overflow: hidden;
  text-align: center;
}
.grand-header::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--gold), var(--gold-light), var(--gold), transparent);
}
.grand-header::after {
  content: '';
  position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--gold-dim), var(--gold), var(--gold-dim), transparent);
}
.deco-line {
  display: flex; align-items: center; justify-content: center;
  gap: 1rem; margin-bottom: 1rem;
}
.deco-line span {
  font-family: 'Space Mono', monospace;
  font-size: 0.6rem; letter-spacing: 0.35em;
  color: var(--gold); text-transform: uppercase;
}
.deco-line::before, .deco-line::after {
  content: ''; flex: 1; max-width: 120px; height: 1px;
  background: linear-gradient(90deg, transparent, var(--gold));
}
.deco-line::after { background: linear-gradient(90deg, var(--gold), transparent); }

.logo {
  font-family: 'Cormorant Garamond', serif;
  font-size: 5.5rem; font-weight: 300;
  letter-spacing: 0.4em;
  background: linear-gradient(135deg, var(--gold-dim) 0%, var(--gold-light) 40%, var(--gold) 60%, var(--gold-dim) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  line-height: 1; margin: 0; text-transform: uppercase;
}
.logo-sub {
  font-family: 'Space Mono', monospace;
  font-size: 0.62rem; letter-spacing: 0.5em;
  color: var(--gold-dim); text-transform: uppercase;
  margin-top: 0.8rem;
}
.deco-ornament {
  color: var(--gold-dim); font-size: 1.2rem;
  margin: 1rem 0 0; letter-spacing: 1rem;
}

/* ── KPI ROW ── */
.kpi-row {
  display: grid; grid-template-columns: repeat(4, 1fr);
  gap: 1px; background: var(--border-gold);
  border: 1px solid var(--border-gold);
  margin: 2rem 0;
}
.kpi-cell {
  background: var(--surface); padding: 1.5rem 1rem; text-align: center;
  position: relative;
}
.kpi-cell::after {
  content: ''; position: absolute; bottom: 0; left: 20%; right: 20%;
  height: 1px; background: var(--gold-dim); opacity: 0.4;
}
.kpi-val {
  font-family: 'Cormorant Garamond', serif;
  font-size: 2.4rem; font-weight: 600; color: var(--gold-light);
  line-height: 1;
}
.kpi-lbl {
  font-family: 'Space Mono', monospace;
  font-size: 0.55rem; letter-spacing: 0.2em;
  color: var(--gold-dim); text-transform: uppercase;
  margin-top: 0.4rem;
}

/* ── SECTION HEADERS ── */
.sec-header {
  display: flex; align-items: center; gap: 1rem;
  margin: 2.5rem 0 1.5rem;
}
.sec-title {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.8rem; font-weight: 300; font-style: italic;
  color: var(--cream); white-space: nowrap;
}
.sec-line {
  flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--gold-dim), transparent);
}
.sec-ornament { color: var(--gold); font-size: 0.9rem; }

/* ── FILM CARD ── */
.film-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-top: 2px solid var(--gold-dim);
  padding: 1.5rem;
  margin-bottom: 1.2rem;
  position: relative;
  transition: border-top-color 0.3s, box-shadow 0.3s;
}
.film-card:hover {
  border-top-color: var(--gold-light);
  box-shadow: 0 0 30px rgba(201, 168, 76, 0.08);
}
.film-rank {
  font-family: 'Cormorant Garamond', serif;
  font-size: 2.8rem; font-weight: 300;
  color: var(--border-gold); line-height: 1;
  float: right; margin-left: 1rem;
}
.film-title {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.25rem; font-weight: 600;
  color: var(--cream); margin: 0 0 0.3rem;
}
.film-meta {
  font-family: 'Space Mono', monospace;
  font-size: 0.65rem; color: var(--gold-dim);
  letter-spacing: 0.05em; margin-bottom: 0.6rem;
}
.film-overview {
  font-size: 0.88rem; line-height: 1.65;
  color: var(--cream-dim);
  display: -webkit-box; -webkit-line-clamp: 3;
  -webkit-box-orient: vertical; overflow: hidden;
}
.film-scores {
  display: flex; gap: 1.5rem;
  font-family: 'Space Mono', monospace;
  font-size: 0.6rem; color: var(--gold-dim);
  margin-top: 1rem; padding-top: 0.8rem;
  border-top: 1px solid var(--border);
}
.score-val { color: var(--gold); font-weight: 700; }
.score-bar-bg { background: var(--border); height: 2px; margin-top: 6px; }
.score-bar-fill { height: 2px; background: linear-gradient(90deg, var(--gold-dim), var(--gold)); }

/* ── GENRE TAGS ── */
.gtag {
  display: inline-block;
  border: 1px solid var(--border-gold);
  font-family: 'Space Mono', monospace;
  font-size: 0.58rem; letter-spacing: 0.08em;
  padding: 2px 9px; margin: 2px 2px 2px 0;
  color: var(--gold-dim); text-transform: uppercase;
}
.gtag-gold {
  border-color: var(--gold); color: var(--gold);
  background: rgba(201,168,76,0.08);
}

/* ── WHY RECOMMENDED BOX ── */
.why-box {
  background: var(--surface2);
  border-left: 2px solid var(--gold);
  padding: 0.8rem 1rem; margin-top: 0.8rem;
  font-size: 0.8rem;
}
.why-title {
  font-family: 'Space Mono', monospace;
  font-size: 0.58rem; letter-spacing: 0.15em;
  color: var(--gold); text-transform: uppercase;
  margin-bottom: 0.4rem;
}
.why-text { color: var(--cream-dim); line-height: 1.5; font-style: italic; }

/* ── FEATURE FILM (selected movie) ── */
.feature-box {
  background: var(--surface);
  border: 1px solid var(--border-gold);
  padding: 0;
  margin-bottom: 1.5rem;
  position: relative;
  overflow: hidden;
}
.feature-box::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--gold), var(--gold-light), var(--gold), transparent);
}
.feature-inner { padding: 2rem 2.5rem; }
.feature-badge {
  font-family: 'Space Mono', monospace;
  font-size: 0.55rem; letter-spacing: 0.3em;
  color: var(--gold); text-transform: uppercase;
  margin-bottom: 0.8rem;
}
.feature-title {
  font-family: 'Cormorant Garamond', serif;
  font-size: 2.8rem; font-weight: 300; font-style: italic;
  color: var(--gold-light); line-height: 1.1; margin: 0 0 0.5rem;
}
.feature-meta {
  font-family: 'Space Mono', monospace;
  font-size: 0.65rem; color: var(--gold-dim);
  letter-spacing: 0.08em; margin-bottom: 1rem;
}
.feature-overview { font-size: 0.92rem; line-height: 1.7; color: var(--cream-dim); font-style: italic; }

/* ── STAR RATING ── */
.stars { font-size: 1.4rem; letter-spacing: 0.1rem; }
.star-gold { color: var(--gold-light); }
.star-empty { color: var(--border); }

/* ── REVIEW CARD ── */
.review-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 2px solid var(--gold);
  padding: 1.2rem 1.5rem; margin-bottom: 0.8rem;
}
.review-movie {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.1rem; color: var(--gold-light); margin-bottom: 0.3rem;
}
.review-stars { font-size: 1rem; margin-bottom: 0.5rem; }
.review-text { font-style: italic; color: var(--cream-dim); font-size: 0.88rem; line-height: 1.6; }

/* ── WATCHLIST ITEM ── */
.wl-item {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 3px solid var(--gold-dim);
  padding: 1rem 1.5rem; margin-bottom: 0.6rem;
  transition: border-left-color 0.2s;
}
.wl-item.done { border-left-color: var(--gold); }
.wl-title { font-family: 'Cormorant Garamond', serif; font-size: 1.1rem; color: var(--cream); }
.wl-meta  { font-family: 'Space Mono', monospace; font-size: 0.62rem; color: var(--gold-dim); margin-top: 0.2rem; }

/* ── COMPARE ── */
.cmp-panel {
  background: var(--surface);
  border: 1px solid var(--border-gold);
  padding: 1.8rem;
}
.cmp-title {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.5rem; font-style: italic;
  color: var(--gold-light); margin-bottom: 1.2rem;
  border-bottom: 1px solid var(--border-gold); padding-bottom: 0.8rem;
}
.cmp-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 0.6rem 0; border-bottom: 1px solid var(--border);
}
.cmp-lbl { font-family: 'Space Mono', monospace; font-size: 0.6rem; color: var(--gold-dim); letter-spacing: 0.1em; text-transform: uppercase; }
.cmp-val { font-size: 0.9rem; color: var(--cream); }
.cmp-win { color: var(--gold); font-weight: 700; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
  gap: 0; border-bottom: 1px solid var(--border-gold);
  background: var(--deep);
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Space Mono', monospace !important;
  font-size: 0.65rem !important; letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  border-radius: 0 !important;
  background: transparent !important;
  color: var(--gold-dim) !important;
  padding: 1rem 1.8rem !important;
  border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
  color: var(--gold) !important;
  border-bottom-color: var(--gold) !important;
  background: rgba(201,168,76,0.05) !important;
}

/* ── BUTTONS ── */
.stButton > button {
  font-family: 'Space Mono', monospace !important;
  font-size: 0.65rem !important; letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  border-radius: 0 !important;
  background: transparent !important;
  border: 1px solid var(--gold-dim) !important;
  color: var(--gold) !important;
  padding: 0.5rem 1.2rem !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  background: rgba(201,168,76,0.1) !important;
  border-color: var(--gold) !important;
}

/* ── FORM ELEMENTS ── */
.stSelectbox > div > div {
  background: var(--surface) !important;
  border: 1px solid var(--border-gold) !important;
  border-radius: 0 !important;
  color: var(--cream) !important;
}
.stTextArea textarea {
  background: var(--surface) !important;
  border: 1px solid var(--border-gold) !important;
  border-radius: 0 !important;
  color: var(--cream) !important;
  font-family: 'EB Garamond', serif !important;
}
.stMultiSelect > div > div {
  background: var(--surface) !important;
  border: 1px solid var(--border-gold) !important;
  border-radius: 0 !important;
}
[data-baseweb="tag"] { background: var(--border-gold) !important; border-radius: 0 !important; }
.stRadio label, .stCheckbox label { color: var(--cream) !important; font-family: 'EB Garamond', serif !important; }

/* ── EXPANDER ── */
[data-testid="stExpander"] {
  background: var(--surface) !important;
  border: 1px solid var(--border-gold) !important;
  border-radius: 0 !important;
}

/* ── PLOTLY OVERRIDE ── */
.js-plotly-plot { border: 1px solid var(--border-gold); }

/* ── TRAILER BUTTON ── */
.trailer-btn {
  display: inline-block;
  background: var(--red);
  color: white !important;
  font-family: 'Space Mono', monospace;
  font-size: 0.6rem; letter-spacing: 0.12em;
  padding: 5px 14px; text-decoration: none;
  text-transform: uppercase;
  border: none; transition: background 0.2s;
  margin-top: 0.5rem;
}
.trailer-btn:hover { background: var(--red-bright); }

/* ── FOOTER ── */
.footer {
  text-align: center; margin-top: 4rem;
  padding: 2rem; border-top: 1px solid var(--border-gold);
  font-family: 'Space Mono', monospace;
  font-size: 0.58rem; letter-spacing: 0.2em;
  color: var(--gold-dim); text-transform: uppercase;
}

/* ── MISC ── */
.stSpinner > div { color: var(--gold) !important; }
div[data-testid="stHorizontalBlock"] { gap: 1rem; }
.stAlert { border-radius: 0 !important; }
[data-testid="stMarkdownContainer"] h3 { font-family: 'Cormorant Garamond', serif !important; color: var(--gold-light) !important; font-weight: 300 !important; font-style: italic; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  ML ENGINE
# ════════════════════════════════════════════════════════════

@st.cache_data
def load_and_build(path: str):
    df = pd.read_csv(path)

    def pnames(c):
        try:    return ' '.join([d['name'] for d in ast.literal_eval(c)])
        except: return ''
    def plist(c):
        try:    return [d['name'] for d in ast.literal_eval(c)]
        except: return []

    df['genres_str']    = df['genres'].apply(pnames)
    df['genres_list']   = df['genres'].apply(plist)
    df['keywords_str']  = df['keywords'].apply(pnames)
    df['keywords_list'] = df['keywords'].apply(plist)
    df['companies_str'] = df['production_companies'].apply(pnames)
    df['overview']      = df['overview'].fillna('')
    df['tagline']       = df['tagline'].fillna('')
    df['release_year']  = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df['budget']        = df['budget'].replace(0, np.nan)
    df['revenue']       = df['revenue'].replace(0, np.nan)
    df['profit']        = df['revenue'] - df['budget']

    # TF-IDF → SVD latent space
    df['soup'] = (
        df['overview'] + ' ' +
        (df['genres_str']   + ' ') * 4 +
        (df['keywords_str'] + ' ') * 3 +
        df['tagline'] + ' ' + df['companies_str']
    )
    tfidf     = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1, 2))
    tfidf_mat = tfidf.fit_transform(df['soup'])
    svd       = TruncatedSVD(n_components=200, random_state=42)
    svd_mat   = svd.fit_transform(tfidf_mat)      # (n_movies, 200)

    # Popularity weighted rating (IMDb formula)
    C = df['vote_average'].mean()
    m = df['vote_count'].quantile(0.70)
    df['weighted_rating'] = (
        (df['vote_count'] / (df['vote_count'] + m)) * df['vote_average'] +
        (m / (df['vote_count'] + m)) * C
    )
    sc = MinMaxScaler()
    df['pop_score'] = sc.fit_transform(df[['weighted_rating']])

    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    return df.reset_index(drop=True), svd_mat, indices


def hybrid_multi(titles, df, svd_mat, indices, n=10, alpha=0.7, gf=None, yr=None):
    """
    Multi-movie hybrid recommender:
    1. Average SVD vectors of all selected titles
    2. Cosine similarity of averaged vector vs all movies
    3. Hybrid = alpha * content_sim + (1-alpha) * popularity
    """
    valid = [t for t in titles if t in indices]
    if not valid:
        return pd.DataFrame()

    # Average latent vector
    vecs     = np.array([svd_mat[indices[t]] for t in valid])
    avg_vec  = vecs.mean(axis=0, keepdims=True)          # (1, 200)

    c_scores = cosine_similarity(avg_vec, svd_mat)[0]    # (n_movies,)
    p_scores = df['pop_score'].values
    hybrid   = alpha * c_scores + (1 - alpha) * p_scores

    # Exclude input movies
    ex_idx = [indices[t] for t in valid]
    s      = pd.Series(hybrid)
    s.iloc[ex_idx] = -1
    top_idx = s.nlargest(80).index.tolist()

    recs = df.iloc[top_idx].copy()
    recs['content_score'] = c_scores[top_idx]
    recs['hybrid_score']  = hybrid[top_idx]

    if gf and gf != 'All':
        recs = recs[recs['genres_str'].str.contains(gf, case=False, na=False)]
    if yr:
        recs = recs[(recs['release_year'] >= yr[0]) & (recs['release_year'] <= yr[1])]

    return recs.head(n)


def explain_rec(input_titles, rec_row, df, indices):
    """Generate a human-readable explanation for why a movie was recommended."""
    # Collect genres and keywords from input movies
    in_genres   = set()
    in_keywords = set()
    for t in input_titles:
        if t not in indices: continue
        row = df.iloc[indices[t]]
        in_genres.update(g.lower() for g in row['genres_list'])
        in_keywords.update(k.lower() for k in row['keywords_list'][:15])

    rec_genres   = set(g.lower() for g in rec_row['genres_list'])
    rec_keywords = set(k.lower() for k in rec_row['keywords_list'][:15])

    shared_g = sorted(in_genres   & rec_genres,   key=str.capitalize)
    shared_k = sorted(in_keywords & rec_keywords,  key=str.capitalize)

    reasons = []
    if shared_g:
        reasons.append(f"Shares genres: {', '.join(g.title() for g in shared_g[:3])}")
    if shared_k:
        reasons.append(f"Common themes: {', '.join(k.title() for k in shared_k[:4])}")
    cs = int(rec_row['content_score'] * 100)
    hs = int(rec_row['hybrid_score']  * 100)
    reasons.append(f"Content match {cs}% · Hybrid score {hs}%")

    return " · ".join(reasons) if reasons else f"Strong thematic similarity (hybrid score {hs}%)"


def mood_rec(df, genres, min_r=6.5, yr=None, n=12):
    r = df[df['genres_str'].apply(lambda x: any(g.lower() in x.lower() for g in genres))]
    if yr:
        r = r[(r['release_year'] >= yr[0]) & (r['release_year'] <= yr[1])]
    r = r[(r['vote_average'] >= min_r) & (r['vote_count'] >= 100)]
    return r.nlargest(n, 'weighted_rating')


def trailer_url(title):
    q = urllib.parse.quote_plus(f"{title} official trailer")
    return f"https://www.youtube.com/results?search_query={q}"


def stars_html(n, total=5):
    return (
        '<span class="stars">' +
        '<span class="star-gold">' + '★' * n + '</span>' +
        '<span class="star-empty">' + '☆' * (total - n) + '</span>' +
        '</span>'
    )


def add_wl(row, yr):
    t = row['title']
    if not any(w['title'] == t for w in st.session_state.watchlist):
        st.session_state.watchlist.append({
            'title': t, 'year': yr,
            'rating': row['vote_average'],
            'genres': ', '.join(row['genres_list'][:3]),
            'watched': False
        })


def fetch_poster(mid, key):
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{mid}?api_key={key}", timeout=3)
        p = r.json().get('poster_path')
        return f"https://image.tmdb.org/t/p/w300{p}" if p else None
    except: return None


# ── PLOTLY THEME ─────────────────────────────────────────────
PLT = dict(
    paper_bgcolor='#161616', plot_bgcolor='#161616',
    font=dict(family='Space Mono, monospace', color='#B8A898', size=10),
    margin=dict(t=50, b=40, l=50, r=20),
)
GOLD='#C9A84C'; GOLD2='#E8C96A'; DIM='#7A6330'
INK='#0F0F0F'; CREAM='#F0E8D5'


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 1rem;">
      <div style="font-family:'Cormorant Garamond',serif;font-size:1.6rem;letter-spacing:.3em;
           background:linear-gradient(135deg,#7A6330,#E8C96A,#C9A84C);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        AURUM
      </div>
      <div style="font-family:'Space Mono',monospace;font-size:.52rem;letter-spacing:.3em;
           color:#7A6330;text-transform:uppercase;margin-top:.3rem;">
        Cinema Intelligence
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    DATA_PATH = st.text_input(
        "CSV Path",
        value=r"C:\Users\Syed Faizan Pasha\OneDrive\Desktop\Ml Activity\archive\tmdb_5000_movies.csv",
    )
    TMDB_KEY = st.text_input("TMDB API Key (optional)", value="", type="password")
    st.markdown("---")

    st.markdown("**Filters**")
    ALL_G = ['All','Action','Adventure','Animation','Comedy','Crime','Documentary',
             'Drama','Family','Fantasy','History','Horror','Music','Mystery',
             'Romance','Science Fiction','Thriller','War','Western']
    gf     = st.selectbox("Genre", ALL_G)
    yr     = st.slider("Release Year", 1950, 2017, (1980, 2017))
    n_recs = st.slider("Results", 5, 20, 10)
    st.markdown("---")

    st.markdown("**Hybrid Model**")
    alpha = st.slider("Content  ←→  Popularity", 0.0, 1.0, 0.72, 0.02)
    st.caption(f"Content {int(alpha*100)}%  ·  Popularity {int((1-alpha)*100)}%")
    st.markdown("---")

    wl_count = len(st.session_state.watchlist)
    rv_count = len(st.session_state.user_ratings)
    st.markdown(f"📌 **Watchlist** — {wl_count} films")
    st.markdown(f"⭐ **Reviews** — {rv_count} films")
    if wl_count > 0:
        if st.button("Clear Watchlist"):
            st.session_state.watchlist = []
            st.rerun()
    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:.58rem;color:#7A6330;line-height:1.8;">
    MODEL STACK<br>
    ─────────────<br>
    TF-IDF 20k features<br>
    SVD 200 dimensions<br>
    IMDb Weighted Rating<br>
    Multi-vector averaging<br>
    Hybrid α-blending<br>
    ─────────────<br>
    TMDB 5000 Dataset
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  GRAND HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="grand-header">
  <div class="deco-line"><span>Est. 2024 · Recommendation Engine</span></div>
  <div class="logo">Aurum Cinema</div>
  <div class="logo-sub">Advanced Film Intelligence · Hybrid ML · Art of Discovery</div>
  <div class="deco-ornament">◆ &nbsp; ◇ &nbsp; ◆</div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  LOAD
# ════════════════════════════════════════════════════════════
try:
    with st.spinner("Calibrating the projection room..."):
        df, svd_mat, indices = load_and_build(DATA_PATH)
    ok = True
except FileNotFoundError:
    st.error(f"File not found: `{DATA_PATH}` — update the path in the sidebar.")
    ok = False
except Exception as e:
    st.error(f"Error: {e}"); ok = False

if ok:
    ml = sorted(df['title'].dropna().unique().tolist())

    # KPI Row
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-cell">
        <div class="kpi-val">{len(df):,}</div>
        <div class="kpi-lbl">Films in Collection</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-val">{int(df['release_year'].min())}–{int(df['release_year'].max())}</div>
        <div class="kpi-lbl">Years Spanned</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-val">{df['vote_average'].mean():.2f}</div>
        <div class="kpi-lbl">Mean Rating</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-val">200</div>
        <div class="kpi-lbl">SVD Dimensions</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────────
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "✦ Discover", "◈ Analytics", "◎ By Mood",
        "⊞ Compare", "★ Reviews", "◉ Watchlist"
    ])

    # ══════════════════════════════════════════════════════════
    # TAB 1 — DISCOVER (Multi-movie hybrid)
    # ══════════════════════════════════════════════════════════
    with t1:
        st.markdown("""
        <div class="sec-header">
          <span class="sec-ornament">◆</span>
          <span class="sec-title">Select Your Films</span>
          <div class="sec-line"></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-family:'EB Garamond',serif;font-style:italic;color:#7A6330;
             font-size:.9rem;margin-bottom:1rem;">
          Choose up to 3 films you admire — our hybrid model will blend their essences to find your perfect match.
        </div>
        """, unsafe_allow_html=True)

        # Multi-select (up to 3)
        defaults = ["The Dark Knight", "Inception"] if "The Dark Knight" in ml else ml[:2]
        sel_movies = st.multiselect(
            "Choose 1–3 films",
            ml,
            default=[m for m in defaults if m in ml],
            max_selections=3,
            key="multi_sel"
        )

        if sel_movies:
            # Show selected films as feature boxes
            fc = st.columns(len(sel_movies))
            for i, mov in enumerate(sel_movies):
                md  = df[df['title'] == mov].iloc[0]
                myr = int(md['release_year']) if pd.notna(md['release_year']) else 'N/A'
                mrt = int(md['runtime'])      if pd.notna(md['runtime'])      else 'N/A'
                gh  = ''.join([f'<span class="gtag">{g}</span>' for g in md['genres_list'][:4]])
                pu  = fetch_poster(md['id'], TMDB_KEY) if TMDB_KEY else None

                with fc[i]:
                    if pu: st.image(pu, width=120)
                    st.markdown(f"""
                    <div class="feature-box" style="margin-top:0.5rem;">
                      <div class="feature-inner">
                        <div class="feature-badge">✦ Selected Film {i+1}</div>
                        <div class="feature-title">{mov}</div>
                        <div class="feature-meta">{myr} · {mrt} min · ★ {md['vote_average']:.1f}</div>
                        <div style="margin:.5rem 0;">{gh}</div>
                        <div class="feature-overview">{md['overview'][:180]}…</div>
                        <div style="margin-top:1rem;">
                          <a class="trailer-btn" href="{trailer_url(mov)}" target="_blank">▶ Watch Trailer</a>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    in_wl = any(w['title'] == mov for w in st.session_state.watchlist)
                    if not in_wl:
                        if st.button("＋ Watchlist", key=f"wl_sel_{i}"):
                            add_wl(md, myr); st.rerun()
                    else:
                        st.markdown('<span class="gtag gtag-gold">✓ Saved</span>', unsafe_allow_html=True)

            # Run hybrid model
            st.markdown("""
            <div class="sec-header">
              <span class="sec-ornament">◆</span>
              <span class="sec-title">Curated Recommendations</span>
              <div class="sec-line"></div>
            </div>
            """, unsafe_allow_html=True)

            # Model info strip
            movie_str = " + ".join([f'"{m}"' for m in sel_movies])
            st.markdown(f"""
            <div style="background:var(--surface);border:1px solid var(--border-gold);
                 padding:1rem 1.5rem;margin-bottom:1.5rem;
                 display:flex;gap:2rem;align-items:center;flex-wrap:wrap;">
              <div>
                <div style="font-family:'Space Mono',monospace;font-size:.55rem;
                     color:var(--gold-dim);letter-spacing:.15em;margin-bottom:.3rem;">INPUT FILMS</div>
                <div style="font-family:'Cormorant Garamond',serif;font-style:italic;
                     color:var(--cream);font-size:.95rem;">{movie_str}</div>
              </div>
              <div>
                <div style="font-family:'Space Mono',monospace;font-size:.55rem;color:var(--gold-dim);letter-spacing:.15em;margin-bottom:.3rem;">ALGORITHM</div>
                <div style="font-family:'Space Mono',monospace;color:var(--gold);font-size:.72rem;">SVD Multi-Vector Hybrid</div>
              </div>
              <div>
                <div style="font-family:'Space Mono',monospace;font-size:.55rem;color:var(--gold-dim);letter-spacing:.15em;margin-bottom:.3rem;">CONTENT α</div>
                <div style="font-family:'Cormorant Garamond',serif;color:var(--gold-light);font-size:1.4rem;">{int(alpha*100)}%</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Consulting the oracle..."):
                recs = hybrid_multi(sel_movies, df, svd_mat, indices,
                                    n=n_recs, alpha=alpha, gf=gf, yr=yr)

            if recs.empty:
                st.warning("No results found — try relaxing the filters in the sidebar.")
            else:
                cols = st.columns(2)
                for i, (_, row) in enumerate(recs.iterrows()):
                    with cols[i % 2]:
                        gh  = ''.join([f'<span class="gtag">{g}</span>' for g in row['genres_list'][:4]])
                        ry  = int(row['release_year']) if pd.notna(row['release_year']) else 'N/A'
                        cs  = int(row['content_score'] * 100)
                        hs  = int(row['hybrid_score']  * 100)
                        ps  = int(row['pop_score']     * 100)
                        why = explain_rec(sel_movies, row, df, indices)
                        rated = row['title'] in st.session_state.user_ratings
                        user_stars = st.session_state.user_ratings.get(row['title'], {}).get('stars', 0)

                        st.markdown(f"""
                        <div class="film-card">
                          <div class="film-rank">{str(i+1).zfill(2)}</div>
                          <div class="film-title">{row['title']}</div>
                          <div class="film-meta">{ry} &nbsp;·&nbsp; ★ {row['vote_average']:.1f} &nbsp;·&nbsp; {int(row['vote_count']):,} votes</div>
                          <div style="margin:.5rem 0;">{gh}</div>
                          <div class="film-overview">{row['overview']}</div>
                          <div class="why-box">
                            <div class="why-title">✦ Why Recommended</div>
                            <div class="why-text">{why}</div>
                          </div>
                          <div class="film-scores">
                            <span>CONTENT <span class="score-val">{cs}%</span></span>
                            <span>HYBRID <span class="score-val" style="color:var(--gold-light);">{hs}%</span></span>
                            <span>POPULARITY <span class="score-val">{ps}%</span></span>
                            {f'<span style="color:var(--gold);">YOUR RATING: {"★"*user_stars}</span>' if rated else ''}
                          </div>
                          <div class="score-bar-bg">
                            <div class="score-bar-fill" style="width:{max(hs,2)}%"></div>
                          </div>
                          <div style="margin-top:.8rem;">
                            <a class="trailer-btn" href="{trailer_url(row['title'])}" target="_blank">▶ Trailer</a>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                        bc1, bc2 = st.columns(2)
                        al = any(w['title'] == row['title'] for w in st.session_state.watchlist)
                        with bc1:
                            if not al:
                                if st.button("＋ Watchlist", key=f"wl_{i}"):
                                    add_wl(row, ry); st.rerun()
                            else:
                                st.markdown('<span class="gtag gtag-gold">✓ In Watchlist</span>', unsafe_allow_html=True)
                        with bc2:
                            if st.button("★ Rate", key=f"rate_btn_{i}"):
                                st.session_state[f"show_rate_{row['title']}"] = True

                        # Inline rating widget
                        if st.session_state.get(f"show_rate_{row['title']}", False):
                            with st.expander("★ Leave a Rating & Review", expanded=True):
                                stars_sel = st.select_slider(
                                    "Stars", options=[1,2,3,4,5],
                                    value=user_stars if user_stars else 3,
                                    key=f"stars_{row['title']}"
                                )
                                review_txt = st.text_area(
                                    "Your Review (optional)",
                                    value=st.session_state.user_ratings.get(row['title'], {}).get('review', ''),
                                    key=f"rev_{row['title']}", height=80
                                )
                                if st.button("Save Review", key=f"save_rev_{row['title']}"):
                                    st.session_state.user_ratings[row['title']] = {
                                        'stars': stars_sel, 'review': review_txt,
                                        'genres': ', '.join(row['genres_list'][:3])
                                    }
                                    st.session_state[f"show_rate_{row['title']}"] = False
                                    st.success("Review saved!")
                                    st.rerun()

    # ══════════════════════════════════════════════════════════
    # TAB 2 — ANALYTICS DASHBOARD
    # ══════════════════════════════════════════════════════════
    with t2:
        st.markdown("""
        <div class="sec-header">
          <span class="sec-ornament">◈</span>
          <span class="sec-title">Collection Analytics</span>
          <div class="sec-line"></div>
        </div>
        """, unsafe_allow_html=True)

        # Row 1
        r1, r2 = st.columns(2)
        with r1:
            fig = px.histogram(df, x='vote_average', nbins=30,
                               title='RATING DISTRIBUTION',
                               color_discrete_sequence=[GOLD])
            fig.update_layout(**PLT, title_font=dict(family='Space Mono', size=11, color=GOLD))
            fig.add_vline(x=df['vote_average'].mean(), line_dash='dot',
                          line_color=GOLD2, annotation_text=f"μ={df['vote_average'].mean():.2f}",
                          annotation_font_color=GOLD2)
            fig.update_traces(marker_line_width=0)
            fig.update_xaxes(showgrid=False, color='#7A6330')
            fig.update_yaxes(gridcolor='#2A2A2A', color='#7A6330')
            st.plotly_chart(fig, use_container_width=True)

        with r2:
            fig = px.histogram(df, x='vote_count', nbins=50, log_y=True,
                               title='VOTE COUNT — THE LONG TAIL',
                               color_discrete_sequence=[DIM])
            fig.update_layout(**PLT, title_font=dict(family='Space Mono', size=11, color=GOLD))
            fig.update_traces(marker_line_width=0)
            fig.update_xaxes(showgrid=False, color='#7A6330')
            fig.update_yaxes(gridcolor='#2A2A2A', color='#7A6330')
            st.plotly_chart(fig, use_container_width=True)

        # Genre analysis
        ge  = df.explode('genres_list')
        gc  = ge['genres_list'].value_counts().reset_index(); gc.columns=['genre','count']
        gav = ge.groupby('genres_list')['vote_average'].mean().reset_index(); gav.columns=['genre','avg']

        r3, r4 = st.columns(2)
        with r3:
            fig = px.bar(gc.head(15).sort_values('count'), x='count', y='genre',
                         orientation='h', title='FILMS PER GENRE',
                         color='count', color_continuous_scale=[[0,DIM],[1,GOLD]])
            fig.update_layout(**PLT, title_font=dict(family='Space Mono', size=11, color=GOLD),
                              coloraxis_showscale=False)
            fig.update_xaxes(showgrid=True, gridcolor='#2A2A2A', color='#7A6330')
            fig.update_yaxes(showgrid=False, color='#7A6330')
            st.plotly_chart(fig, use_container_width=True)

        with r4:
            fig = px.bar(gav.sort_values('avg', ascending=False), x='genre', y='avg',
                         title='AVG RATING BY GENRE',
                         color='avg', color_continuous_scale=[[0,DIM],[0.5,GOLD],[1,GOLD2]])
            fig.update_layout(**PLT, title_font=dict(family='Space Mono', size=11, color=GOLD),
                              coloraxis_showscale=False)
            fig.update_xaxes(tickangle=45, showgrid=False, color='#7A6330')
            fig.update_yaxes(range=[5, 8], gridcolor='#2A2A2A', color='#7A6330')
            st.plotly_chart(fig, use_container_width=True)

        # Films per year
        yd  = df[df['release_year'] >= 1950]['release_year'].value_counts().sort_index().reset_index()
        yd.columns = ['year', 'count']
        fig = px.area(yd, x='year', y='count', title='FILMS RELEASED PER YEAR',
                      color_discrete_sequence=[GOLD])
        fig.update_layout(**PLT, title_font=dict(family='Space Mono', size=11, color=GOLD))
        fig.update_traces(fill='tozeroy', fillcolor='rgba(201,168,76,0.1)')
        fig.update_xaxes(showgrid=False, color='#7A6330')
        fig.update_yaxes(gridcolor='#2A2A2A', color='#7A6330')
        st.plotly_chart(fig, use_container_width=True)

        # Budget vs Revenue
        fin = df.dropna(subset=['budget','revenue'])
        fig = px.scatter(fin, x='budget', y='revenue', hover_name='title',
                         hover_data={'vote_average': True, 'release_year': True},
                         title='BUDGET vs REVENUE · SIZE = VOTE COUNT',
                         color='vote_average',
                         color_continuous_scale=[[0,INK],[0.5,DIM],[1,GOLD]],
                         size='vote_count', size_max=22)
        mv  = max(fin['budget'].max(), fin['revenue'].max())
        fig.add_shape(type='line', x0=0, y0=0, x1=mv, y1=mv,
                      line=dict(dash='dot', color=GOLD2, width=1))
        fig.update_layout(**PLT, title_font=dict(family='Space Mono', size=11, color=GOLD),
                          coloraxis_colorbar=dict(title='Rating', thickness=10, tickcolor=GOLD))
        fig.update_xaxes(title='Budget ($)', showgrid=True, gridcolor='#2A2A2A', color='#7A6330')
        fig.update_yaxes(title='Revenue ($)', showgrid=True, gridcolor='#2A2A2A', color='#7A6330')
        st.plotly_chart(fig, use_container_width=True)

        # Runtime + Correlation side by side
        r5, r6 = st.columns(2)
        with r5:
            rc  = df['runtime'].dropna(); rc = rc[(rc > 30) & (rc < 300)]
            fig = px.histogram(rc, nbins=60, title='RUNTIME DISTRIBUTION',
                               color_discrete_sequence=[DIM])
            fig.update_layout(**PLT, title_font=dict(family='Space Mono', size=11, color=GOLD))
            fig.add_vline(x=rc.mean(), line_dash='dot', line_color=GOLD,
                          annotation_text=f"μ={rc.mean():.0f}m", annotation_font_color=GOLD)
            fig.update_traces(marker_line_width=0)
            fig.update_xaxes(showgrid=False, title='Minutes', color='#7A6330')
            fig.update_yaxes(gridcolor='#2A2A2A', color='#7A6330')
            st.plotly_chart(fig, use_container_width=True)

        with r6:
            nc   = ['budget','revenue','popularity','runtime','vote_average','vote_count']
            corr = df[nc].corr().round(2)
            fig  = go.Figure(data=go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.columns,
                text=corr.values, texttemplate='%{text}',
                colorscale=[[0, INK],[0.5, DIM],[1, GOLD]],
                zmin=-1, zmax=1, showscale=True
            ))
            fig.update_layout(**PLT, title='CORRELATION HEATMAP',
                              title_font=dict(family='Space Mono', size=11, color=GOLD))
            st.plotly_chart(fig, use_container_width=True)

        # Top 15 table
        st.markdown("""
        <div class="sec-header">
          <span class="sec-ornament">◆</span>
          <span class="sec-title">Top 15 by Weighted Rating</span>
          <div class="sec-line"></div>
        </div>""", unsafe_allow_html=True)
        t15 = df[df['vote_count'] >= df['vote_count'].quantile(0.7)].nlargest(15,'weighted_rating')[
            ['title','release_year','vote_average','vote_count','weighted_rating']].reset_index(drop=True)
        t15.index += 1
        t15.columns = ['Title','Year','Rating','Votes','Weighted Rating']
        t15['Weighted Rating'] = t15['Weighted Rating'].round(3)
        t15['Year'] = t15['Year'].astype('Int64')
        st.dataframe(t15, use_container_width=True, height=460)

    # ══════════════════════════════════════════════════════════
    # TAB 3 — BROWSE BY MOOD
    # ══════════════════════════════════════════════════════════
    with t3:
        st.markdown("""
        <div class="sec-header">
          <span class="sec-ornament">◎</span>
          <span class="sec-title">What Moves You Tonight?</span>
          <div class="sec-line"></div>
        </div>""", unsafe_allow_html=True)

        MOODS = {
            "Laughter":   (["Comedy"],                    "😂"),
            "Suspense":   (["Thriller","Horror"],         "😱"),
            "Romance":    (["Romance","Drama"],           "❤️"),
            "Adventure":  (["Action","Adventure"],        "🚀"),
            "Wonder":     (["Science Fiction","Fantasy"], "🧠"),
            "Melancholy": (["Drama"],                     "🌧️"),
            "Warmth":     (["Family","Animation"],        "👨‍👩‍👧"),
            "Mystery":    (["Crime","Mystery"],           "🔍"),
            "History":    (["History","Documentary"],     "🏛️"),
            "Music":      (["Music"],                     "🎵"),
        }
        mc = st.columns(5)
        for i, (mood, (_, em)) in enumerate(MOODS.items()):
            with mc[i % 5]:
                if st.button(f"{em} {mood}", key=f"mood_{i}", use_container_width=True):
                    st.session_state.selected_mood = mood
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        mm1, mm2 = st.columns(2)
        with mm1: mnr  = st.slider("Minimum Rating", 1.0, 9.0, 6.5, 0.5, key="mr")
        with mm2: myr2 = st.slider("Year Range", 1950, 2017, (1990, 2017), key="my")

        cm = st.session_state.selected_mood
        genres_for_mood, emoji = MOODS.get(cm, (["Drama"], "◆"))
        st.markdown(f"""
        <div class="sec-header">
          <span class="sec-ornament">{emoji}</span>
          <span class="sec-title">{cm}</span>
          <div class="sec-line"></div>
        </div>""", unsafe_allow_html=True)

        mr = mood_rec(df, genres_for_mood, mnr, myr2, 12)
        if mr.empty:
            st.warning("No results. Try lowering the minimum rating.")
        else:
            mcols = st.columns(3)
            for i, (_, row) in enumerate(mr.iterrows()):
                with mcols[i % 3]:
                    gh = ''.join([f'<span class="gtag">{g}</span>' for g in row['genres_list'][:3]])
                    ry = int(row['release_year']) if pd.notna(row['release_year']) else 'N/A'
                    st.markdown(f"""
                    <div class="film-card">
                      <div class="film-title">{row['title']}</div>
                      <div class="film-meta">{ry} · ★ {row['vote_average']:.1f}</div>
                      <div style="margin:.4rem 0;">{gh}</div>
                      <div class="film-overview">{row['overview']}</div>
                      <div style="margin-top:.8rem;">
                        <a class="trailer-btn" href="{trailer_url(row['title'])}" target="_blank">▶ Trailer</a>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    al = any(w['title'] == row['title'] for w in st.session_state.watchlist)
                    if not al:
                        if st.button("＋ Save", key=f"ms{i}"): add_wl(row, ry); st.rerun()
                    else:
                        st.markdown('<span class="gtag gtag-gold">✓ Saved</span>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # TAB 4 — COMPARE
    # ══════════════════════════════════════════════════════════
    with t4:
        st.markdown("""
        <div class="sec-header">
          <span class="sec-ornament">⊞</span>
          <span class="sec-title">Film Comparison</span>
          <div class="sec-line"></div>
        </div>""", unsafe_allow_html=True)

        ca, cb = st.columns(2)
        with ca: ma = st.selectbox("Film A", ml, index=ml.index("The Dark Knight") if "The Dark Knight" in ml else 0, key="ca")
        with cb: mb = st.selectbox("Film B", ml, index=ml.index("Inception") if "Inception" in ml else 1, key="cb")

        if ma and mb and ma != mb:
            a = df[df['title'] == ma].iloc[0]
            b = df[df['title'] == mb].iloc[0]

            # Cosine similarity between the two
            ia  = indices[ma]; ib = indices[mb]
            sab = float(cosine_similarity(svd_mat[ia:ia+1], svd_mat[ib:ib+1])[0][0])
            sp  = int(sab * 100)

            # Shared genres/keywords
            ag = set(a['genres_list']); bg = set(b['genres_list'])
            ak = set(a['keywords_list'][:20]); bk = set(b['keywords_list'][:20])
            shared_g = ag & bg; shared_k = ak & bk

            st.markdown(f"""
            <div style="background:var(--surface);border:1px solid var(--border-gold);
                 padding:2rem;text-align:center;margin-bottom:1.5rem;">
              <div style="font-family:'Space Mono',monospace;font-size:.6rem;
                   letter-spacing:.25em;color:var(--gold-dim);margin-bottom:.5rem;">
                CONTENT SIMILARITY
              </div>
              <div style="font-family:'Cormorant Garamond',serif;font-size:5rem;
                   font-weight:300;color:var(--gold-light);line-height:1;">{sp}%</div>
              <div style="background:var(--border);height:4px;
                   max-width:400px;margin:.8rem auto;border-radius:2px;">
                <div style="width:{max(sp,2)}%;height:4px;
                     background:linear-gradient(90deg,var(--gold-dim),var(--gold-light));border-radius:2px;"></div>
              </div>
              {'<div style="font-family:EB Garamond,serif;font-style:italic;color:var(--gold-dim);font-size:.85rem;margin-top:.5rem;">Shared genres: ' + ', '.join(shared_g) + '</div>' if shared_g else ''}
              {'<div style="font-family:EB Garamond,serif;font-style:italic;color:var(--gold-dim);font-size:.82rem;margin-top:.3rem;">Common themes: ' + ', '.join(list(shared_k)[:5]) + '</div>' if shared_k else ''}
            </div>
            """, unsafe_allow_html=True)

            for col, movie, data, other in [(ca, ma, a, b), (cb, mb, b, a)]:
                with col:
                    gh  = ''.join([f'<span class="gtag">{g}</span>' for g in data['genres_list'][:4]])
                    myr = int(data['release_year']) if pd.notna(data['release_year']) else 'N/A'
                    rt  = int(data['runtime'])      if pd.notna(data['runtime'])      else 0
                    ort = int(other['runtime'])     if pd.notna(other['runtime'])     else 0
                    bud = f"${data['budget']/1e6:.0f}M"   if pd.notna(data['budget'])  else 'N/A'
                    rev = f"${data['revenue']/1e6:.0f}M"  if pd.notna(data['revenue']) else 'N/A'
                    pro = f"${data['profit']/1e6:.0f}M"   if pd.notna(data.get('profit')) else 'N/A'
                    rc  = "cmp-win" if data['vote_average'] >= other['vote_average'] else "cmp-val"
                    vc  = "cmp-win" if data['vote_count']   >= other['vote_count']   else "cmp-val"
                    tc  = "cmp-win" if (rt or 999) <= (ort or 999)                  else "cmp-val"
                    st.markdown(f"""
                    <div class="cmp-panel">
                      <div class="cmp-title">{movie}</div>
                      <div style="margin-bottom:.8rem;">{gh}</div>
                      <div class="cmp-row"><span class="cmp-lbl">Year</span><span class="cmp-val">{myr}</span></div>
                      <div class="cmp-row"><span class="cmp-lbl">Runtime</span><span class="{tc}">{rt} min</span></div>
                      <div class="cmp-row"><span class="cmp-lbl">Rating</span><span class="{rc}">{data['vote_average']:.1f} / 10</span></div>
                      <div class="cmp-row"><span class="cmp-lbl">Votes</span><span class="{vc}">{int(data['vote_count']):,}</span></div>
                      <div class="cmp-row"><span class="cmp-lbl">Budget</span><span class="cmp-val">{bud}</span></div>
                      <div class="cmp-row"><span class="cmp-lbl">Revenue</span><span class="cmp-val">{rev}</span></div>
                      <div class="cmp-row"><span class="cmp-lbl">Profit</span><span class="cmp-val">{pro}</span></div>
                      <div class="cmp-row"><span class="cmp-lbl">Language</span><span class="cmp-val">{data['original_language'].upper()}</span></div>
                      <div style="margin-top:1.2rem;font-style:italic;color:var(--cream-dim);font-size:.85rem;line-height:1.6;">{data['overview'][:220]}…</div>
                      <div style="margin-top:1rem;">
                        <a class="trailer-btn" href="{trailer_url(movie)}" target="_blank">▶ Watch Trailer</a>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    al = any(w['title'] == movie for w in st.session_state.watchlist)
                    st.markdown("<br>", unsafe_allow_html=True)
                    if not al:
                        if st.button(f"＋ Save to Watchlist", key=f"csa_{movie}"): add_wl(data, myr); st.rerun()
                    else:
                        st.markdown('<span class="gtag gtag-gold">✓ In Watchlist</span>', unsafe_allow_html=True)

            st.markdown('<div style="font-family:Space Mono,monospace;font-size:.58rem;color:var(--gold-dim);margin-top:.5rem;letter-spacing:.1em;">GOLD = WINNER IN THAT CATEGORY</div>', unsafe_allow_html=True)

        elif ma == mb:
            st.info("Please select two different films to compare.")

    # ══════════════════════════════════════════════════════════
    # TAB 5 — REVIEWS
    # ══════════════════════════════════════════════════════════
    with t5:
        st.markdown("""
        <div class="sec-header">
          <span class="sec-ornament">★</span>
          <span class="sec-title">Your Reviews</span>
          <div class="sec-line"></div>
        </div>""", unsafe_allow_html=True)

        # Rate any movie
        st.markdown("""<div style="font-family:'Cormorant Garamond',serif;font-style:italic;
                       color:var(--gold-dim);margin-bottom:1rem;font-size:.95rem;">
                       Rate any film in the collection and leave your thoughts.</div>""",
                    unsafe_allow_html=True)

        rc1, rc2, rc3 = st.columns([3, 1, 1])
        with rc1:
            rate_movie = st.selectbox("Choose a Film to Rate", ml, key="rate_sel")
        with rc2:
            rate_stars = st.select_slider("Stars", options=[1,2,3,4,5], value=4, key="rate_stars_main")
        with rc3:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(stars_html(rate_stars), unsafe_allow_html=True)

        rate_text = st.text_area("Your Review", placeholder="Share your thoughts on this film...", height=100, key="rate_text_main")
        if st.button("＋ Submit Review", key="submit_main"):
            if rate_movie:
                md_rate = df[df['title'] == rate_movie].iloc[0]
                st.session_state.user_ratings[rate_movie] = {
                    'stars': rate_stars, 'review': rate_text,
                    'genres': ', '.join(md_rate['genres_list'][:3])
                }
                st.success(f"Review for **{rate_movie}** saved!")
                st.rerun()

        # Show all reviews
        if st.session_state.user_ratings:
            st.markdown("""
            <div class="sec-header" style="margin-top:2rem;">
              <span class="sec-ornament">◆</span>
              <span class="sec-title">Your Collection</span>
              <div class="sec-line"></div>
            </div>""", unsafe_allow_html=True)

            # Stats
            all_stars = [v['stars'] for v in st.session_state.user_ratings.values()]
            avg_s     = sum(all_stars) / len(all_stars)
            rv1, rv2, rv3 = st.columns(3)
            for col, val, lbl in [
                (rv1, len(all_stars), "Films Rated"),
                (rv2, f"{avg_s:.1f}", "Average Stars"),
                (rv3, max(all_stars, key=lambda x: x), "Highest Given"),
            ]:
                with col:
                    st.markdown(f"""
                    <div style="background:var(--surface);border:1px solid var(--border-gold);
                         padding:1.2rem;text-align:center;margin-bottom:1rem;">
                      <div style="font-family:'Cormorant Garamond',serif;font-size:2.2rem;
                           color:var(--gold-light);">{val}</div>
                      <div style="font-family:'Space Mono',monospace;font-size:.55rem;
                           color:var(--gold-dim);letter-spacing:.15em;text-transform:uppercase;">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            # Review cards
            for title, rv in st.session_state.user_ratings.items():
                col_rv, col_del = st.columns([6, 1])
                with col_rv:
                    st.markdown(f"""
                    <div class="review-card">
                      <div class="review-movie">{title}</div>
                      <div class="review-stars">{stars_html(rv['stars'])}</div>
                      {f'<div class="review-text">"{rv["review"]}"</div>' if rv.get('review') else ''}
                      <div style="font-family:Space Mono,monospace;font-size:.58rem;
                           color:var(--gold-dim);margin-top:.6rem;">{rv.get("genres","")}</div>
                    </div>""", unsafe_allow_html=True)
                with col_del:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    if st.button("✕", key=f"del_rv_{title}"):
                        del st.session_state.user_ratings[title]
                        st.rerun()

            # Export
            st.markdown("<br>", unsafe_allow_html=True)
            rv_df = pd.DataFrame([
                {'Title': t, 'Stars': v['stars'], 'Review': v.get('review',''), 'Genres': v.get('genres','')}
                for t, v in st.session_state.user_ratings.items()
            ])
            st.download_button(
                "↓ Export Reviews CSV",
                data=rv_df.to_csv(index=False).encode(),
                file_name="my_reviews.csv", mime="text/csv"
            )
        else:
            st.markdown("""
            <div style="text-align:center;padding:4rem;color:var(--gold-dim);">
              <div style="font-family:'Cormorant Garamond',serif;font-size:3rem;font-weight:300;">No Reviews Yet</div>
              <div style="font-style:italic;margin-top:.5rem;font-size:.9rem;">Rate films from above or from the Discover tab</div>
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # TAB 6 — WATCHLIST
    # ══════════════════════════════════════════════════════════
    with t6:
        st.markdown("""
        <div class="sec-header">
          <span class="sec-ornament">◉</span>
          <span class="sec-title">My Watchlist</span>
          <div class="sec-line"></div>
        </div>""", unsafe_allow_html=True)

        if not st.session_state.watchlist:
            st.markdown("""
            <div style="text-align:center;padding:5rem;">
              <div style="font-family:'Cormorant Garamond',serif;font-size:4rem;
                   font-weight:300;font-style:italic;color:var(--gold-dim);">
                Empty House
              </div>
              <div style="font-style:italic;color:var(--cream-dim);margin-top:.5rem;">
                Save films from the Discover, Mood, or Compare tabs
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            tot = len(st.session_state.watchlist)
            wa  = sum(1 for w in st.session_state.watchlist if w['watched'])
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1px;
                 background:var(--border-gold);border:1px solid var(--border-gold);margin-bottom:1.5rem;">
              <div style="background:var(--surface);padding:1.5rem;text-align:center;">
                <div style="font-family:'Cormorant Garamond',serif;font-size:2.5rem;color:var(--gold-light);">{tot}</div>
                <div style="font-family:'Space Mono',monospace;font-size:.55rem;color:var(--gold-dim);letter-spacing:.15em;">TOTAL FILMS</div>
              </div>
              <div style="background:var(--surface);padding:1.5rem;text-align:center;">
                <div style="font-family:'Cormorant Garamond',serif;font-size:2.5rem;color:var(--gold);">{wa}</div>
                <div style="font-family:'Space Mono',monospace;font-size:.55rem;color:var(--gold-dim);letter-spacing:.15em;">SCREENED</div>
              </div>
              <div style="background:var(--surface);padding:1.5rem;text-align:center;">
                <div style="font-family:'Cormorant Garamond',serif;font-size:2.5rem;color:var(--cream-dim);">{tot-wa}</div>
                <div style="font-family:'Space Mono',monospace;font-size:.55rem;color:var(--gold-dim);letter-spacing:.15em;">AWAITING</div>
              </div>
            </div>""", unsafe_allow_html=True)

            sf = st.radio("Filter:", ["All", "Awaiting", "Screened"], horizontal=True)

            for i, m in enumerate(st.session_state.watchlist):
                if sf == "Screened" and not m['watched']: continue
                if sf == "Awaiting" and     m['watched']: continue
                ci, ca2 = st.columns([6, 2])
                with ci:
                    cls  = "wl-item done" if m['watched'] else "wl-item"
                    stat = "SCREENED ✦"   if m['watched'] else "AWAITING"
                    sc   = "color:var(--gold)" if m['watched'] else "color:var(--gold-dim)"
                    st.markdown(f"""
                    <div class="{cls}">
                      <div style="flex:1;">
                        <div class="wl-title">{m['title']}</div>
                        <div class="wl-meta">{m['year']} &nbsp;·&nbsp; ★ {m['rating']:.1f} &nbsp;·&nbsp; {m['genres']}</div>
                      </div>
                      <span style="font-family:'Space Mono',monospace;font-size:.58rem;
                            letter-spacing:.1em;{sc};">{stat}</span>
                    </div>""", unsafe_allow_html=True)
                with ca2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    lbl = "Mark Screened" if not m['watched'] else "Mark Awaiting"
                    if st.button(lbl, key=f"tog{i}", use_container_width=True):
                        st.session_state.watchlist[i]['watched'] = not m['watched']; st.rerun()
                    if st.button("Remove", key=f"rem{i}", use_container_width=True):
                        st.session_state.watchlist.pop(i); st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                "↓ Export Watchlist CSV",
                data=pd.DataFrame(st.session_state.watchlist).to_csv(index=False).encode(),
                file_name="aurum_watchlist.csv", mime="text/csv"
            )

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  ◆ &nbsp; Aurum Cinema &nbsp; · &nbsp;
  TF-IDF + SVD 200-dim + Hybrid α-Blending &nbsp; · &nbsp;
  Scikit-learn · Plotly · Streamlit &nbsp; · &nbsp;
  TMDB 5000 Collection &nbsp; · &nbsp; ◆
</div>
""", unsafe_allow_html=True)
