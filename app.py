# ============================================================
#  🎬 CineMatch PRO — Advanced Movie Recommendation System
#  Run:     streamlit run app.py
#  Install: pip install streamlit pandas scikit-learn plotly requests
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import ast, requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="CineMatch PRO", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

if 'watchlist'     not in st.session_state: st.session_state.watchlist     = []
if 'selected_mood' not in st.session_state: st.session_state.selected_mood = "Comedy"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Lora:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap');
:root{--cream:#F2EDE3;--ink:#111111;--red:#C8102E;--gold:#B8962E;--gray:#6B6B6B;--border:#D8D0C4;--white:#FFFFFF;}
html,body,[class*="css"]{font-family:'Lora',Georgia,serif;background:var(--cream)!important;color:var(--ink);}
.stApp{background:var(--cream)!important;}
[data-testid="stSidebar"]{background:var(--ink)!important;}
[data-testid="stSidebar"] *{color:var(--cream)!important;}
[data-testid="stSidebar"] .stTextInput input{background:#222!important;border-color:#444!important;color:var(--cream)!important;}
[data-testid="stSidebar"] .stSelectbox>div>div{background:#222!important;border-color:#444!important;color:var(--cream)!important;}
.masthead{background:var(--ink);padding:2.5rem 3rem 2rem;position:relative;overflow:hidden;}
.masthead::after{content:'';position:absolute;bottom:0;left:0;right:0;height:4px;background:var(--red);}
.masthead-eyebrow{font-family:'JetBrains Mono',monospace;font-size:.7rem;letter-spacing:.3em;color:var(--red);text-transform:uppercase;margin-bottom:.5rem;}
.masthead-title{font-family:'Bebas Neue',sans-serif;font-size:5rem;line-height:.9;color:var(--cream);margin:0;letter-spacing:.05em;}
.masthead-sub{font-family:'Lora',serif;font-style:italic;color:#888;font-size:1rem;margin-top:.8rem;}
.masthead-tag{display:inline-block;background:var(--red);color:white;font-family:'JetBrains Mono',monospace;font-size:.65rem;letter-spacing:.15em;padding:3px 10px;margin-right:8px;margin-top:1rem;}
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--ink);margin-bottom:2.5rem;}
.kpi-cell{background:var(--cream);padding:1.5rem;text-align:center;}
.kpi-value{font-family:'Bebas Neue',sans-serif;font-size:2.8rem;color:var(--ink);line-height:1;}
.kpi-label{font-family:'JetBrains Mono',monospace;font-size:.62rem;color:var(--gray);letter-spacing:.15em;text-transform:uppercase;margin-top:.3rem;}
.section-title{font-family:'Bebas Neue',sans-serif;font-size:2.2rem;letter-spacing:.05em;color:var(--ink);margin:2rem 0 .2rem;line-height:1;}
.section-label{font-family:'JetBrains Mono',monospace;font-size:.65rem;letter-spacing:.2em;color:var(--red);text-transform:uppercase;margin-bottom:.2rem;}
.section-rule{border:none;border-top:2px solid var(--ink);margin:0 0 1.5rem;}
.film-card{background:var(--white);border:1px solid var(--border);border-top:3px solid var(--ink);padding:1.4rem;margin-bottom:1.2rem;position:relative;transition:border-top-color .2s,transform .2s;}
.film-card:hover{border-top-color:var(--red);transform:translateY(-2px);}
.film-number{font-family:'Bebas Neue',sans-serif;font-size:3rem;color:var(--border);line-height:1;float:right;margin-left:1rem;}
.film-title{font-family:'Bebas Neue',sans-serif;font-size:1.3rem;letter-spacing:.03em;color:var(--ink);margin:0 0 .3rem;}
.film-meta{font-family:'JetBrains Mono',monospace;font-size:.7rem;color:var(--gray);margin-bottom:.5rem;}
.film-overview{font-size:.83rem;line-height:1.6;color:#444;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;}
.film-score{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:var(--gray);margin-top:.8rem;padding-top:.6rem;border-top:1px solid var(--border);display:flex;justify-content:space-between;}
.tag{display:inline-block;border:1px solid var(--border);font-family:'JetBrains Mono',monospace;font-size:.62rem;letter-spacing:.05em;padding:2px 8px;margin:2px 2px 2px 0;color:var(--gray);}
.tag-red{border-color:var(--red);color:var(--red);}
.feature-film{background:var(--ink);color:var(--cream);padding:2rem 2.5rem;margin-bottom:1.5rem;position:relative;}
.feature-film::before{content:'NOW SHOWING';font-family:'JetBrains Mono',monospace;font-size:.6rem;letter-spacing:.3em;color:var(--red);display:block;margin-bottom:1rem;}
.feature-title{font-family:'Bebas Neue',sans-serif;font-size:3rem;color:var(--cream);line-height:1;margin:0 0 .5rem;}
.feature-meta{font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#aaa;margin-bottom:1rem;}
.feature-overview{font-style:italic;color:#ccc;font-size:.9rem;line-height:1.6;}
.wl-item{background:var(--white);border-left:4px solid var(--border);padding:1rem 1.2rem;margin-bottom:.6rem;display:flex;align-items:center;gap:1rem;}
.wl-item.watched{border-left-color:var(--red);}
.wl-title{font-family:'Bebas Neue',sans-serif;font-size:1.1rem;}
.wl-meta{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:var(--gray);}
.compare-panel{background:var(--white);border:1px solid var(--border);padding:1.5rem;}
.compare-head{font-family:'Bebas Neue',sans-serif;font-size:1.6rem;margin-bottom:1rem;border-bottom:2px solid var(--ink);padding-bottom:.5rem;}
.compare-row{display:flex;justify-content:space-between;padding:.55rem 0;border-bottom:1px solid var(--border);font-size:.82rem;}
.compare-lbl{color:var(--gray);font-family:'JetBrains Mono',monospace;font-size:.68rem;letter-spacing:.05em;}
.compare-val{font-weight:600;}
.compare-win{color:var(--red);font-weight:700;}
.score-bar-bg{background:var(--border);height:3px;border-radius:2px;margin-top:4px;}
.score-bar-hybrid{height:3px;border-radius:2px;background:var(--gold);}
.stButton>button{font-family:'JetBrains Mono',monospace!important;font-size:.72rem!important;letter-spacing:.05em!important;border-radius:0!important;border:1px solid var(--border)!important;background:var(--white)!important;color:var(--ink)!important;}
.stButton>button:hover{border-color:var(--red)!important;color:var(--red)!important;}
.stSelectbox>div>div{background:var(--white)!important;border-radius:0!important;border-color:var(--border)!important;}
.stTabs [data-baseweb="tab-list"]{gap:0;border-bottom:2px solid var(--ink);}
.stTabs [data-baseweb="tab"]{font-family:'JetBrains Mono',monospace!important;font-size:.72rem!important;letter-spacing:.1em!important;border-radius:0!important;background:transparent!important;color:var(--gray)!important;padding:.8rem 1.5rem!important;}
.stTabs [aria-selected="true"]{background:var(--ink)!important;color:var(--cream)!important;}
.footer{margin-top:4rem;padding:2rem 0;border-top:2px solid var(--ink);text-align:center;font-family:'JetBrains Mono',monospace;font-size:.65rem;letter-spacing:.1em;color:var(--gray);text-transform:uppercase;}
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(paper_bgcolor='#FAFAF7',plot_bgcolor='#FAFAF7',font=dict(family='JetBrains Mono, monospace',color='#111111',size=11),margin=dict(t=50,b=40,l=40,r=20))
RED='#C8102E'; GOLD='#B8962E'; INK='#111111'; CREAM='#F2EDE3'

@st.cache_data
def load_and_build(path):
    df = pd.read_csv(path)
    def pnames(c):
        try: return ' '.join([d['name'] for d in ast.literal_eval(c)])
        except: return ''
    def plist(c):
        try: return [d['name'] for d in ast.literal_eval(c)]
        except: return []
    df['genres_str']  = df['genres'].apply(pnames)
    df['genres_list'] = df['genres'].apply(plist)
    df['keywords_str']= df['keywords'].apply(pnames)
    df['companies_str']= df['production_companies'].apply(pnames)
    df['overview']    = df['overview'].fillna('')
    df['tagline']     = df['tagline'].fillna('')
    df['release_year']= pd.to_datetime(df['release_date'],errors='coerce').dt.year
    df['budget']      = df['budget'].replace(0,np.nan)
    df['revenue']     = df['revenue'].replace(0,np.nan)
    df['profit']      = df['revenue'] - df['budget']
    df['soup'] = df['overview']+' '+(df['genres_str']+' ')*4+(df['keywords_str']+' ')*3+df['tagline']+' '+df['companies_str']
    tfidf = TfidfVectorizer(stop_words='english',max_features=20000,ngram_range=(1,2))
    tm    = tfidf.fit_transform(df['soup'])
    svd   = TruncatedSVD(n_components=200,random_state=42)
    sm    = svd.fit_transform(tm)
    csim  = cosine_similarity(sm,sm)
    C = df['vote_average'].mean(); m = df['vote_count'].quantile(0.70)
    df['weighted_rating'] = (df['vote_count']/(df['vote_count']+m))*df['vote_average']+(m/(df['vote_count']+m))*C
    sc = MinMaxScaler()
    df['pop_score'] = sc.fit_transform(df[['weighted_rating']])
    idx = pd.Series(df.index,index=df['title']).drop_duplicates()
    return df.reset_index(drop=True), csim, sm, idx

def hybrid_rec(title,df,csim,sm,idx,n=10,alpha=0.7,gf=None,yr=None):
    i   = idx[title]
    cs  = csim[i]; ps = df['pop_score'].values
    hy  = alpha*cs+(1-alpha)*ps
    s   = pd.Series(hy); s[i]=-1
    ti  = s.nlargest(60).index.tolist()
    r   = df.iloc[ti].copy()
    r['content_score'] = cs[ti]; r['hybrid_score'] = hy[ti]
    if gf and gf!='All': r = r[r['genres_str'].str.contains(gf,case=False,na=False)]
    if yr: r = r[(r['release_year']>=yr[0])&(r['release_year']<=yr[1])]
    return r.head(n)

def mood_rec(df,genres,min_r=6.0,yr=None,n=12):
    r = df.copy()
    r = r[r['genres_str'].apply(lambda x: any(g.lower() in x.lower() for g in genres))]
    if yr: r = r[(r['release_year']>=yr[0])&(r['release_year']<=yr[1])]
    r = r[(r['vote_average']>=min_r)&(r['vote_count']>=100)]
    return r.nlargest(n,'weighted_rating')

def fetch_poster(mid,key):
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{mid}?api_key={key}",timeout=3)
        p = r.json().get('poster_path')
        return f"https://image.tmdb.org/t/p/w300{p}" if p else None
    except: return None

def add_wl(row,yr):
    t = row['title']
    if not any(w['title']==t for w in st.session_state.watchlist):
        st.session_state.watchlist.append({'title':t,'year':yr,'rating':row['vote_average'],'genres':','.join(row['genres_list'][:3]),'watched':False})
        return True
    return False

# SIDEBAR
with st.sidebar:
    st.markdown("### CINEMATCH PRO"); st.markdown("---")
    DATA_PATH = st.text_input("CSV PATH", value=r"C:\Users\Syed Faizan Pasha\OneDrive\Desktop\Ml Activity\archive\tmdb_5000_movies.csv")
    TMDB_KEY  = st.text_input("TMDB API KEY (optional)", value="", type="password")
    st.markdown("---"); st.markdown("**FILTERS**")
    ALL_G  = ['All','Action','Adventure','Animation','Comedy','Crime','Documentary','Drama','Family','Fantasy','History','Horror','Music','Mystery','Romance','Science Fiction','Thriller','War','Western']
    gf     = st.selectbox("Genre", ALL_G)
    yr     = st.slider("Year Range",1950,2017,(1985,2017))
    n_recs = st.slider("Results",5,20,10)
    st.markdown("---"); st.markdown("**HYBRID MODEL**")
    alpha = st.slider("Content ← → Popularity",0.0,1.0,0.7,0.05)
    st.caption(f"Content: {int(alpha*100)}%  |  Popularity: {int((1-alpha)*100)}%")
    st.markdown("---")
    wl = len(st.session_state.watchlist)
    st.markdown(f"**WATCHLIST** — {wl} films")
    if wl>0 and st.button("Clear"): st.session_state.watchlist=[]; st.rerun()
    st.markdown("---")
    st.markdown("**ALGORITHM**\n\nTF-IDF (20k features)\n+ TruncatedSVD (200 dims)\n+ Weighted Popularity\n= Hybrid Score")

# MASTHEAD
st.markdown("""
<div class="masthead">
  <div class="masthead-eyebrow">Advanced Recommendation System</div>
  <div class="masthead-title">CINEMATCH<br>PRO</div>
  <div class="masthead-sub">Hybrid ML · SVD · Content-Based + Popularity · Analytics Dashboard</div>
  <div><span class="masthead-tag">TF-IDF</span><span class="masthead-tag">SVD 200-DIM</span><span class="masthead-tag">HYBRID MODEL</span><span class="masthead-tag">TMDB 5000</span></div>
</div>
""", unsafe_allow_html=True)

try:
    with st.spinner("Building hybrid model (SVD + TF-IDF)..."):
        df, csim, sm, idx = load_and_build(DATA_PATH)
    ok = True
except FileNotFoundError:
    st.error(f"File not found: {DATA_PATH}"); ok = False
except Exception as e:
    st.error(f"Error: {e}"); ok = False

if ok:
    ml = sorted(df['title'].dropna().unique().tolist())
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-cell"><div class="kpi-value">{len(df):,}</div><div class="kpi-label">Films</div></div>
      <div class="kpi-cell"><div class="kpi-value">{int(df['release_year'].min())}–{int(df['release_year'].max())}</div><div class="kpi-label">Years Covered</div></div>
      <div class="kpi-cell"><div class="kpi-value">{df['vote_average'].mean():.1f}</div><div class="kpi-label">Avg Rating</div></div>
      <div class="kpi-cell"><div class="kpi-value">200</div><div class="kpi-label">SVD Dimensions</div></div>
    </div>""", unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs(["RECOMMEND","ANALYTICS","COMPARE","BROWSE MOOD","WATCHLIST"])

    # ── TAB 1: RECOMMEND ──────────────────────────────────────
    with t1:
        st.markdown('<div class="section-label">Select a Film</div>', unsafe_allow_html=True)
        di = ml.index("The Dark Knight") if "The Dark Knight" in ml else 0
        sel = st.selectbox("",ml,index=di,label_visibility="collapsed",key="rs")
        if sel:
            md  = df[df['title']==sel].iloc[0]
            myr = int(md['release_year']) if pd.notna(md['release_year']) else 'N/A'
            mrt = int(md['runtime'])      if pd.notna(md['runtime'])      else 'N/A'
            th  = ''.join([f'<span class="tag">{g}</span>' for g in md['genres_list'][:5]])
            pu  = fetch_poster(md['id'],TMDB_KEY) if TMDB_KEY else None
            ci, cf = st.columns([1,4])
            with ci:
                if pu: st.image(pu,width=140)
                else: st.markdown('<div style="width:140px;height:210px;background:#222;display:flex;align-items:center;justify-content:center;font-size:3rem;">🎞️</div>',unsafe_allow_html=True)
            with cf:
                st.markdown(f'<div class="feature-film"><div class="feature-title">{sel}</div><div class="feature-meta">{myr} · {mrt} min · ★{md["vote_average"]:.1f} · {int(md["vote_count"]):,} votes</div><div style="margin-bottom:.8rem;">{th}</div><div class="feature-overview">{md["overview"]}</div></div>',unsafe_allow_html=True)
            iw = any(w['title']==sel for w in st.session_state.watchlist)
            if not iw:
                if st.button("+ Add to Watchlist",key="as"): add_wl(md,myr); st.rerun()
            else: st.markdown('<span class="tag tag-red">✓ IN WATCHLIST</span>',unsafe_allow_html=True)

            st.markdown(f'<div style="background:var(--white);border:1px solid var(--border);padding:1rem 1.5rem;margin:1rem 0 1.5rem;display:flex;gap:2rem;"><div><div class="section-label">Model</div><div style="font-family:Bebas Neue,sans-serif;font-size:1.1rem;">Hybrid SVD</div></div><div><div class="section-label">Content</div><div style="font-family:Bebas Neue,sans-serif;font-size:1.1rem;">{int(alpha*100)}%</div></div><div><div class="section-label">Popularity</div><div style="font-family:Bebas Neue,sans-serif;font-size:1.1rem;">{int((1-alpha)*100)}%</div></div><div><div class="section-label">SVD Dims</div><div style="font-family:Bebas Neue,sans-serif;font-size:1.1rem;">200</div></div><div style="flex:1;font-size:.78rem;color:var(--gray);font-style:italic;">Adjust the slider in the sidebar to tune the hybrid model</div></div>',unsafe_allow_html=True)

            with st.spinner("Running hybrid model..."):
                recs = hybrid_rec(sel,df,csim,sm,idx,n=n_recs,alpha=alpha,gf=gf,yr=yr)
            if len(recs)==0: st.warning("No results. Relax the sidebar filters.")
            else:
                st.markdown(f'<div class="section-label">Recommendations for</div><div class="section-title">{sel}</div><hr class="section-rule">',unsafe_allow_html=True)
                cols = st.columns(2)
                for i,(_, row) in enumerate(recs.iterrows()):
                    with cols[i%2]:
                        gh = ''.join([f'<span class="tag">{g}</span>' for g in row['genres_list'][:4]])
                        ry = int(row['release_year']) if pd.notna(row['release_year']) else 'N/A'
                        cs = int(row['content_score']*100); hs = int(row['hybrid_score']*100)
                        st.markdown(f'<div class="film-card"><div class="film-number">{str(i+1).zfill(2)}</div><div class="film-title">{row["title"]}</div><div class="film-meta">{ry} · ★{row["vote_average"]:.1f} · {int(row["vote_count"]):,} votes</div><div style="margin:.5rem 0;">{gh}</div><div class="film-overview">{row["overview"]}</div><div class="film-score"><span>CONTENT {cs}%</span><span style="color:var(--gold);">HYBRID {hs}%</span><span>POP {int(row["pop_score"]*100)}%</span></div><div class="score-bar-bg"><div class="score-bar-hybrid" style="width:{max(hs,2)}%"></div></div></div>',unsafe_allow_html=True)
                        al = any(w['title']==row['title'] for w in st.session_state.watchlist)
                        if not al:
                            if st.button("+ Save",key=f"sv{i}"): add_wl(row,ry); st.rerun()
                        else: st.markdown('<span class="tag tag-red" style="font-size:.6rem;">✓ SAVED</span>',unsafe_allow_html=True)

    # ── TAB 2: ANALYTICS ──────────────────────────────────────
    with t2:
        st.markdown('<div class="section-title">Analytics Dashboard</div><hr class="section-rule">',unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            fig=px.histogram(df,x='vote_average',nbins=30,title='RATING DISTRIBUTION',color_discrete_sequence=[RED])
            fig.update_layout(**PLOT_LAYOUT,title_font=dict(family='Bebas Neue',size=18))
            fig.add_vline(x=df['vote_average'].mean(),line_dash='dash',line_color=INK,annotation_text=f"Mean {df['vote_average'].mean():.2f}",annotation_font_color=INK)
            fig.update_traces(marker_line_width=0); fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor='#E0D8CC')
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig=px.histogram(df,x='vote_count',nbins=50,log_y=True,title='VOTE COUNT (LOG SCALE) — LONG TAIL',color_discrete_sequence=[INK])
            fig.update_layout(**PLOT_LAYOUT,title_font=dict(family='Bebas Neue',size=18))
            fig.update_traces(marker_line_width=0); fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor='#E0D8CC')
            st.plotly_chart(fig,use_container_width=True)

        ge = df.explode('genres_list')
        gc = ge['genres_list'].value_counts().reset_index(); gc.columns=['genre','count']
        ga = ge.groupby('genres_list')['vote_average'].mean().reset_index(); ga.columns=['genre','avg']
        c3,c4=st.columns(2)
        with c3:
            fig=px.bar(gc.head(15).sort_values('count'),x='count',y='genre',orientation='h',title='FILMS PER GENRE',color_discrete_sequence=[RED])
            fig.update_layout(**PLOT_LAYOUT,title_font=dict(family='Bebas Neue',size=18)); fig.update_xaxes(showgrid=True,gridcolor='#E0D8CC'); fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig,use_container_width=True)
        with c4:
            fig=px.bar(ga.sort_values('avg',ascending=False),x='genre',y='avg',title='AVG RATING BY GENRE',color='avg',color_continuous_scale=[[0,CREAM],[.5,GOLD],[1,RED]])
            fig.update_layout(**PLOT_LAYOUT,title_font=dict(family='Bebas Neue',size=18),coloraxis_showscale=False)
            fig.update_xaxes(tickangle=45,showgrid=False); fig.update_yaxes(range=[5,8],gridcolor='#E0D8CC')
            st.plotly_chart(fig,use_container_width=True)

        yd = df[df['release_year']>=1950]['release_year'].value_counts().sort_index().reset_index(); yd.columns=['year','count']
        fig=px.area(yd,x='year',y='count',title='FILMS RELEASED PER YEAR',color_discrete_sequence=[RED])
        fig.update_layout(**PLOT_LAYOUT,title_font=dict(family='Bebas Neue',size=18))
        fig.update_traces(fill='tozeroy',fillcolor='rgba(200,16,46,0.1)'); fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor='#E0D8CC')
        st.plotly_chart(fig,use_container_width=True)

        fin=df.dropna(subset=['budget','revenue'])
        fig=px.scatter(fin,x='budget',y='revenue',hover_name='title',hover_data={'vote_average':True,'release_year':True},title='BUDGET vs REVENUE',color='vote_average',color_continuous_scale=[[0,CREAM],[.5,GOLD],[1,RED]],size='vote_count',size_max=20)
        mv=max(fin['budget'].max(),fin['revenue'].max())
        fig.add_shape(type='line',x0=0,y0=0,x1=mv,y1=mv,line=dict(dash='dash',color=INK,width=1))
        fig.add_annotation(x=mv*.8,y=mv*.85,text='Break-even',showarrow=False,font=dict(size=10,color=INK))
        fig.update_layout(**PLOT_LAYOUT,title_font=dict(family='Bebas Neue',size=18))
        fig.update_xaxes(showgrid=True,gridcolor='#E0D8CC',title='Budget ($)'); fig.update_yaxes(showgrid=True,gridcolor='#E0D8CC',title='Revenue ($)')
        st.plotly_chart(fig,use_container_width=True)

        c5,c6=st.columns([2,1])
        with c5:
            st.markdown('<div class="section-label">Top 15 by Weighted Rating</div>',unsafe_allow_html=True)
            t15=df[df['vote_count']>=df['vote_count'].quantile(0.7)].nlargest(15,'weighted_rating')[['title','release_year','vote_average','vote_count','weighted_rating']].reset_index(drop=True)
            t15.index+=1; t15.columns=['Title','Year','Rating','Votes','Weighted Rating']
            t15['Weighted Rating']=t15['Weighted Rating'].round(3); t15['Year']=t15['Year'].astype('Int64')
            st.dataframe(t15,use_container_width=True,height=430)
        with c6:
            lc=df['original_language'].value_counts().head(8).reset_index(); lc.columns=['lang','count']; lc['lang']=lc['lang'].str.upper()
            fig=px.pie(lc,values='count',names='lang',title='LANGUAGES',color_discrete_sequence=[RED,INK,GOLD,'#888','#bbb','#ddd','#444','#ccc'])
            fig.update_layout(**PLOT_LAYOUT,title_font=dict(family='Bebas Neue',size=18))
            fig.update_traces(textposition='inside',textinfo='label+percent',textfont_family='JetBrains Mono')
            st.plotly_chart(fig,use_container_width=True)

        rc=df['runtime'].dropna(); rc=rc[(rc>30)&(rc<300)]
        fig=px.histogram(rc,nbins=60,title='RUNTIME DISTRIBUTION',color_discrete_sequence=[GOLD])
        fig.update_layout(**PLOT_LAYOUT,title_font=dict(family='Bebas Neue',size=18))
        fig.add_vline(x=rc.mean(),line_dash='dash',line_color=RED,annotation_text=f"Mean {rc.mean():.0f} min",annotation_font_color=RED)
        fig.update_traces(marker_line_width=0); fig.update_xaxes(showgrid=False,title='Runtime (min)'); fig.update_yaxes(gridcolor='#E0D8CC')
        st.plotly_chart(fig,use_container_width=True)

        nc=['budget','revenue','popularity','runtime','vote_average','vote_count']
        corr=df[nc].corr().round(2)
        fig=go.Figure(data=go.Heatmap(z=corr.values,x=corr.columns,y=corr.columns,text=corr.values,texttemplate='%{text}',colorscale=[[0,CREAM],[.5,'white'],[1,RED]],zmin=-1,zmax=1,showscale=True))
        fig.update_layout(**PLOT_LAYOUT,title='FEATURE CORRELATION HEATMAP',title_font=dict(family='Bebas Neue',size=18))
        st.plotly_chart(fig,use_container_width=True)

    # ── TAB 3: COMPARE ────────────────────────────────────────
    with t3:
        st.markdown('<div class="section-title">Compare Two Films</div><hr class="section-rule">',unsafe_allow_html=True)
        ca,cb=st.columns(2)
        with ca: ma=st.selectbox("Film A",ml,index=ml.index("The Dark Knight") if "The Dark Knight" in ml else 0,key="ca")
        with cb: mb=st.selectbox("Film B",ml,index=ml.index("Inception") if "Inception" in ml else 1,key="cb")
        if ma and mb and ma!=mb:
            a=df[df['title']==ma].iloc[0]; b=df[df['title']==mb].iloc[0]
            sab=csim[idx[ma]][idx[mb]]; sp=int(sab*100)
            st.markdown(f'<div style="text-align:center;margin:1rem 0;padding:1.5rem;background:#111;color:#F2EDE3;"><div style="font-family:JetBrains Mono,monospace;font-size:.7rem;letter-spacing:.2em;color:#C8102E;">SIMILARITY SCORE</div><div style="font-family:Bebas Neue,sans-serif;font-size:4rem;line-height:1;">{sp}%</div><div style="background:#333;border-radius:2px;height:6px;margin:.5rem auto;max-width:300px;"><div style="width:{max(sp,3)}%;height:6px;background:#C8102E;border-radius:2px;"></div></div></div>',unsafe_allow_html=True)
            for col,movie,data,other in [(ca,ma,a,b),(cb,mb,b,a)]:
                with col:
                    gh=''.join([f'<span class="tag">{g}</span>' for g in data['genres_list'][:4]])
                    myr=int(data['release_year']) if pd.notna(data['release_year']) else 'N/A'
                    rt=int(data['runtime']) if pd.notna(data['runtime']) else 0
                    ort=int(other['runtime']) if pd.notna(other['runtime']) else 0
                    bud=f"${data['budget']/1e6:.0f}M" if pd.notna(data['budget']) else 'N/A'
                    rev=f"${data['revenue']/1e6:.0f}M" if pd.notna(data['revenue']) else 'N/A'
                    rc="compare-win" if data['vote_average']>=other['vote_average'] else "compare-val"
                    vc="compare-win" if data['vote_count']>=other['vote_count']     else "compare-val"
                    tc="compare-win" if rt<=ort                                     else "compare-val"
                    st.markdown(f'<div class="compare-panel"><div class="compare-head">{movie}</div><div style="margin-bottom:.8rem;">{gh}</div><div class="compare-row"><span class="compare-lbl">YEAR</span><span class="compare-val">{myr}</span></div><div class="compare-row"><span class="compare-lbl">RUNTIME</span><span class="{tc}">{rt} min</span></div><div class="compare-row"><span class="compare-lbl">RATING</span><span class="{rc}">{data["vote_average"]:.1f}/10</span></div><div class="compare-row"><span class="compare-lbl">VOTES</span><span class="{vc}">{int(data["vote_count"]):,}</span></div><div class="compare-row"><span class="compare-lbl">BUDGET</span><span class="compare-val">{bud}</span></div><div class="compare-row"><span class="compare-lbl">REVENUE</span><span class="compare-val">{rev}</span></div><div class="compare-row"><span class="compare-lbl">LANGUAGE</span><span class="compare-val">{data["original_language"].upper()}</span></div><div style="margin-top:1rem;font-size:.82rem;color:#444;font-style:italic;line-height:1.5;">{data["overview"][:260]}...</div></div>',unsafe_allow_html=True)
                    al=any(w['title']==movie for w in st.session_state.watchlist)
                    if not al:
                        if st.button(f"+ Save {movie[:18]}",key=f"cs{movie}"): add_wl(data,myr); st.rerun()
                    else: st.markdown('<span class="tag tag-red">✓ SAVED</span>',unsafe_allow_html=True)
            st.markdown('<div style="font-size:.78rem;color:var(--red);margin-top:1rem;">RED = winner in that category</div>',unsafe_allow_html=True)
        elif ma==mb: st.info("Select two different films.")

    # ── TAB 4: BROWSE MOOD ────────────────────────────────────
    with t4:
        st.markdown('<div class="section-title">Browse by Mood</div><hr class="section-rule">',unsafe_allow_html=True)
        MOODS={"Comedy":["Comedy"],"Thriller":["Thriller","Horror"],"Romance":["Romance","Drama"],"Adventure":["Action","Adventure"],"Sci-Fi":["Science Fiction"],"Drama":["Drama"],"Family":["Family","Animation"],"Crime":["Crime","Mystery"],"History":["History","Documentary"],"Musical":["Music"]}
        EMOJIS={"Comedy":"😂","Thriller":"😱","Romance":"❤️","Adventure":"🚀","Sci-Fi":"🧠","Drama":"😢","Family":"👨‍👩‍👧","Crime":"🔍","History":"🏛️","Musical":"🎵"}
        mc=st.columns(5)
        for i,(mood,_) in enumerate(MOODS.items()):
            with mc[i%5]:
                if st.button(f"{EMOJIS[mood]} {mood}",key=f"m{i}",use_container_width=True):
                    st.session_state.selected_mood=mood; st.rerun()
        st.markdown("<br>",unsafe_allow_html=True)
        mm1,mm2=st.columns(2)
        with mm1: mnr=st.slider("Min Rating",1.0,9.0,6.5,.5,key="mr")
        with mm2: myr2=st.slider("Year",1950,2017,(1990,2017),key="my")
        cm=st.session_state.selected_mood
        st.markdown(f'<div class="section-label">Showing</div><div class="section-title">{EMOJIS.get(cm,"")} {cm}</div><hr class="section-rule">',unsafe_allow_html=True)
        mr=mood_rec(df,MOODS.get(cm,["Drama"]),mnr,myr2,12)
        if len(mr)==0: st.warning("No results. Lower the minimum rating.")
        else:
            mcls=st.columns(3)
            for i,(_,row) in enumerate(mr.iterrows()):
                with mcls[i%3]:
                    gh=''.join([f'<span class="tag">{g}</span>' for g in row['genres_list'][:3]])
                    ry=int(row['release_year']) if pd.notna(row['release_year']) else 'N/A'
                    st.markdown(f'<div class="film-card"><div class="film-title">{row["title"]}</div><div class="film-meta">{ry} · ★{row["vote_average"]:.1f}</div><div style="margin:.4rem 0;">{gh}</div><div class="film-overview">{row["overview"]}</div></div>',unsafe_allow_html=True)
                    al=any(w['title']==row['title'] for w in st.session_state.watchlist)
                    if not al:
                        if st.button("+ Save",key=f"ms{i}"): add_wl(row,ry); st.rerun()
                    else: st.markdown('<span class="tag tag-red" style="font-size:.6rem;">✓ SAVED</span>',unsafe_allow_html=True)

    # ── TAB 5: WATCHLIST ──────────────────────────────────────
    with t5:
        st.markdown('<div class="section-title">My Watchlist</div><hr class="section-rule">',unsafe_allow_html=True)
        if not st.session_state.watchlist:
            st.markdown('<div style="text-align:center;padding:5rem;"><div style="font-family:Bebas Neue,sans-serif;font-size:4rem;color:#D8D0C4;">NO FILMS YET</div><div style="font-size:.85rem;margin-top:1rem;color:#999;">Add films from the other tabs using the + Save button</div></div>',unsafe_allow_html=True)
        else:
            tot=len(st.session_state.watchlist); wa=sum(1 for w in st.session_state.watchlist if w['watched'])
            st.markdown(f'<div class="kpi-grid" style="grid-template-columns:repeat(3,1fr);margin-bottom:1.5rem;"><div class="kpi-cell"><div class="kpi-value">{tot}</div><div class="kpi-label">Total</div></div><div class="kpi-cell"><div class="kpi-value" style="color:var(--red);">{wa}</div><div class="kpi-label">Watched</div></div><div class="kpi-cell"><div class="kpi-value">{tot-wa}</div><div class="kpi-label">To Watch</div></div></div>',unsafe_allow_html=True)
            sf=st.radio("Filter:",["All","To Watch","Watched"],horizontal=True)
            for i,m in enumerate(st.session_state.watchlist):
                if sf=="Watched"  and not m['watched']: continue
                if sf=="To Watch" and     m['watched']:  continue
                ci2,ca2=st.columns([6,2])
                with ci2:
                    cls="wl-item watched" if m['watched'] else "wl-item"
                    st2="WATCHED ✓" if m['watched'] else "TO WATCH"
                    st.markdown(f'<div class="{cls}"><div style="flex:1;"><div class="wl-title">{m["title"]}</div><div class="wl-meta">{m["year"]} · ★{m["rating"]:.1f} · {m["genres"]}</div></div><span class="tag {"tag-red" if m["watched"] else ""}">{st2}</span></div>',unsafe_allow_html=True)
                with ca2:
                    st.markdown("<br>",unsafe_allow_html=True)
                    lbl="Mark Watched" if not m['watched'] else "Unmark"
                    if st.button(lbl,key=f"t{i}",use_container_width=True): st.session_state.watchlist[i]['watched']=not m['watched']; st.rerun()
                    if st.button("Remove",key=f"r{i}",use_container_width=True): st.session_state.watchlist.pop(i); st.rerun()
            st.markdown("<br>",unsafe_allow_html=True)
            st.download_button("↓ Export CSV",data=pd.DataFrame(st.session_state.watchlist).to_csv(index=False).encode(),file_name="watchlist.csv",mime="text/csv")

st.markdown('<div class="footer">CineMatch PRO &nbsp;·&nbsp; TF-IDF + TruncatedSVD + Weighted Popularity &nbsp;·&nbsp; Scikit-learn · Plotly · Streamlit &nbsp;·&nbsp; TMDB 5000</div>',unsafe_allow_html=True)