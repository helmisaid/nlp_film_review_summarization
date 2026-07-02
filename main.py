import os
import sys
import time
import pandas as pd
import joblib
import streamlit as st

# ============================================================
# PATH SETUP — memastikan modul src/ bisa diakses
# ============================================================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.preprocess import clean_text
from src.summarizer import generate_summary

# ============================================================
# PAGE CONFIG — konfigurasi halaman Streamlit
# ============================================================
st.set_page_config(
    page_title="🎬 Sistem Peringkasan Ulasan Film",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — styling dark mode modern bertema film
# ============================================================
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Styling ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(165deg, #0a0a0f 0%, #0d1117 40%, #10141c 100%);
    }

    /* ── Sidebar Styling ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        border-right: 1px solid rgba(56, 139, 253, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #c9d1d9 !important;
    }

    /* ── Header Gradient ── */
    .gradient-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid rgba(56, 139, 253, 0.2);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), 0 0 60px rgba(56, 139, 253, 0.05);
        position: relative;
        overflow: hidden;
    }
    .gradient-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #388bfd, #58a6ff, #79c0ff, #58a6ff, #388bfd);
    }
    .gradient-header h1 {
        color: #f0f6fc !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.3rem !important;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 10px rgba(56, 139, 253, 0.3);
    }
    .gradient-header p {
        color: #8b949e !important;
        font-size: 1rem !important;
        font-weight: 400;
        margin: 0;
    }

    /* ── Card Styling ── */
    .custom-card {
        background: linear-gradient(145deg, #161b22 0%, #1c2128 100%);
        border: 1px solid rgba(56, 139, 253, 0.12);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
        border-color: rgba(56, 139, 253, 0.3);
    }
    .card-title {
        color: #58a6ff !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.8rem !important;
    }
    .card-value {
        color: #f0f6fc !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    /* ── Sentiment Cards ── */
    .sentiment-positive {
        background: linear-gradient(145deg, #0d2818 0%, #122b1c 100%);
        border: 1px solid rgba(63, 185, 80, 0.3);
    }
    .sentiment-positive:hover {
        border-color: rgba(63, 185, 80, 0.5);
        box-shadow: 0 8px 30px rgba(63, 185, 80, 0.1);
    }
    .sentiment-positive .card-value {
        color: #3fb950 !important;
    }
    .sentiment-negative {
        background: linear-gradient(145deg, #2d1117 0%, #301218 100%);
        border: 1px solid rgba(248, 81, 73, 0.3);
    }
    .sentiment-negative:hover {
        border-color: rgba(248, 81, 73, 0.5);
        box-shadow: 0 8px 30px rgba(248, 81, 73, 0.1);
    }
    .sentiment-negative .card-value {
        color: #f85149 !important;
    }

    /* ── Summary Card ── */
    .summary-card {
        background: linear-gradient(145deg, #161b22 0%, #1c2128 100%);
        border: 1px solid rgba(56, 139, 253, 0.12);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .summary-card p {
        color: #c9d1d9 !important;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .summary-bullet {
        color: #c9d1d9 !important;
        font-size: 0.95rem;
        line-height: 1.8;
        padding: 0.4rem 0;
        border-bottom: 1px solid rgba(56, 139, 253, 0.06);
    }
    .summary-bullet:last-child {
        border-bottom: none;
    }
    .summary-index {
        color: #58a6ff;
        font-weight: 700;
        margin-right: 0.5rem;
    }

    /* ── Stats Metric Override ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #161b22 0%, #1c2128 100%);
        border: 1px solid rgba(56, 139, 253, 0.12);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 139, 253, 0.3);
    }
    div[data-testid="stMetric"] label {
        color: #58a6ff !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.75rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f0f6fc !important;
        font-weight: 700 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 1.8rem !important;
        transition: all 0.25s ease !important;
        letter-spacing: 0.3px;
    }
    /* Primary analyze button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #388bfd 0%, #58a6ff 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(56, 139, 253, 0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 25px rgba(56, 139, 253, 0.45) !important;
        transform: translateY(-1px);
    }
    /* Secondary clear button */
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        color: #8b949e !important;
        border: 1px solid rgba(139, 148, 158, 0.3) !important;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: rgba(139, 148, 158, 0.6) !important;
        color: #c9d1d9 !important;
    }

    /* ── Text Area ── */
    .stTextArea textarea {
        background: #0d1117 !important;
        border: 1px solid rgba(56, 139, 253, 0.15) !important;
        border-radius: 12px !important;
        color: #c9d1d9 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 1rem !important;
        transition: border-color 0.2s ease !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(56, 139, 253, 0.5) !important;
        box-shadow: 0 0 15px rgba(56, 139, 253, 0.1) !important;
    }
    .stTextArea textarea::placeholder {
        color: #484f58 !important;
    }

    /* ── Select Box ── */
    .stSelectbox > div > div {
        background: #0d1117 !important;
        border: 1px solid rgba(56, 139, 253, 0.15) !important;
        border-radius: 12px !important;
        color: #c9d1d9 !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid rgba(56, 139, 253, 0.12);
        border-radius: 10px;
        color: #8b949e;
        font-weight: 500;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #c9d1d9;
        border-color: rgba(56, 139, 253, 0.3);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(56, 139, 253, 0.15), rgba(56, 139, 253, 0.08)) !important;
        border-color: rgba(56, 139, 253, 0.4) !important;
        color: #58a6ff !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: #161b22 !important;
        border-radius: 12px !important;
        color: #c9d1d9 !important;
        font-weight: 500;
    }

    /* ── Divider ── */
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56, 139, 253, 0.2), transparent);
        margin: 1.5rem 0;
    }

    /* ── Film Info Card ── */
    .film-info-card {
        background: linear-gradient(145deg, #161b22 0%, #1c2128 100%);
        border: 1px solid rgba(56, 139, 253, 0.12);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .film-info-card h3 {
        color: #f0f6fc !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    .film-info-card .genre-badge {
        display: inline-block;
        background: rgba(56, 139, 253, 0.15);
        color: #58a6ff;
        border: 1px solid rgba(56, 139, 253, 0.3);
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.3rem;
    }
    .film-info-card .desc {
        color: #8b949e;
        font-size: 0.9rem;
        line-height: 1.6;
        margin-top: 0.5rem;
    }

    /* ── Sidebar Info Box ── */
    .sidebar-info {
        background: linear-gradient(145deg, rgba(56, 139, 253, 0.05), rgba(56, 139, 253, 0.02));
        border: 1px solid rgba(56, 139, 253, 0.15);
        border-radius: 12px;
        padding: 1.2rem;
        margin-top: 1rem;
        text-align: center;
    }
    .sidebar-info h4 {
        color: #58a6ff !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    .sidebar-info p {
        color: #8b949e !important;
        font-size: 0.85rem !important;
        margin: 0.2rem 0 !important;
    }

    /* ── Hide default Streamlit elements ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHING — fungsi load model dan data (TIDAK DIUBAH)
# ============================================================

@st.cache_resource
def load_model():
    """Load model SVM dan vectorizer TF-IDF yang sudah dilatih."""
    model = joblib.load("model/model_svm.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer


@st.cache_data
def load_data():
    """Load data film dan ulasan dari CSV."""
    movies = pd.read_csv("data/processed/movies.csv")
    reviews = pd.read_csv("data/processed/reviews.csv")
    return movies, reviews


# ============================================================
# HELPER — fungsi bantu untuk analisis sentimen
# ============================================================

def analyze_sentiment(text, model, vectorizer):
    """
    Menjalankan pipeline analisis sentimen:
    1. clean_text() — preprocessing (dari src/preprocess.py)
    2. vectorizer.transform() — vektorisasi TF-IDF
    3. model.predict() — prediksi SVM
    4. model.predict_proba() — confidence score
    
    Returns: dict dengan keys sentiment, confidence, cleaned_text
             atau None jika teks tidak valid.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return None

    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = max(proba) * 100

    sentiment = "positive" if pred == 1 else "negative"
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "cleaned_text": cleaned,
    }


def compute_statistics(original_text, summary_sentences):
    """
    Menghitung statistik teks:
    - Jumlah kata pada input
    - Jumlah kalimat pada summary
    - Compression ratio
    """
    words = original_text.split()
    word_count = len(words)

    # Hitung jumlah kalimat asli (split berdasarkan tanda baca)
    import re
    original_sentences = [s.strip() for s in re.split(r'[.!?]+', original_text) if s.strip()]
    original_sentence_count = max(len(original_sentences), 1)

    summary_sentence_count = len(summary_sentences)
    summary_word_count = sum(len(s.split()) for s in summary_sentences)

    compression_ratio = (1 - summary_word_count / max(word_count, 1)) * 100 if word_count > 0 else 0

    return {
        "word_count": word_count,
        "original_sentences": original_sentence_count,
        "summary_sentences": summary_sentence_count,
        "summary_words": summary_word_count,
        "compression_ratio": compression_ratio,
    }


def render_sentiment_card(sentiment, confidence):
    """Render card sentimen dengan warna yang sesuai."""
    if sentiment == "positive":
        emoji = "😊"
        label = "POSITIF"
        css_class = "sentiment-positive"
    else:
        emoji = "😞"
        label = "NEGATIF"
        css_class = "sentiment-negative"

    st.markdown(f"""
    <div class="custom-card {css_class}">
        <div class="card-title">🎭 Analisis Sentimen</div>
        <div class="card-value">{emoji} {label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_confidence_metric(confidence):
    """Render confidence score menggunakan st.metric."""
    st.metric(
        label="🎯 Confidence Score",
        value=f"{confidence:.1f}%",
    )


def render_summary_card(summary_sentences):
    """Render card ringkasan hasil TextRank."""
    bullets_html = ""
    for i, s in enumerate(summary_sentences, 1):
        bullets_html += f'<div class="summary-bullet"><span class="summary-index">#{i}</span> {s}</div>'

    st.markdown(f"""
    <div class="summary-card">
        <div class="card-title">📝 Ringkasan TextRank</div>
        {bullets_html}
    </div>
    """, unsafe_allow_html=True)


def render_statistics(stats):
    """Render statistik teks menggunakan st.metric dalam columns."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="📊 Jumlah Kata", value=f"{stats['word_count']}")
    with col2:
        st.metric(label="📄 Kalimat Asli", value=f"{stats['original_sentences']}")
    with col3:
        st.metric(label="📉 Compression Ratio", value=f"{stats['compression_ratio']:.1f}%")


def simulate_progress():
    """Simulasi progress bar untuk UX yang lebih baik."""
    progress_bar = st.progress(0, text="Memulai analisis...")
    steps = [
        (20, "Preprocessing teks..."),
        (45, "Vektorisasi TF-IDF..."),
        (65, "Prediksi SVM..."),
        (80, "Menjalankan TextRank..."),
        (95, "Menyusun ringkasan..."),
        (100, "Selesai! ✅"),
    ]
    for pct, msg in steps:
        time.sleep(0.3)
        progress_bar.progress(pct, text=msg)
    time.sleep(0.3)
    progress_bar.empty()


# ============================================================
# LOAD DATA & MODEL
# ============================================================
df_movies, df_reviews = load_data()
model, vectorizer = load_model()


# ============================================================
# SIDEBAR — informasi aplikasi dan kelompok
# ============================================================
with st.sidebar:
    # Logo dan judul sidebar
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 3.5rem; margin-bottom: 0.5rem;">🎬</div>
        <h2 style="color: #f0f6fc; font-weight: 800; margin: 0; font-size: 1.2rem;">
            Film Review AI
        </h2>
        <p style="color: #58a6ff; font-size: 0.8rem; font-weight: 500; letter-spacing: 1px;">
            NLP SUMMARIZATION SYSTEM
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Deskripsi aplikasi
    st.markdown("""
    <p style="color: #8b949e; font-size: 0.85rem; line-height: 1.6;">
        Aplikasi berbasis <b style="color: #58a6ff;">Natural Language Processing</b> 
        untuk menganalisis sentimen dan meringkas ulasan film secara otomatis 
        menggunakan algoritma <b style="color: #58a6ff;">SVM</b> dan 
        <b style="color: #58a6ff;">TextRank</b>.
    </p>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Fitur aplikasi
    st.markdown("""
    <div style="color: #c9d1d9; font-size: 0.85rem;">
        <p style="font-weight: 600; color: #58a6ff; margin-bottom: 0.5rem;">✨ Fitur Utama</p>
        <p style="margin: 0.3rem 0;">🔍 Analisis Sentimen (SVM)</p>
        <p style="margin: 0.3rem 0;">📝 Peringkasan Otomatis (TextRank)</p>
        <p style="margin: 0.3rem 0;">🎥 Database Film Terintegrasi</p>
        <p style="margin: 0.3rem 0;">📊 Statistik & Metrik Detail</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Informasi kelompok
    st.markdown("""
    <div class="sidebar-info">
        <h4>👥 Kelompok 7</h4>
        <p style="font-weight: 500; color: #c9d1d9 !important;">D4 Teknik Informatika</p>
        <p>Universitas Airlangga</p>
        <div style="margin-top: 0.8rem; padding-top: 0.8rem; border-top: 1px solid rgba(56, 139, 253, 0.15);">
            <p style="font-size: 0.75rem !important; color: #484f58 !important;">
                UAS — NLP Film Review Summarization
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# HEADER — gradient header utama
# ============================================================
st.markdown("""
<div class="gradient-header">
    <h1>🎬 Sistem Peringkasan Ulasan Film</h1>
    <p>Analisis sentimen menggunakan <strong>SVM</strong> dan peringkasan otomatis menggunakan <strong>TextRank</strong></p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# TABS — dua mode input: Pilih Film & Input Bebas
# ============================================================
tab_film, tab_bebas = st.tabs(["🎥 Pilih Film dari Database", "✍️ Input Ulasan Bebas"])


# ────────────────────────────────────────────────────────────
# TAB 1: PILIH FILM DARI DATABASE
# ────────────────────────────────────────────────────────────
with tab_film:
    st.markdown("")  # spacing

    # Daftar film dari database
    film_names = df_movies["nama_film"].tolist()
    selected_name = st.selectbox(
        "🎬 Pilih Film",
        film_names,
        help="Pilih film untuk melihat ringkasan ulasan",
    )

    # Data film terpilih
    selected_film = df_movies[df_movies["nama_film"] == selected_name].iloc[0]
    film_id = selected_film["id_film"]

    # Card info film
    st.markdown(f"""
    <div class="film-info-card">
        <h3>🎬 {selected_film['nama_film']}</h3>
        <span class="genre-badge">🏷️ {selected_film['genre']}</span>
        <p class="desc">{selected_film['deskripsi']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Filter ulasan berdasarkan film
    film_reviews = df_reviews[df_reviews["id_film"] == film_id]
    pos_reviews = film_reviews[film_reviews["sentiment"] == "positive"]["ulasan"].tolist()
    neg_reviews = film_reviews[film_reviews["sentiment"] == "negative"]["ulasan"].tolist()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Ringkasan ulasan positif & negatif ──
    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown("""
        <div class="card-title" style="margin-bottom: 0.5rem;">😊 Ringkasan Ulasan Positif</div>
        """, unsafe_allow_html=True)
        if pos_reviews:
            summary_pos = generate_summary(pos_reviews, n=3)
            render_summary_card(summary_pos)
        else:
            st.info("Belum ada ulasan positif untuk film ini.")

    with col_neg:
        st.markdown("""
        <div class="card-title" style="margin-bottom: 0.5rem;">😞 Ringkasan Ulasan Negatif</div>
        """, unsafe_allow_html=True)
        if neg_reviews:
            summary_neg = generate_summary(neg_reviews, n=3)
            render_summary_card(summary_neg)
        else:
            st.info("Belum ada ulasan negatif untuk film ini.")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Semua ulasan dalam expander ──
    with st.expander(f"📋 Semua Ulasan ({len(film_reviews)} ulasan)", expanded=False):
        if len(film_reviews) > 0:
            st.dataframe(
                film_reviews[["ulasan", "sentiment"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("Belum ada ulasan untuk film ini.")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Form input ulasan baru untuk film ──
    st.markdown("""
    <div class="card-title" style="margin-bottom: 0.5rem;">✍️ Tulis Ulasan Baru</div>
    """, unsafe_allow_html=True)

    ulasan_film = st.text_area(
        "Masukkan ulasan Anda untuk film ini",
        placeholder="Contoh: Film ini memiliki alur cerita yang sangat menarik dan akting para pemain sangat memukau...",
        height=120,
        key="ulasan_film_input",
    )

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        btn_analyze_film = st.button("🔍 Analisis", key="btn_analyze_film", type="primary")
    with col_btn2:
        btn_clear_film = st.button("🗑️ Hapus", key="btn_clear_film", type="secondary")

    # Tombol clear — gunakan rerun untuk reset
    if btn_clear_film:
        st.rerun()

    # Proses analisis ulasan film
    if btn_analyze_film:
        if ulasan_film.strip():
            with st.spinner("🔄 Menganalisis ulasan..."):
                simulate_progress()

                # Analisis sentimen menggunakan fungsi backend
                result = analyze_sentiment(ulasan_film, model, vectorizer)

            if result:
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown("""
                <div class="card-title" style="font-size: 1rem; margin-bottom: 1rem;">📊 Hasil Analisis</div>
                """, unsafe_allow_html=True)

                # Sentiment & Confidence dalam 2 kolom
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    render_sentiment_card(result["sentiment"], result["confidence"])
                with res_col2:
                    render_confidence_metric(result["confidence"])

                # Simpan ulasan ke reviews.csv (logika asli dipertahankan)
                new_id = df_reviews["id_ulasan"].max() + 1 if len(df_reviews) > 0 else 1
                new_row = pd.DataFrame([{
                    "id_ulasan": new_id,
                    "id_film": film_id,
                    "ulasan": ulasan_film,
                    "sentiment": result["sentiment"],
                }])
                updated_reviews = pd.concat([df_reviews, new_row], ignore_index=True)
                updated_reviews.to_csv("data/processed/reviews.csv", index=False)
                st.cache_data.clear()

                st.markdown("")  # spacing
                st.success("✅ Ulasan berhasil disimpan! Refresh halaman untuk melihat perubahan.")
            else:
                st.warning("⚠️ Teks tidak dapat diproses. Pastikan ulasan berisi kata-kata yang valid.")
        else:
            st.warning("⚠️ Harap isi ulasan terlebih dahulu.")


# ────────────────────────────────────────────────────────────
# TAB 2: INPUT ULASAN BEBAS
# ────────────────────────────────────────────────────────────
with tab_bebas:
    st.markdown("")  # spacing

    st.markdown("""
    <p style="color: #8b949e; font-size: 0.9rem; margin-bottom: 1rem;">
        Masukkan ulasan film apa saja untuk dianalisis sentimen dan diringkas secara otomatis.
    </p>
    """, unsafe_allow_html=True)

    # Text area besar untuk input bebas
    ulasan_bebas = st.text_area(
        "Masukkan ulasan film",
        placeholder="Tulis atau tempel ulasan film di sini...\n\nContoh: Film ini sangat mengecewakan. Alur ceritanya membosankan dan tidak masuk akal. Akting para pemain terasa dipaksakan. Efek visual yang dijanjikan ternyata tidak sesuai ekspektasi. Satu-satunya hal yang menarik hanyalah soundtrack-nya yang cukup bagus.",
        height=200,
        key="ulasan_bebas_input",
    )

    # Tombol aksi
    col_a1, col_a2, _ = st.columns([1, 1, 4])
    with col_a1:
        btn_analyze_bebas = st.button("🔍 Analyze", key="btn_analyze_bebas", type="primary")
    with col_a2:
        btn_clear_bebas = st.button("🗑️ Clear", key="btn_clear_bebas", type="secondary")

    # Tombol clear
    if btn_clear_bebas:
        st.rerun()

    # Proses analisis ulasan bebas
    if btn_analyze_bebas:
        if ulasan_bebas.strip():
            with st.spinner("🔄 Menganalisis ulasan..."):
                simulate_progress()

                # 1. Analisis sentimen menggunakan backend
                result = analyze_sentiment(ulasan_bebas, model, vectorizer)

                # 2. Generate summary menggunakan TextRank
                summary_sentences = generate_summary([ulasan_bebas], n=3)

                # 3. Hitung statistik
                stats = compute_statistics(ulasan_bebas, summary_sentences)

            if result:
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                # ── Section: Hasil Analisis ──
                st.markdown("""
                <div class="card-title" style="font-size: 1rem; margin-bottom: 1rem;">📊 Hasil Analisis</div>
                """, unsafe_allow_html=True)

                # Card 1 & 2: Sentiment + Confidence
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    render_sentiment_card(result["sentiment"], result["confidence"])
                with res_col2:
                    render_confidence_metric(result["confidence"])

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                # Card 3: Summary (TextRank)
                st.markdown("""
                <div class="card-title" style="font-size: 1rem; margin-bottom: 0.5rem;">📝 Ringkasan Otomatis</div>
                """, unsafe_allow_html=True)

                if summary_sentences:
                    render_summary_card(summary_sentences)
                else:
                    st.info("Teks terlalu pendek untuk diringkas.")

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                # Card 4: Statistics
                st.markdown("""
                <div class="card-title" style="font-size: 1rem; margin-bottom: 0.5rem;">📈 Statistik Teks</div>
                """, unsafe_allow_html=True)
                render_statistics(stats)

            else:
                st.warning("⚠️ Teks tidak dapat diproses. Pastikan ulasan berisi kata-kata yang valid.")
        else:
            st.warning("⚠️ Harap isi ulasan terlebih dahulu.")


# ============================================================
# FOOTER — informasi tambahan di bawah halaman
# ============================================================
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 1rem 0 2rem 0;">
    <p style="color: #484f58; font-size: 0.8rem; margin: 0;">
        🎬 Film Review Summarization System — Kelompok 7 · D4 Teknik Informatika · Universitas Airlangga
    </p>
    <p style="color: #30363d; font-size: 0.7rem; margin: 0.3rem 0 0 0;">
        Powered by SVM & TextRank · Built with Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
