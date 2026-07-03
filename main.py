import os
import sys
import re
import pandas as pd
import joblib
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.preprocess import clean_text
from src.summarizer import generate_summary

# =====================================================================
# 3.1 — Setup Streamlit & Konfigurasi Halaman
# =====================================================================

st.set_page_config(
    page_title="🎬 Film Review Summarizer",
    page_icon="🎬",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
/* ---- Global Styles ---- */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: #0b0f19;
    color: #f1f5f9;
}

/* ---- Header & Hero Section ---- */
.hero-container {
    padding: 2.5rem 1.5rem;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: -0.03em;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: #94a3b8;
    max-width: 700px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ---- Glassmorphism Film Card ---- */
.film-card {
    background: rgba(17, 24, 39, 0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 16px;
    overflow: hidden;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1px solid rgba(255, 255, 255, 0.07);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    height: 100%;
    cursor: pointer;
}
.film-card:hover {
    transform: translateY(-8px) scale(1.02);
    border-color: rgba(129, 140, 248, 0.5);
    box-shadow: 0 15px 35px rgba(99, 102, 241, 0.25);
}
.film-card img {
    width: 100%;
    height: 340px;
    object-fit: cover;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    transition: transform 0.5s ease;
}
.film-card:hover img {
    transform: scale(1.05);
}
.film-card-body {
    padding: 1.3rem;
}
.film-card-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.4rem;
    line-height: 1.3;
}
.film-card-genre {
    font-size: 0.85rem;
    color: #a78bfa;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ---- Detail Page Elements ---- */
.detail-container {
    background: rgba(17, 24, 39, 0.5);
    border-radius: 20px;
    padding: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}
.detail-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}
.detail-genre {
    font-size: 1.05rem;
    color: #a78bfa;
    font-weight: 600;
    margin-bottom: 1.2rem;
}
.detail-desc {
    font-size: 1.05rem;
    color: #cbd5e1;
    line-height: 1.75;
}

/* ---- Custom Summary Boxes ---- */
.summary-box {
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(8px);
    border-radius: 14px;
    padding: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 4px 25px rgba(0, 0, 0, 0.1);
    height: 100%;
}
.summary-pos {
    border-left: 5px solid #10b981;
    box-shadow: inset 5px 0 15px rgba(16, 185, 129, 0.05);
}
.summary-neg {
    border-left: 5px solid #ef4444;
    box-shadow: inset 5px 0 15px rgba(239, 68, 68, 0.05);
}
.summary-box-title {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.title-pos { color: #34d399; }
.title-neg { color: #f87171; }

.summary-list {
    margin: 0;
    padding-left: 1.2rem;
    color: #e2e8f0;
}
.summary-item {
    font-size: 0.98rem;
    line-height: 1.65;
    margin-bottom: 0.75rem;
}
.summary-item:last-child {
    margin-bottom: 0;
}

/* ---- Streamlit Widgets Styling ---- */
div[data-testid="stMetric"] {
    background: rgba(17, 24, 39, 0.6) !important;
    border-radius: 14px !important;
    padding: 1.2rem 1.5rem !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
}

/* Form Styling */
.stForm {
    background: rgba(17, 24, 39, 0.4) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    padding: 2rem !important;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #0b0f19;
}
::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #334155;
}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# 3.2 — Pemuatan Aset & Model (dengan Caching)
# =====================================================================

@st.cache_resource
def load_model():
    """Muat model SVM dan TF-IDF vectorizer dari file pkl."""
    model = joblib.load("model/model_svm.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    # Patch kompatibilitas: model dari sklearn 1.9.0 menyimpan probability='deprecated',
    # sedangkan sklearn 1.8.0 memerlukan nilai integer/boolean di level C (libsvm).
    if hasattr(model, "probability") and not isinstance(model.probability, (bool, int)):
        model.probability = False
    return model, vectorizer


@st.cache_data
def load_data():
    """Muat movies.csv dan reviews.csv."""
    movies = pd.read_csv("data/processed/movies.csv")
    reviews = pd.read_csv("data/processed/reviews.csv")
    return movies, reviews


# =====================================================================
# Inisialisasi session state
# =====================================================================

# Ambil ID film dari parameter query URL jika ada (Routing URL)
query_params = st.query_params
selected_id_param = query_params.get("id", None)

if selected_id_param is not None:
    try:
        st.session_state["selected_film"] = int(selected_id_param)
    except ValueError:
        st.session_state["selected_film"] = None
else:
    st.session_state["selected_film"] = None

# Inisialisasi alert hasil analisis sentimen agar tidak hilang setelah rerun
if "sentiment_alert" not in st.session_state:
    st.session_state["sentiment_alert"] = None

# Muat data & model
df_movies, df_reviews = load_data()
model, vectorizer = load_model()


# =====================================================================
# Routing: Katalog vs Detail
# =====================================================================

if st.session_state["selected_film"] is None:
    # ==================================================================
    # 3.3 — Halaman Utama: Katalog Film
    # ==================================================================

    st.markdown("""
    <div class="hero-container">
        <p class="hero-title">🎬 Film Review Summarizer</p>
        <p class="hero-subtitle">
            Sistem Peringkasan Ulasan Film Berbasis Graf Semantik — 
            Pilih film untuk melihat ringkasan sentimen dan tulis ulasan baru.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Tentukan jumlah kolom grid (maks 4 per baris)
    num_films = len(df_movies)
    cols_per_row = min(4, num_films) if num_films > 0 else 1

    for row_start in range(0, num_films, cols_per_row):
        cols = st.columns(cols_per_row)
        for idx, col in enumerate(cols):
            film_idx = row_start + idx
            if film_idx >= num_films:
                break
            film = df_movies.iloc[film_idx]

            with col:
                # Poster film
                poster_url = film["gambar"] if pd.notna(film["gambar"]) and str(film["gambar"]).startswith("http") and len(str(film["gambar"])) > 10 else None
                
                # Desain HTML Card yang premium & clickable
                card_html = f"""
                <a href="/?id={film['id_film']}" target="_self" style="text-decoration: none; color: inherit;">
                    <div class="film-card">
                        {f'<img src="{poster_url}" alt="{film["nama_film"]}" />' if poster_url else 
                         f'<div style="width:100%;height:340px;background:linear-gradient(135deg,#6366f1,#a855f7);'
                         f'display:flex;align-items:center;justify-content:center;font-size:3.5rem;">🎬</div>'}
                        <div class="film-card-body">
                            <div class="film-card-genre">🎭 {film['genre']}</div>
                            <div class="film-card-title">{film['nama_film']}</div>
                        </div>
                    </div>
                </a>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("")  # spacing antar baris

else:
    # ==================================================================
    # 3.4 — Halaman Detail Film
    # ==================================================================

    selected_id = st.session_state["selected_film"]

    # Filter data film
    film_row = df_movies[df_movies["id_film"] == selected_id]
    if film_row.empty:
        st.error("Film tidak ditemukan.")
        if st.button("← Kembali ke Katalog"):
            st.query_params.clear()
            st.rerun()
        st.stop()

    film = film_row.iloc[0]

    # Tombol kembali
    if st.button("← Kembali ke Katalog"):
        st.query_params.clear()
        st.rerun()

    st.markdown("---")

    # Layout: poster (kiri) + metadata (kanan)
    col_poster, col_info = st.columns([1, 2])

    with col_poster:
        poster_url = film["gambar"] if pd.notna(film["gambar"]) and str(film["gambar"]).startswith("http") and len(str(film["gambar"])) > 10 else None
        if poster_url:
            st.image(poster_url, use_container_width=True)
        else:
            st.markdown(
                '<div style="width:100%;height:380px;background:linear-gradient(135deg,#667eea,#764ba2);'
                'border-radius:16px;display:flex;align-items:center;justify-content:center;'
                'font-size:4rem;">🎬</div>',
                unsafe_allow_html=True,
            )

    with col_info:
        st.markdown(f'<p class="detail-title">{film["nama_film"]}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="detail-genre">🎭 {film["genre"]}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="detail-desc">{film["deskripsi"]}</p>', unsafe_allow_html=True)

    st.markdown("---")

    # Filter ulasan film ini
    film_reviews = df_reviews[df_reviews["id_film"] == selected_id]
    pos_reviews_df = film_reviews[film_reviews["sentiment"] == "positive"]
    neg_reviews_df = film_reviews[film_reviews["sentiment"] == "negative"]

    pos_reviews = pos_reviews_df["ulasan"].tolist()
    neg_reviews = neg_reviews_df["ulasan"].tolist()

    # Metrik jumlah ulasan
    m1, m2, m3 = st.columns(3)
    m1.metric("📊 Total Ulasan", len(film_reviews))
    m2.metric("👍 Ulasan Positif", len(pos_reviews))
    m3.metric("👎 Ulasan Negatif", len(neg_reviews))

    st.markdown("---")

    # ==================================================================
    # 5.4 — Tampilan Dual-Box Summary
    # ==================================================================

    st.subheader("📝 Ringkasan Ulasan")

    col_pos, col_neg = st.columns(2)

    with col_pos:
        if pos_reviews:
            summary_pos = generate_summary(pos_reviews, n=3)
            items_html = "".join([f'<li class="summary-item">{s}</li>' for s in summary_pos])
            st.markdown(f"""
            <div class="summary-box summary-pos">
                <div class="summary-box-title title-pos">✅ Ringkasan Ulasan Positif</div>
                <ol class="summary-list">
                    {items_html}
                </ol>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="summary-box summary-pos">
                <div class="summary-box-title title-pos">✅ Ringkasan Ulasan Positif</div>
                <p style="color: #94a3b8; font-size: 0.95rem; margin: 0; padding-left: 0.5rem;">Belum ada ulasan positif untuk film ini.</p>
            </div>
            """, unsafe_allow_html=True)

    with col_neg:
        if neg_reviews:
            summary_neg = generate_summary(neg_reviews, n=3)
            items_html = "".join([f'<li class="summary-item">{s}</li>' for s in summary_neg])
            st.markdown(f"""
            <div class="summary-box summary-neg">
                <div class="summary-box-title title-neg">❌ Ringkasan Ulasan Negatif</div>
                <ol class="summary-list">
                    {items_html}
                </ol>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="summary-box summary-neg">
                <div class="summary-box-title title-neg">❌ Ringkasan Ulasan Negatif</div>
                <p style="color: #94a3b8; font-size: 0.95rem; margin: 0; padding-left: 0.5rem;">Belum ada ulasan negatif untuk film ini.</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ==================================================================
    # 5.5 — Tampilan Daftar Ulasan Mentah
    # ==================================================================

    with st.expander(f"📋 Semua Ulasan ({len(film_reviews)} ulasan)"):
        if len(film_reviews) > 0:
            # Filter sentimen (opsional)
            filter_sentiment = st.selectbox(
                "Filter berdasarkan sentimen:",
                ["Semua", "positive", "negative"],
                key="filter_sentiment",
            )

            if filter_sentiment == "Semua":
                df_display = film_reviews[["ulasan", "sentiment"]]
            else:
                df_display = film_reviews[film_reviews["sentiment"] == filter_sentiment][["ulasan", "sentiment"]]

            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.write("Belum ada ulasan untuk film ini.")

    st.markdown("---")

    # ==================================================================
    # 3.5 — Form Input Ulasan Baru
    # ==================================================================

    st.subheader("✍️ Tulis Ulasan Baru")

    # Tampilkan notifikasi hasil analisis sentimen jika ada (setelah rerun)
    if st.session_state["sentiment_alert"] is not None:
        alert_type, alert_msg = st.session_state["sentiment_alert"]
        if alert_type == "success":
            st.success(alert_msg)
        else:
            st.error(alert_msg)
        # Hapus alert setelah ditampilkan agar tidak muncul terus-menerus
        st.session_state["sentiment_alert"] = None

    with st.form(key="form_ulasan"):
        ulasan_baru = st.text_area(
            "Masukkan ulasan Anda:",
            height=120,
            placeholder="Tulis pendapat Anda tentang film ini...",
        )
        submitted = st.form_submit_button("🔍 Analisis Sentimen")

    if submitted:
        if ulasan_baru.strip():
            # 1. Bersihkan teks
            cleaned = clean_text(ulasan_baru)

            if cleaned:
                try:
                    # 2. Transform menggunakan vectorizer
                    vec = vectorizer.transform([cleaned])

                    # 3. Prediksi sentimen
                    pred = model.predict(vec)[0]

                    # 4. Hitung confidence
                    #    Model dilatih dengan SVC(kernel='linear') tanpa probability=True,
                    #    jadi gunakan decision_function sebagai skor kepercayaan.
                    try:
                        proba = model.predict_proba(vec)[0]
                        confidence = max(proba) * 100
                    except AttributeError:
                        # Fallback: decision_function → sigmoid approximation
                        import numpy as np
                        decision = model.decision_function(vec)[0]
                        confidence = float(1 / (1 + np.exp(-abs(decision)))) * 100

                    # 5. Tentukan label sentimen & simpan hasil alert ke session state
                    if pred in (1, "positive"):
                        sentiment = "positive"
                        st.session_state["sentiment_alert"] = ("success", f"✅ Sentimen: POSITIF (confidence: {confidence:.1f}%)")
                    else:
                        sentiment = "negative"
                        st.session_state["sentiment_alert"] = ("error", f"❌ Sentimen: NEGATIF (confidence: {confidence:.1f}%)")

                    # 6. Simpan ulasan baru ke reviews.csv
                    new_id = int(df_reviews["id_ulasan"].max()) + 1 if len(df_reviews) > 0 else 1
                    new_row = pd.DataFrame([{
                        "id_ulasan": new_id,
                        "id_film": selected_id,
                        "ulasan": ulasan_baru,
                        "sentiment": sentiment,
                    }])
                    updated_reviews = pd.concat([df_reviews, new_row], ignore_index=True)
                    updated_reviews.to_csv("data/processed/reviews.csv", index=False)

                    # 7. Refresh data & rerun
                    st.cache_data.clear()
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat analisis: {e}")
            else:
                st.warning("⚠️ Teks tidak dapat diproses. Pastikan ulasan berisi kata-kata yang valid.")
        else:
            st.warning("⚠️ Harap isi ulasan terlebih dahulu.")
