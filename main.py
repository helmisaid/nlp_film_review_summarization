import os
import sys
import pandas as pd
import joblib
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.preprocess import clean_text
from src.summarizer import generate_summary

# --- Caching ---

@st.cache_resource
def load_model():
    model = joblib.load("model/model_svm.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer


@st.cache_data
def load_data():
    movies = pd.read_csv("data/processed/movies.csv")
    reviews = pd.read_csv("data/processed/reviews.csv")
    return movies, reviews


# --- Main ---

st.title("Sistem Peringkasan Ulasan Film")

df_movies, df_reviews = load_data()
model, vectorizer = load_model()

# Pilih film
film_names = df_movies["nama_film"].tolist()
selected_name = st.selectbox("Pilih Film", film_names)

selected_film = df_movies[df_movies["nama_film"] == selected_name].iloc[0]
film_id = selected_film["id_film"]

# Info film
st.write(f"**Genre:** {selected_film['genre']}")
st.write(f"**Deskripsi:** {selected_film['deskripsi']}")

# Filter reviews
film_reviews = df_reviews[df_reviews["id_film"] == film_id]

pos_reviews = film_reviews[film_reviews["sentiment"] == "positive"]["ulasan"].tolist()
neg_reviews = film_reviews[film_reviews["sentiment"] == "negative"]["ulasan"].tolist()

# Ringkasan
st.subheader("Ringkasan Positif")
if pos_reviews:
    summary_pos = generate_summary(pos_reviews, n=3)
    for i, s in enumerate(summary_pos, 1):
        st.write(f"{i}. {s}")
else:
    st.write("Belum ada ulasan positif.")

st.subheader("Ringkasan Negatif")
if neg_reviews:
    summary_neg = generate_summary(neg_reviews, n=3)
    for i, s in enumerate(summary_neg, 1):
        st.write(f"{i}. {s}")
else:
    st.write("Belum ada ulasan negatif.")

# Semua ulasan
with st.expander(f"Semua Ulasan ({len(film_reviews)} ulasan)"):
    if len(film_reviews) > 0:
        st.dataframe(film_reviews[["ulasan", "sentiment"]])
    else:
        st.write("Belum ada ulasan untuk film ini.")

# Form input ulasan baru
st.subheader("Tulis Ulasan Baru")
ulasan_baru = st.text_area("Masukkan ulasan Anda")

if st.button("Analisis Sentimen"):
    if ulasan_baru.strip():
        cleaned = clean_text(ulasan_baru)
        if cleaned:
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]
            confidence = max(proba) * 100

            if pred == 1:
                sentiment = "positive"
                st.success(f"Sentimen: POSITIF (confidence: {confidence:.1f}%)")
            else:
                sentiment = "negative"
                st.error(f"Sentimen: NEGATIF (confidence: {confidence:.1f}%)")

            # Simpan ke reviews.csv
            new_id = df_reviews["id_ulasan"].max() + 1 if len(df_reviews) > 0 else 1
            new_row = pd.DataFrame([{
                "id_ulasan": new_id,
                "id_film": film_id,
                "ulasan": ulasan_baru,
                "sentiment": sentiment
            }])
            df_reviews = pd.concat([df_reviews, new_row], ignore_index=True)
            df_reviews.to_csv("data/processed/reviews.csv", index=False)
            st.cache_data.clear()
            st.rerun()
        else:
            st.warning("Teks tidak dapat diproses. Pastikan ulasan berisi kata-kata yang valid.")
    else:
        st.warning("Harap isi ulasan terlebih dahulu.")
