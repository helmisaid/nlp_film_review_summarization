# 🎬 Workflow Proyek: Sistem Peringkasan Ulasan Film Berbasis Graf Semantik
> **Kelompok 7 — D4 Teknik Informatika, Universitas Airlangga**

---

## 📁 Struktur File Proyek (Target Akhir)

```
nlp_film_reviews_summarization/
│
├── 📂 data/                                    ← Semua data proyek
│   ├── raw/                                    ← Data mentah, JANGAN dimodifikasi
│   │   └── IMDB Dataset.csv                    ← Dataset IMDb asli (Bahasa Inggris)
│   └── processed/                              ← Data hasil pemrosesan/transformasi
│       ├── IMDB_Indonesia.csv                  ← Hasil terjemahan (output translator)
│       ├── movies.csv                          ← Tabel induk metadata film
│       └── reviews.csv                         ← Tabel ulasan (relasi ke movies.csv)
│
├── 📂 model/                                   ← Artefak model ML yang sudah dilatih
│   ├── model_svm.pkl                           ← Model SVM terlatih (via joblib)
│   └── vectorizer.pkl                          ← TF-IDF Vectorizer terlatih (via joblib)
│
├── 📂 src/                                     ← Kode sumber utama (modular & reusable)
│   ├── __init__.py                             ← Penanda modul Python
│   ├── preprocess.py                           ← Fungsi pra-pemrosesan teks
│   ├── graph.py                                ← Pembangunan graf semantik (NetworkX)
│   └── summarizer.py                           ← TextRank & ekstraksi kalimat ringkasan
│
├── 📂 scripts/                                 ← Skrip sekali-jalan (pipeline offline)
│   ├── translate_dataset.py                    ← Terjemahan IMDb EN → ID (dari translator.py)
│   ├── train_model.py                          ← Pelatihan & penyimpanan model SVM
│   └── prepare_database.py                     ← Pembuatan movies.csv & reviews.csv
│
├── 📂 docs/                                    ← Dokumentasi proyek
│   └── PROYEK UAS_KELOMPOK 7.pdf
│
├── main.py                                     ← Entry point aplikasi Streamlit
├── WORKFLOW.md                                 ← Panduan alur kerja proyek (file ini)
├── README.md                                   ← Deskripsi proyek untuk GitHub
├── requirements.txt                            ← Daftar dependensi pustaka Python
└── .gitignore                                  ← Abaikan venv/ & file sementara
```


---

## 🗺️ Peta Alur Keseluruhan

```
[TAHAP 1] Training Model SVM
        ↓
[TAHAP 2] Perancangan Basis Data Relasional
        ↓
[TAHAP 3] Aplikasi Streamlit + Klasifikasi Real-Time
        ↓
[TAHAP 4] Pra-pemrosesan NLP + Pembangunan Graf Semantik
        ↓
[TAHAP 5] TextRank Summarization + Evaluasi ROUGE
```

---

## ✅ TAHAP 0: Persiapan Lingkungan (Prerequisites)

### 0.1 — Setup Virtual Environment & Instalasi Dependensi
- [ ] Pastikan Python 3.9+ terinstal
- [ ] Buat virtual environment agar dependensi terisolasi dari sistem:
  ```bash
  python -m venv venv
  ```
- [ ] Aktifkan virtual environment:
  - Windows: `.\venv\Scripts\activate`
  - Mac/Linux: `source venv/bin/activate`
- [ ] Install semua library yang dibutuhkan melalui file `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```
- [ ] Verifikasi instalasi dengan `pip list`

### 0.2 — Download Resource NLTK
- [ ] Jalankan perintah Python berikut untuk mengunduh resource NLTK yang dibutuhkan:
  ```python
  import nltk
  nltk.download('punkt')         # Sentence tokenizer (untuk memecah kalimat)
  nltk.download('stopwords')     # Daftar stopwords (NLTK menyediakan corpus 'indonesian')
  ```

### 0.3 — Verifikasi Dataset
- [ ] Pastikan `dataset/IMDB Dataset.csv` tersedia (66 MB, 50.000 baris)
- [ ] Pastikan `dataset/IMDB Dataset_Indonesia.csv` tersedia (hasil dari `translator.py`)
- [ ] Cek isi dataset: kolom wajib `ulasan` dan `sentiment` harus ada di CSV Indonesia

---

## 📌 TAHAP 1: Pengembangan Model Klasifikasi SVM

> **File output:** `model/model_svm.pkl`, `model/vectorizer.pkl`
> **File kerja:** `train_model.py`

### 1.1 — Memuat Dataset Terjemahan
- [ ] Baca `dataset/IMDB Dataset_Indonesia.csv` menggunakan `pd.read_csv()`
- [ ] Tampilkan info awal: jumlah baris, kolom, dan contoh 5 baris pertama (`df.head()`)
- [ ] Cek distribusi label sentimen: hitung jumlah `positive` dan `negative`
- [ ] Tangani nilai kosong/null: hapus baris yang kolom `ulasan`-nya kosong (`df.dropna()`)

### 1.2 — Pra-Pemrosesan Teks untuk Training
- [ ] **Case Folding**: Konversi semua teks ulasan ke huruf kecil (`str.lower()`)
- [ ] **Pembersihan Karakter**: Hapus karakter non-alfabet (angka, tanda baca, simbol HTML) menggunakan `re.sub()`
  - Pola regex: `[^a-zA-Z\s]`
- [ ] **Tokenisasi**: Pecah teks menjadi token kata individual (opsional, TF-IDF menangani ini)
- [ ] **Stopwords Removal** (opsional untuk dataset Indonesia, karena terjemahan dari Inggris):
  - Gunakan daftar stopwords Bahasa Indonesia (dari file eksternal atau `Sastrawi`)
- [ ] Simpan kolom teks yang sudah bersih ke kolom baru `ulasan_bersih`

### 1.3 — Encoding Label
- [ ] Konversi label teks ke numerik:
  - `'positive'` → `1`
  - `'negative'` → `0`
- [ ] Simpan ke kolom baru `label` menggunakan `map()` atau `LabelEncoder`

### 1.4 — Ekstraksi Fitur dengan TF-IDF
- [ ] Inisialisasi `TfidfVectorizer` dengan parameter:
  - `max_features=10000` (maksimal 10.000 kata teratas sebagai fitur)
  - `ngram_range=(1, 2)` (unigram dan bigram)
  - `min_df=2` (abaikan kata yang muncul kurang dari 2 kali)
- [ ] Pisahkan data X (fitur teks: `ulasan_bersih`) dan y (label: `label`)

### 1.5 — Split Data Training dan Testing
- [ ] Bagi dataset dengan `train_test_split()`:
  - `test_size=0.2` (80% training, 20% testing)
  - `random_state=42` (untuk reproducibility)
  - `stratify=y` (menjaga proporsi kelas positif/negatif di kedua split)
- [ ] Fit `TfidfVectorizer` HANYA pada data training: `vectorizer.fit(X_train)`
- [ ] Transform data training: `X_train_vec = vectorizer.transform(X_train)`
- [ ] Transform data testing: `X_test_vec = vectorizer.transform(X_test)`

### 1.6 — Pelatihan Model SVM
- [ ] Inisialisasi model SVM: `SVC(kernel='linear', C=1.0, probability=True)`
  - `kernel='linear'`: Sesuai spesifikasi proyek
  - `probability=True`: Diperlukan untuk mendapatkan skor probabilitas prediksi
- [ ] Latih model: `model_svm.fit(X_train_vec, y_train)`
- [ ] Tampilkan pesan konfirmasi bahwa training selesai

### 1.7 — Evaluasi Model
- [ ] Prediksi data testing: `y_pred = model_svm.predict(X_test_vec)`
- [ ] Hitung dan cetak metrik evaluasi menggunakan `classification_report()`:
  - **Accuracy**: Persentase prediksi benar keseluruhan
  - **Precision**: Keakuratan prediksi positif
  - **Recall**: Kemampuan menemukan semua data positif
  - **F1-Score**: Harmonic mean dari Precision dan Recall
- [ ] Cetak **Confusion Matrix** menggunakan `confusion_matrix()` untuk visualisasi error
- [ ] (Opsional) Plot confusion matrix menggunakan `matplotlib` dan `seaborn`

### 1.8 — Penyimpanan Model dan Vectorizer
- [ ] Buat direktori `model/` jika belum ada: `os.makedirs('model', exist_ok=True)`
- [ ] Simpan model SVM: `joblib.dump(model_svm, 'model/model_svm.pkl')`
- [ ] Simpan vectorizer: `joblib.dump(vectorizer, 'model/vectorizer.pkl')`
- [ ] Verifikasi file tersimpan dengan mengecek ukuran file

---

## 📌 TAHAP 2: Perancangan Basis Data Relasional

> **File output:** `dataset/movies.csv`, `dataset/reviews.csv`
> **File kerja:** `prepare_database.py`

### 2.1 — Rancangan Skema Tabel

#### Tabel `movies.csv` (Tabel Induk)
| Kolom         | Tipe      | Keterangan                      |
|---------------|-----------|---------------------------------|
| `id_film`     | int (PK)  | Identifier unik film            |
| `nama_film`   | str       | Judul film                      |
| `gambar`      | str (URL) | URL poster film                 |
| `deskripsi`   | str       | Sinopsis singkat film           |
| `genre`       | str       | Genre film (Action, Drama, dll) |

#### Tabel `reviews.csv` (Tabel Anak)
| Kolom       | Tipe     | Keterangan                                |
|-------------|----------|-------------------------------------------|
| `id_ulasan` | int (PK) | Identifier unik ulasan (auto-increment)   |
| `id_film`   | int (FK) | Referensi ke `movies.csv`                 |
| `ulasan`    | str      | Teks ulasan dalam Bahasa Indonesia        |
| `sentiment` | str      | Label sentimen: `positive` atau `negative`|

### 2.2 — Pembuatan `movies.csv` (Data Mock)
- [ ] Buat DataFrame Pandas dengan minimal **10 film** sebagai data dummy
- [ ] Setiap film harus memiliki semua kolom yang telah dirancang
- [ ] Gunakan URL poster dari sumber yang valid (misal: IMDB, TMDB, atau placeholder)
- [ ] Simpan ke `dataset/movies.csv` dengan `df.to_csv(..., index=False)`

### 2.3 — Pemetaan & Pembuatan `reviews.csv`
- [ ] Ambil sejumlah baris dari `IMDB Dataset_Indonesia.csv` sebagai ulasan awal
- [ ] Tambahkan kolom `id_film` secara manual atau semi-otomatis:
  - Kelompokkan ulasan ke masing-masing film (misal: 50-100 ulasan per film)
  - Lakukan pemetaan `id_film` berdasarkan urutan/batch
- [ ] Tambahkan kolom `id_ulasan` sebagai auto-increment: `range(1, len(df)+1)`
- [ ] Pastikan kolom `sentiment` sudah ada (dari dataset asli: `positive`/`negative`)
- [ ] Simpan ke `dataset/reviews.csv` dengan `df.to_csv(..., index=False)`

### 2.4 — Validasi Relasional
- [ ] Pastikan semua `id_film` di `reviews.csv` ada di `movies.csv` (tidak ada orphan record)
- [ ] Cek tidak ada duplikasi `id_ulasan`
- [ ] Cek tidak ada nilai null pada kolom wajib

---

## 📌 TAHAP 3: Aplikasi Streamlit + Klasifikasi Real-Time

> **File kerja:** `main.py`

### 3.1 — Setup Streamlit & Konfigurasi Halaman
- [ ] Import semua library yang diperlukan di bagian atas `main.py`
- [ ] Konfigurasi halaman Streamlit:
  ```python
  st.set_page_config(
      page_title="🎬 Film Review Summarizer",
      page_icon="🎬",
      layout="wide"
  )
  ```
- [ ] Tambahkan CSS kustom (opsional) untuk mempercantik tampilan

### 3.2 — Pemuatan Aset & Model (dengan Caching)
- [ ] Buat fungsi `load_model()` dengan decorator `@st.cache_resource`:
  - Muat `model/model_svm.pkl` menggunakan `joblib.load()`
  - Muat `model/vectorizer.pkl` menggunakan `joblib.load()`
  - Return keduanya sebagai tuple
- [ ] Buat fungsi `load_data()` dengan decorator `@st.cache_data`:
  - Baca `dataset/movies.csv`
  - Baca `dataset/reviews.csv`
  - Return keduanya sebagai tuple DataFrame

### 3.3 — Halaman Utama: Katalog Film
- [ ] Tampilkan judul aplikasi dan deskripsi singkat di bagian atas
- [ ] Tampilkan daftar film dari `movies.csv` dalam format grid (gunakan `st.columns()`)
- [ ] Untuk setiap film, tampilkan:
  - Poster film (`st.image()`)
  - Nama film (`st.subheader()`)
  - Genre (`st.caption()`)
  - Tombol "📋 Lihat Detail" dengan `key=f"btn_{id_film}"`
- [ ] Gunakan `st.session_state` untuk melacak `id_film` yang dipilih
- [ ] Saat tombol diklik, simpan `id_film` ke `st.session_state['selected_film']`

### 3.4 — Halaman Detail Film
- [ ] Deteksi apakah ada `selected_film` di `st.session_state`
- [ ] Jika ya, filter data film dan ulasan berdasarkan `id_film` terpilih
- [ ] Tampilkan metadata film:
  - Poster (kolom kiri), Judul + Genre + Deskripsi (kolom kanan)
  - Tombol "← Kembali" untuk reset `st.session_state`
- [ ] Tampilkan jumlah ulasan positif dan negatif sebagai metrik (`st.metric()`)

### 3.5 — Form Input Ulasan Baru
- [ ] Buat form menggunakan `st.form(key='form_ulasan')`:
  - `st.text_area()` untuk input teks ulasan
  - `st.form_submit_button("Analisis Sentimen")`
- [ ] Ketika form di-submit:
  1. Ambil teks ulasan dari input
  2. Bersihkan teks (case folding, hapus karakter non-alfabet)
  3. Transform teks menggunakan `vectorizer.transform([teks_bersih])`
  4. Prediksi sentimen: `model_svm.predict(teks_vec)`
  5. Ambil probabilitas: `model_svm.predict_proba(teks_vec)`
  6. Tampilkan hasil:
     - Jika positif: `st.success("✅ Sentimen: POSITIF (confidence: X%)")`
     - Jika negatif: `st.error("❌ Sentimen: NEGATIF (confidence: X%)")`
  7. Tambahkan ulasan baru ke `reviews.csv`:
     - Buat baris baru dengan `id_ulasan` baru (max existing + 1)
     - Append ke DataFrame dan simpan kembali ke CSV
  8. Refresh data dengan `st.cache_data.clear()` dan `st.rerun()`

---

## 📌 TAHAP 4: Pra-Pemrosesan NLP & Pembangunan Graf Semantik

> **Semua logika ini terintegrasi di dalam `main.py` sebagai fungsi terpisah**

### 4.1 — Fungsi Pra-Pemrosesan Teks Kalimat
- [ ] Buat fungsi `preprocess_text(text: str) -> str`:
  - **Case Folding**: `text.lower()`
  - **Hapus Karakter Non-Alfabet**: `re.sub(r'[^a-z\s]', '', text)`
  - **Stopwords Removal**: Hapus kata-kata umum Bahasa Indonesia
    - Gunakan daftar stopwords Indonesia (bisa dari library `PySastrawi` atau file teks)
    - Alternatif: buat list manual 50-100 stopwords umum Indonesia
  - **Strip whitespace**: `text.strip()`
  - Return teks yang sudah bersih

### 4.2 — Fungsi Segmentasi Kalimat
- [ ] Buat fungsi `segment_sentences(ulasan_list: list) -> list`:
  - Gabungkan semua ulasan menjadi satu blob teks per kelompok sentimen
  - Pecah menjadi kalimat individual menggunakan `nltk.sent_tokenize()`
  - Filter kalimat yang terlalu pendek (< 5 kata): buang kalimat tidak informatif
  - Hapus kalimat duplikat menggunakan `list(dict.fromkeys())`
  - Return list kalimat unik yang valid

### 4.3 — Fungsi Representasi TF-IDF per Kalimat
- [ ] Buat fungsi `build_sentence_vectors(sentences: list)`:
  - Pra-proses setiap kalimat dengan fungsi `preprocess_text()`
  - Inisialisasi `TfidfVectorizer` baru khusus untuk kumpulan kalimat ini
  - `fit_transform()` pada list kalimat yang sudah bersih
  - Return matrix TF-IDF berbentuk sparse matrix dan daftar kalimat bersih

### 4.4 — Fungsi Perhitungan Cosine Similarity
- [ ] Buat fungsi `compute_similarity_matrix(tfidf_matrix)`:
  - Gunakan `cosine_similarity()` dari `sklearn.metrics.pairwise`
  - Hitung similarity antar semua pasangan kalimat: output matrix N×N
  - Terapkan threshold: set nilai similarity < 0.1 menjadi 0 (hindari edge lemah)
  - Return similarity matrix

### 4.5 — Fungsi Pembangunan Graf Semantik
- [ ] Buat fungsi `build_semantic_graph(similarity_matrix, sentences)`:
  - Inisialisasi graf tidak berarah: `G = nx.Graph()`
  - Tambahkan node untuk setiap kalimat: `G.add_node(i, sentence=sentences[i])`
  - Iterasi semua pasangan kalimat (i, j) dengan i < j:
    - Jika `similarity_matrix[i][j] > 0`:
      - Tambahkan edge: `G.add_edge(i, j, weight=similarity_matrix[i][j])`
  - Return graf G

---

## 📌 TAHAP 5: Perangkingan Kalimat (TextRank) & Evaluasi ROUGE

> **Semua logika ini terintegrasi di dalam `main.py` sebagai fungsi terpisah**

### 5.1 — Fungsi TextRank: Menghitung Skor Kalimat
- [ ] Buat fungsi `rank_sentences_textrank(G: nx.Graph) -> dict`:
  - Gunakan `nx.pagerank()` dengan parameter:
    - `alpha=0.85` (damping factor standar TextRank)
    - `weight='weight'` (gunakan bobot edge dari cosine similarity)
    - `max_iter=100`
  - Return dictionary: `{node_id: skor_pagerank}`
  - Tangani kasus edge case: jika graf kosong atau tidak terhubung, return skor uniform

### 5.2 — Fungsi Ekstraksi Top-N Kalimat
- [ ] Buat fungsi `extract_top_sentences(scores: dict, sentences: list, n=3) -> list`:
  - Urutkan `scores` dari skor tertinggi ke terendah: `sorted(scores, key=..., reverse=True)`
  - Ambil indeks N teratas
  - Urutkan ulang indeks tersebut secara **ascending** (agar urutan kalimat sesuai teks asli)
  - Kembalikan list kalimat berdasarkan indeks tersebut

### 5.3 — Fungsi Orkestrasi Ringkasan
- [ ] Buat fungsi `generate_summary(id_film: int, df_reviews: pd.DataFrame, sentiment_label: str) -> list`:
  - Filter `df_reviews` berdasarkan `id_film` dan `sentiment` yang sesuai
  - Jika jumlah ulasan < 3: langsung return ulasan mentah tanpa summarization
  - Panggil `segment_sentences()` → `build_sentence_vectors()` → `compute_similarity_matrix()`
  - Panggil `build_semantic_graph()` → `rank_sentences_textrank()` → `extract_top_sentences()`
  - Return list 3 kalimat ringkasan terbaik

### 5.4 — Tampilan Dual-Box Summary di Streamlit
- [ ] Di halaman detail film, setelah metadata:
  - Buat dua kolom: kiri untuk ringkasan positif, kanan untuk ringkasan negatif
  - Panggil `generate_summary(id_film, df_reviews, 'positive')` → simpan ke `summary_pos`
  - Panggil `generate_summary(id_film, df_reviews, 'negative')` → simpan ke `summary_neg`
  - **Panel Hijau (Positif)**:
    ```
    st.success("✅ Ringkasan Ulasan Positif")
    for i, kalimat in enumerate(summary_pos, 1):
        st.write(f"{i}. {kalimat}")
    ```
  - **Panel Merah (Negatif)**:
    ```
    st.error("❌ Ringkasan Ulasan Negatif")
    for i, kalimat in enumerate(summary_neg, 1):
        st.write(f"{i}. {kalimat}")
    ```
- [ ] Tangani kasus data kosong: tampilkan pesan "Belum ada ulasan [positif/negatif]"

### 5.5 — Tampilan Daftar Ulasan Mentah
- [ ] Di bagian bawah halaman detail, tampilkan semua ulasan dengan `st.expander()`
  - Judul expander: `f"📋 Semua Ulasan ({len(df_filtered)} ulasan)"`
  - Di dalam expander: tampilkan `st.dataframe()` dengan kolom `ulasan` dan `sentiment`
  - Tambahkan filter/search pada tabel (opsional menggunakan `st.selectbox()` untuk sentimen)

### 5.6 — Evaluasi Kualitas Ringkasan (ROUGE)
- [ ] Buat skrip atau seksi terpisah untuk evaluasi ROUGE:
  - **Buat Referensi Ringkasan Manual**: Tulis secara manual 3 ringkasan (positif & negatif) untuk 2-3 film sebagai "ground truth"
  - Simpan referensi di dictionary Python atau file JSON terpisah
  - **Hitung Skor ROUGE** menggunakan library `rouge-score`:
    ```python
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=False)
    scores = scorer.score(reference_summary, generated_summary)
    ```
  - Cetak hasil: `ROUGE-1 F1`, `ROUGE-2 F1` untuk setiap film
  - Hitung rata-rata skor ROUGE keseluruhan

---

## 🧪 Rencana Pengujian

### Unit Test Fungsi NLP
- [ ] Test `preprocess_text()`: input teks kotor → output teks bersih
- [ ] Test `segment_sentences()`: input list ulasan → output list kalimat valid
- [ ] Test `generate_summary()` dengan ulasan < 3 (edge case)
- [ ] Test `generate_summary()` dengan dataset film yang berbeda

### Integration Test Streamlit
- [ ] Test form input ulasan baru: prediksi sentimen tampil benar
- [ ] Test navigasi katalog → detail film → kembali
- [ ] Test apakah ulasan baru tersimpan ke `reviews.csv` setelah submit
- [ ] Test tampilan dual-box ringkasan setelah ulasan baru ditambahkan

---

## 📋 Ringkasan Urutan Eksekusi Skrip

| Urutan | Skrip / Aksi                   | Output                                          |
|--------|--------------------------------|-------------------------------------------------|
| 1      | `translator.py` (sudah ada)    | `dataset/IMDB Dataset_Indonesia.csv`            |
| 2      | `train_model.py`               | `model/model_svm.pkl`, `model/vectorizer.pkl`   |
| 3      | `prepare_database.py`          | `dataset/movies.csv`, `dataset/reviews.csv`     |
| 4      | `streamlit run main.py`        | Aplikasi web berjalan di `localhost:8501`        |

---

## 📦 Daftar Library yang Digunakan

| Library          | Kegunaan                                          |
|------------------|---------------------------------------------------|
| `pandas`         | Manipulasi DataFrame CSV                          |
| `scikit-learn`   | TF-IDF Vectorizer, SVM, train_test_split, metrics |
| `joblib`         | Simpan & muat model `.pkl`                        |
| `nltk`           | Sentence tokenization                             |
| `networkx`       | Pembangunan dan analisis graf semantik            |
| `streamlit`      | Framework antarmuka web interaktif                |
| `deep-translator`| Terjemahan teks Inggris → Indonesia              |
| `rouge-score`    | Evaluasi kualitas ringkasan (ROUGE-1, ROUGE-2)    |
| `tqdm`           | Progress bar untuk proses panjang                 |
| `re`             | Regex untuk pembersihan teks                      |
| `matplotlib`     | Visualisasi confusion matrix                      |
