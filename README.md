# 🎬 Sistem Peringkasan Ulasan Film Berbasis Graf Semantik & Klasifikasi Sentimen Real-Time
> **Tugas Besar Proyek UAS Data Mining — Kelompok 7 (D4 Teknik Informatika, Universitas Airlangga)**

Repositori ini berisi kode sumber untuk aplikasi web katalog film berbasis **Streamlit** yang memprediksi sentimen ulasan secara *real-time* menggunakan **SVM (Support Vector Machine)**, serta melakukan peringkasan ekstraktif ulasan menggunakan algoritma **TextRank (Graf Semantik)**.

---

## 🚀 Panduan Memulai (Setelah Clone Repositori)

Ikuti langkah-langkah di bawah ini secara berurutan untuk menyiapkan lingkungan kerja di komputer lokal Anda:

### 1. Masuk ke Direktori Proyek
Buka terminal/CMD/PowerShell Anda, lalu masuk ke folder hasil clone:
```bash
cd nlp_film_reviews_summarization
```

### 2. Buat & Aktifkan Virtual Environment (`venv`)
Buat lingkungan terisolasi agar versi library tidak bentrok dengan sistem global Anda:

* **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
* **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
*(Setelah aktif, Anda akan melihat tanda `(venv)` di sebelah kiri prompt terminal Anda)*

### 3. Instalasi Dependensi Pustaka
Instal semua library Python yang dibutuhkan melalui file `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Unduh Resource NLTK
NLTK membutuhkan file data pendukung untuk tokenisasi kalimat. Unduh dengan menjalankan perintah satu baris ini di terminal:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## 🛠️ Alur Kerja Skrip (Pipeline)

Setelah penyiapan lingkungan selesai, berikut cara menjalankan komponen-komponen proyek:

### 1. (Opsional) Menerjemahkan Dataset Baru
Jika ingin menerjemahkan ulasan baru dari bahasa Inggris ke Indonesia (secara default sudah disediakan hasil terjemahan 10.000 ulasan di folder `data/processed/`):
```bash
python scripts/translator.py
```

### 2. Melatih Model Klasifikasi SVM
Untuk melatih kembali model SVM menggunakan dataset 10.000 ulasan yang ada di `data/processed/`, mengevaluasi akurasinya, dan mengekspor file biner model (`.pkl`):
```bash
python scripts/train_model.py
```
*Skrip ini akan menyimpan visualisasi Confusion Matrix di `docs/confusion_matrix.png` dan berkas model di folder `model/`.*

### 3. Menjalankan Aplikasi Web Streamlit
Setelah model SVM terlatih (`model_svm.pkl` & `vectorizer.pkl`) berhasil dibuat, Anda bisa menjalankan aplikasi web interaktifnya:
```bash
streamlit run main.py
```
Aplikasi otomatis akan terbuka di browser Anda pada alamat `http://localhost:8501`.

---

## 📁 Struktur Folder Proyek

```
nlp_film_reviews_summarization/
│
├── 📂 data/                                    ← Semua berkas data CSV
│   ├── raw/                                    ← Dataset mentah (IMDB Dataset.csv)
│   └── processed/                              ← Dataset bersih & hasil olahan
│
├── 📂 model/                                   ← File biner model ML (.pkl)
│
├── 📂 src/                                     ← Modul NLP utama (modular & reusable)
│   ├── preprocess.py                           ← Pembersihan teks & stopwords Sastrawi
│   ├── graph.py                                ← Pembentukan graf semantik NetworkX
│   └── summarizer.py                           ← Algoritma TextRank & Peringkasan
│
├── 📂 scripts/                                 ← Skrip eksekusi pipeline offline
│   ├── translator.py                           ← Penerjemah dataset
│   └── train_model.py                          ← Pelatihan & evaluasi model SVM
│
├── 📂 docs/                                    ← Dokumentasi & gambar evaluasi
│
├── main.py                                     ← Kode utama aplikasi web Streamlit
├── WORKFLOW.md                                 ← Detail rincian langkah pengerjaan
└── requirements.txt                            ← Daftar dependensi library Python
```
