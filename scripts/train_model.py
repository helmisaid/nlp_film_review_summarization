import os
import sys
import pandas as pd

# Menambahkan root project ke sys.path agar modul src bisa di-import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import clean_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_verify_data(file_path):
    # 1.1.1 — Memuat Dataset Terjemahan
    print(f"Membaca dataset dari: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File tidak ditemukan di {file_path}")
        return None
        
    df = pd.read_csv(file_path)
    
    # 1.1.2 — Menampilkan informasi awal dataset
    print("\n--- Informasi Awal Dataset ---")
    print(f"Jumlah baris: {df.shape[0]}")
    print(f"Jumlah kolom: {df.shape[1]}")
    print(f"Nama kolom  : {df.columns.tolist()}")
    
    # 1.1.4 — Menangani nilai kosong/null
    df = df.dropna(subset=['ulasan'])
    return df

def preprocess_dataset(df):
    # 1.2 — Pra-Pemrosesan Teks untuk Training
    print("\n--- Memulai Tahap 1.2: Pra-Pemrosesan Teks ---")
    print("Memproses ulasan...")
    
    # Melakukan clean_text ke kolom baru 'ulasan_bersih'
    df['ulasan_bersih'] = df['ulasan'].apply(clean_text)
    
    print("Pra-pemrosesan selesai!")
    print("\nPerbandingan 3 Ulasan Sebelum vs Sesudah Pembersihan:")
    for idx, row in df.head(3).iterrows():
        print(f"\n[Ulasan Asli #{idx+1}]:")
        print(row['ulasan'][:150] + "...")
        print(f"[Ulasan Bersih #{idx+1}]:")
        print(row['ulasan_bersih'][:150] + "...")
        print("-" * 50)
        
    return df

def encode_labels(df):
    # 1.3 — Encoding Label
    print("\n--- Memulai Tahap 1.3: Encoding Label ---")
    if 'sentiment' not in df.columns:
        raise ValueError("Kolom 'sentiment' tidak ditemukan!")
    
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    print("Encoding selesai! Distribusi label baru:")
    print(df['label'].value_counts())
    return df

if __name__ == "__main__":
    DATA_PATH = os.path.join("data", "processed", "IMDB Dataset_Indonesia.csv")
    OUTPUT_PATH = os.path.join("data", "processed", "IMDB_Indonesia_Clean.csv")
    
    df = load_and_verify_data(DATA_PATH)
    if df is not None:
        df = preprocess_dataset(df)
        df = encode_labels(df)
        
        # Menyimpan hasil pra-pemrosesan & encoding ke file CSV baru
        print(f"\nMenyimpan hasil ke: {OUTPUT_PATH}")
        df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
        print("Penyimpanan selesai!")
        
        # 1.4 — Ekstraksi Fitur dengan TF-IDF (Inisialisasi)
        print("\n--- Memulai Tahap 1.4: Ekstraksi Fitur (TF-IDF Vectorizer) ---")
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
        X = df['ulasan_bersih']
        y = df['label']
        print("TF-IDF Vectorizer berhasil diinisialisasi.")
        
        # 1.5 — Split Data Training dan Testing
        print("\n--- Memulai Tahap 1.5: Split & Transformasi Data ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Data Training (X_train): {X_train.shape}, Data Testing (X_test): {X_test.shape}")
        
        # Fit HANYA pada data training untuk menghindari data leakage
        print("Melakukan fitting TF-IDF pada X_train...")
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        print(f"Bentuk representasi vectorizer X_train_vec: {X_train_vec.shape}")
        print(f"Bentuk representasi vectorizer X_test_vec: {X_test_vec.shape}")
        
        # 1.6 — Pelatihan Model SVM (Konfigurasi Final)
        print("\n--- Memulai Tahap 1.6: Pelatihan Model SVM ---")
        print("Melatih model SVM (Linear Kernel, C=1.0)...")
        model_svm = SVC(kernel='linear', C=1.0)
        model_svm.fit(X_train_vec, y_train)
        print("Pelatihan model SVM selesai!")
        
        # 1.7 — Evaluasi Model
        print("\n--- Memulai Tahap 1.7: Evaluasi Model ---")
        y_pred = model_svm.predict(X_test_vec)
        
        # Klasifikasi laporan (Accuracy, Precision, Recall, F1-Score)
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        
        # Visualisasi Confusion Matrix dengan Seaborn & Matplotlib
        print("Membuat visualisasi Confusion Matrix...")
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['negative', 'positive'],
                    yticklabels=['negative', 'positive'])
        plt.title('Confusion Matrix - Sentimen Ulasan Film')
        plt.ylabel('Aktual (Ground Truth)')
        plt.xlabel('Prediksi Model')
        
        # Simpan plot ke direktori docs/
        os.makedirs("docs", exist_ok=True)
        plot_path = os.path.join("docs", "confusion_matrix.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Visualisasi Confusion Matrix disimpan ke: {plot_path}")
        
        # 1.8 — Penyimpanan Model dan Vectorizer
        print("\n--- Memulai Tahap 1.8: Penyimpanan Model dan Vectorizer ---")
        os.makedirs("model", exist_ok=True)
        
        model_file = "model/model_svm.pkl"
        vec_file = "model/vectorizer.pkl"
        
        joblib.dump(model_svm, model_file)
        joblib.dump(vectorizer, vec_file)
        print(f"Model SVM berhasil disimpan ke: {model_file}")
        print(f"Vectorizer TF-IDF berhasil disimpan ke: {vec_file}")
        
        # Verifikasi ukuran file
        print(f"Ukuran file {model_file}: {os.path.getsize(model_file) / 1024:.2f} KB")
        print(f"Ukuran file {vec_file}: {os.path.getsize(vec_file) / 1024:.2f} KB")
        print("\nSeluruh rangkaian Tahap 1 berhasil diselesaikan!")





