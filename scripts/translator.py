import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm
import time

def translate_movie_reviews(input_file, output_file, max_rows=1000):
    # 1. Membaca dataset ulasan film bahasa Inggris (hanya 1000 baris pertama)
    try:
        df = pd.read_csv(input_file, nrows=max_rows)
        print(f"Berhasil memuat dataset. Total data: {len(df)} baris (maksimal {max_rows} baris).")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' tidak ditemukan. Pastikan path file sudah benar.")
        return

    # Validasi keberadaan kolom wajib
    if 'review' not in df.columns or 'sentiment' not in df.columns:
        print("Error: Dataset harus memiliki kolom 'review' dan 'sentiment'!")
        return

    # 2. Inisialisasi Google Translator (Dari English ke Indonesian)
    translator = GoogleTranslator(source='en', target='id')

    # Mengaktifkan progress bar dari tqdm untuk Pandas DataFrame
    tqdm.pandas()

    # Fungsi pembantu untuk translasi dengan proteksi error handling
    def do_translation(text):
        if not isinstance(text, str) or text.strip() == "":
            return ""
        
        # Google Translate membatasi maksimal 5000 karakter per request.
        # Jika teks ulasan terlalu panjang, kita potong ke 4500 karakter aman.
        if len(text) > 4500:
            text = text[:4500]
            
        try:
            # Eksekusi translasi
            return translator.translate(text)
        except Exception as e:
            # Jika terjadi gangguan koneksi atau limit, jeda 2 detik lalu coba lagi sekali
            time.sleep(2)
            try:
                return translator.translate(text)
            except:
                # Jika tetap gagal, kembalikan teks asli agar baris data tidak hilang
                return text

    print("\nMemulai proses translasi ke Bahasa Indonesia...")
    print("Proses ini memakan waktu tergantung jumlah data dan kecepatan internet Anda.\n")

    # 3. Membuat kolom baru 'ulasan' hasil terjemahan dari kolom 'review'
    # Kolom 'sentiment' tetap utuh apa adanya (positive/negative)
    df['ulasan'] = df['review'].progress_apply(do_translation)

    # 4. Menyimpan DataFrame hasil akhir ke file CSV baru
    # Kita simpan kolom penting saja agar struktur database rapi sesuai PRD
    kolom_output = ['ulasan', 'sentiment']
    
    # Jika di data awal Anda ada kolom id_film atau tconst, masukkan ke dalam daftar di bawah ini:
    if 'id_film' in df.columns:
        kolom_output.insert(0, 'id_film')
    elif 'tconst' in df.columns:
        kolom_output.insert(0, 'tconst')
    if 'movie_title' in df.columns:
        kolom_output.insert(1, 'movie_title')

    df_final = df[kolom_output]
    df_final.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\nProses selesai! Data berhasil disimpan di: '{output_file}'")

# --- Cara Menjalankan Skrip ---
if __name__ == "__main__":
    # Sesuaikan 'nama_file_input.csv' dengan nama file asli bahasa Inggris Anda
    FILE_INPUT = "dataset/IMDB Dataset.csv" 
    FILE_OUTPUT = "dataset/IMDB Dataset_Indonesia.csv"
    
    translate_movie_reviews(FILE_INPUT, FILE_OUTPUT)