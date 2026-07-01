import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Inisialisasi stopword Sastrawi
factory = StopWordRemoverFactory()
stop_words = set(factory.get_stop_words())

# Menambahkan kata kustom tambahan yang sering muncul tapi kurang bermakna
custom_stopwords = {'yg', 'dg', 'dr', 'dgn', 'saja', 'movie', 'film', 'br', 'nya', 'bahwa', 'ke'}
stop_words.update(custom_stopwords)

# Mengeluarkan kata-kata negasi penting dari stopwords agar tidak dihapus (Negation Handling)
negation_words = {'tidak', 'bukan', 'tanpa', 'kurang', 'jangan', 'tiada', 'belum', 'tak', 'tidaklah', 'bukanlah'}
stop_words = stop_words - negation_words

def clean_text(text):
    """
    Melakukan pra-pemrosesan teks (preprocessing) Bahasa Indonesia:
    1. Menghapus tag HTML (seperti <br />)
    2. Case folding (mengubah ke huruf kecil)
    3. Pembersihan karakter non-alfabet (menyisakan huruf a-z dan spasi)
    4. Penghapusan stopwords (kata-kata umum yang kurang informatif)
    5. Menghapus spasi berlebih
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Hapus tag HTML
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 2. Case folding
    text = text.lower()
    
    # 3. Hapus karakter selain a-z dan spasi (angka, tanda baca, simbol)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 4. Tokenisasi sederhana & Stopwords Removal
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words and len(word) > 1]
    
    # 5. Gabungkan kembali dan bersihkan spasi berlebih
    return " ".join(cleaned_words)
