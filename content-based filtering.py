import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi Halaman
st.set_page_config(page_title="Music Recommender", layout="centered")

@st.cache_data
def load_data():
    # Pastikan file music.csv berada di folder yang sama
    df = pd.read_csv('music.csv')
    # Preprocessing genres (sama seperti di notebook)
    df['genres_processed'] = df['genres'].fillna('').apply(lambda x: x.replace('|', ' '))
    return df

@st.cache_resource
def calculate_similarity(df):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['genres_processed'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_recommendations(title, df, cosine_sim_matrix, num_recommendations=10):
    try:
        # Mendapatkan index musik berdasarkan judul
        idx = df[df['title'] == title].index[0]
        
        # Menghitung skor kemiripan
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Mengambil n rekomendasi teratas (mengecualikan diri sendiri)
        sim_scores = sim_scores[1:num_recommendations+1]
        music_indices = [i[0] for i in sim_scores]
        
        return df[['title', 'genres', 'year']].iloc[music_indices]
    except IndexError:
        return None

# --- UI Streamlit ---

st.title("🎵 Music Recommendation System")
st.markdown("Dapatkan rekomendasi lagu berdasarkan kemiripan genre menggunakan *Cosine Similarity*.")

# Load Data
df_music = load_data()
cosine_sim = calculate_similarity(df_music)

# Input User
st.subheader("Cari Lagu Favorit Anda")
music_list = df_music['title'].values
selected_music = st.selectbox("Pilih atau ketik judul lagu:", music_list)

num_rec = st.slider("Jumlah rekomendasi:", 1, 20, 10)

if st.button("Tampilkan Rekomendasi"):
    recommendations = get_recommendations(selected_music, df_music, cosine_sim, num_rec)
    
    if recommendations is not None:
        st.success(f"Lagu yang mirip dengan **'{selected_music}'**:")
        
        # Menampilkan hasil dalam tabel yang bersih
        st.table(recommendations.reset_index(drop=True))
    else:
        st.error("Maaf, lagu tidak ditemukan.")

# Informasi Tambahan di Sidebar
st.sidebar.header("Tentang")
st.sidebar.info("Aplikasi ini menggunakan TF-IDF Vectorizer untuk memproses genre musik dan Cosine Similarity untuk mencari kemiripan antar lagu.")