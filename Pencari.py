import streamlit as st
import pandas as pd
import numpy as np
import re
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import io
import warnings
warnings.filterwarnings('ignore')

# Setup halaman
st.set_page_config(
    page_title="Mesin Pencari Berita Kompas",
    page_icon="🔍",
    layout="centered"
)

# Title
st.title("🔍 Mesin Pencari Berita Kompas.com")
st.markdown("---")

# === RAW URL DARI GITHUB (OTOMATIS LOAD) ===
RAW_GITHUB_URL = "https://raw.githubusercontent.com/Wenda23/Mesin-Pencari-Berita-Kompas.com/main/dataset_berita.xlsx"

# Inisialisasi session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'tfidf_matrix' not in st.session_state:
    st.session_state.tfidf_matrix = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'total_berita' not in st.session_state:
    st.session_state.total_berita = 0

@st.cache_resource
def load_and_process_dataset():
    """Load dataset dari GitHub dan proses secara otomatis"""
    try:
        # Load dataset dari GitHub
        df = pd.read_excel(RAW_GITHUB_URL)
        
        # Validasi kolom
        required = ['No', 'URL', 'Judul', 'Text']
        if not all(col in df.columns for col in required):
            st.error(f"Dataset harus memiliki kolom: {', '.join(required)}")
            return None, None, None, False
        
        # Preprocessing
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = re.sub(r'\s+', ' ', text)
            return text
        
        # Stopword & Stemmer
        factory_stop = StopWordRemoverFactory()
        stopword = factory_stop.create_stop_word_remover()
        
        factory_stem = StemmerFactory()
        stemmer = factory_stem.create_stemmer()
        
        # Proses teks
        df["clean_text"] = df["Text"].apply(clean_text)
        df["clean_text"] = df["clean_text"].apply(lambda x: stopword.remove(x))
        df["clean_text"] = df["clean_text"].apply(lambda x: stemmer.stem(x))
        
        # TF-IDF
        processed_paper = df["clean_text"].tolist()
        vectorizer = TfidfVectorizer(use_idf=True)
        tfidf_matrix = vectorizer.fit_transform(processed_paper)
        
        return df, vectorizer, tfidf_matrix, True
        
    except Exception as e:
        st.error(f"Gagal load dataset: {str(e)}")
        return None, None, None, False

# Fungsi query expansion
@st.cache_data
def get_synonyms(word):
    try:
        data = {"q": word}
        encoded = urllib.parse.urlencode(data).encode("utf-8")
        req = urllib.request.Request(
            "http://www.sinonimkata.com/search.php",
            data=encoded,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        content = urllib.request.urlopen(req, timeout=3)
        soup = BeautifulSoup(content, 'html.parser')
        synonym = soup.find('td', attrs={'width': '90%'})
        if synonym:
            words = synonym.find_all('a')
            return [word] + [w.getText() for w in words[:2]]
        return [word]
    except:
        return [word]

def expand_query(query_words):
    expanded = set()
    expanded.add(' '.join(query_words))
    
    for i, word in enumerate(query_words):
        synonyms = get_synonyms(word)
        for syn in synonyms[:2]:
            if syn != word:
                new_query = query_words.copy()
                new_query[i] = syn
                expanded.add(' '.join(new_query))
    
    return expanded

# LOAD DATA OTOMATIS SAAT APLIKASI PERTAMA KALI DIJALANKAN
if not st.session_state.data_loaded:
    with st.spinner("📡 Mengunduh dan memproses dataset dari GitHub..."):
        df, vectorizer, tfidf_matrix, success = load_and_process_dataset()
        
        if success:
            st.session_state.df = df
            st.session_state.vectorizer = vectorizer
            st.session_state.tfidf_matrix = tfidf_matrix
            st.session_state.data_loaded = True
            st.session_state.total_berita = len(df)
            st.rerun()

# Sidebar untuk informasi
with st.sidebar:
    st.header("ℹ️ Informasi")
    
    if st.session_state.data_loaded:
        st.success(f"✅ **{st.session_state.total_berita}** berita terindex")
        st.caption(f"Dataset dari: GitHub")
        st.caption(f"Update: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
        
        st.markdown("---")
        st.markdown("### 💡 Tips Pencarian")
        st.markdown("""
        - Gunakan kata kunci spesifik
        - Sistem otomatis mencari sinonim
        - Hasil diurutkan berdasarkan relevansi
        - Semakin panjang query, semakin akurat
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Statistik Cepat")
        
        # Hitung statistik sederhana
        if st.session_state.df is not None:
            # Kata kunci populer (dari judul)
            all_titles = ' '.join(st.session_state.df['Judul'].tolist())
            words = all_titles.lower().split()
            from collections import Counter
            common = Counter(words).most_common(5)
            st.markdown("**Top kata kunci:**")
            for word, count in common:
                st.caption(f"- {word}: {count}x")
    else:
        st.error("❌ Gagal memuat data")
        st.info("Pastikan file dataset_berita.xlsx ada di repository GitHub")
    
    st.markdown("---")
    st.caption("© 2024 Mesin Pencari Berita Kompas.com")

# Main content
if st.session_state.data_loaded:
    # Search box
    st.markdown("### 🔎 Cari Berita")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Kata kunci",
            placeholder="Contoh: harga emas, politik, prabowo, kpk, teknologi...",
            label_visibility="collapsed"
        )
    with col2:
        search_clicked = st.button("🔍 Cari", type="primary", use_container_width=True)
    
    # Contoh query cepat
    st.markdown("**✨ Contoh pencarian:**")
    contoh_cols = st.columns(5)
    contoh_queries = ["harga emas", "politik", "prabowo", "kpk", "teknologi"]
    
    for idx, contoh in enumerate(contoh_queries):
        with contoh_cols[idx]:
            if st.button(contoh, key=f"contoh_{idx}"):
                query = contoh
                search_clicked = True
    
    st.markdown("---")
    
    # Proses pencarian
    if search_clicked and query:
        with st.spinner("🔍 Mencari berita yang relevan..."):
            # Preprocessing query
            factory_stop = StopWordRemoverFactory()
            stopword = factory_stop.create_stop_word_remover()
            factory_stem = StemmerFactory()
            stemmer = factory_stem.create_stemmer()
            
            # Proses query
            query_lower = query.lower()
            query_clean = re.sub(r'[^a-zA-Z\s]', '', query_lower)
            query_no_stop = stopword.remove(query_clean)
            query_words = [stemmer.stem(w) for w in query_no_stop.split() if w.strip()]
            
            if query_words:
                # Query expansion
                expanded_queries = expand_query(query_words)
                
                # Cari dengan semua query expansion
                all_results = {}
                
                for eq in expanded_queries:
                    try:
                        q_vec = st.session_state.vectorizer.transform([eq])
                        similarities = cosine_similarity(st.session_state.tfidf_matrix, q_vec)
                        
                        for i in range(len(similarities)):
                            score = similarities[i][0]
                            if score > 0.05:
                                if i not in all_results or all_results[i] < score:
                                    all_results[i] = score
                    except:
                        continue
                
                # Urutkan hasil
                sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
                
                if sorted_results:
                    st.success(f"✅ Ditemukan {len(sorted_results)} berita relevan untuk: **{query}**")
                    st.markdown("---")
                    
                    # Tampilkan hasil
                    for idx, (doc_id, score) in enumerate(sorted_results[:30], 1):
                        row = st.session_state.df.iloc[doc_id]
                        
                        with st.container():
                            # Warna skor berdasarkan relevansi
                            if score >= 0.3:
                                score_color = "🟢"
                            elif score >= 0.15:
                                score_color = "🟡"
                            else:
                                score_color = "🟠"
                            
                            st.markdown(f"**{idx}. {row['Judul']}**")
                            st.caption(f"🔗 {row['URL']}")
                            st.markdown(f"{score_color} **Skor relevansi:** `{score:.4f}`")
                            
                            # Snippet
                            snippet = str(row['Text'])[:250] + "..." if len(str(row['Text'])) > 250 else str(row['Text'])
                            st.markdown(f"📝 {snippet}")
                            st.markdown(f"🔗 [Baca selengkapnya]({row['URL']})", unsafe_allow_html=True)
                            st.markdown("---")
                else:
                    st.warning(f"😔 Tidak ada berita yang cocok dengan '{query}'. Coba kata kunci lain.")
            else:
                st.warning("⚠️ Masukkan kata kunci yang valid")
    
    elif search_clicked and not query:
        st.warning("⚠️ Masukkan kata kunci pencarian terlebih dahulu")
    
    # Tampilkan beberapa berita terbaru jika belum mencari
    if not search_clicked:
        st.markdown("### 📰 Berita Terbaru dalam Database")
        st.caption(f"Menampilkan 5 dari {st.session_state.total_berita} berita yang tersedia")
        st.markdown("---")
        
        for idx, (_, row) in enumerate(st.session_state.df.head(5).iterrows()):
            with st.container():
                st.markdown(f"**{idx+1}. {row['Judul'][:100]}**")
                st.caption(f"🔗 {row['URL'][:80]}...")
                st.markdown(f"[📖 Baca berita]({row['URL']})", unsafe_allow_html=True)
                st.markdown("---")

else:
    # Jika data gagal load
    st.error("❌ Gagal memuat dataset dari GitHub")
    st.markdown("""
    ### 🔧 Solusi:
    
    1. **Periksa URL dataset:**
