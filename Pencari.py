import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sentence_transformers import SentenceTransformer
import time

# Setup halaman
st.set_page_config(
    page_title="Mesin Pencari Berita",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .search-container {
        background: rgba(255,255,255,0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    .result-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    .result-url {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 0.75rem;
        word-break: break-all;
    }
    .result-score {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: #e8f5e9;
        color: #2e7d32;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .result-snippet {
        color: #555;
        margin-top: 0.75rem;
        line-height: 1.5;
    }
    .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
    }
    @media (max-width: 768px) {
        .result-title {
            font-size: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Inisialisasi session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.df = None
    st.session_state.vectorizer = None
    st.session_state.tfidf_matrix = None
    st.session_state.processed_paper = None

@st.cache_resource
def load_data():
    """Load data yang sudah diproses"""
    try:
        # Coba load dari file yang sudah ada
        df = pd.read_excel("dataset_berita.xlsx")
        
        # Load processed papers
        with open("processed_paper.pkl", "rb") as f:
            processed_paper = pickle.load(f)
        
        # Load vectorizer
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
            tfidf_matrix = vectorizer.transform(processed_paper)
        
        return df, vectorizer, tfidf_matrix, processed_paper
    except:
        # Jika file belum ada, proses ulang
        st.warning("Data belum tersedia. Silakan upload file dataset_berita.xlsx")
        return None, None, None, None

@st.cache_resource
def load_preprocessing_components():
    """Load komponen preprocessing"""
    factory_stop = StopWordRemoverFactory()
    stopword = factory_stop.create_stop_word_remover()
    
    factory_stem = StemmerFactory()
    stemmer = factory_stem.create_stemmer()
    
    return stopword, stemmer

@st.cache_resource
def load_semantic_model():
    """Load model semantic"""
    try:
        model = SentenceTransformer('firqaaa/indo-sentence-bert-base')
        return model
    except:
        return None

def preprocess_query(query, stopword, stemmer):
    """Preprocessing query"""
    query = query.lower()
    query = re.sub(r'[^a-zA-Z\s]', '', query)
    query = stopword.remove(query)
    query = query.split()
    query = [stemmer.stem(w) for w in query]
    return query

def get_sinonim(kata):
    """Ambil sinonim dari web"""
    try:
        data = {"q": kata}
        encoded_data = urllib.parse.urlencode(data).encode("utf-8")
        content = urllib.request.urlopen("http://www.sinonimkata.com/search.php", encoded_data, timeout=5)
        soup = BeautifulSoup(content, 'html.parser')
        synonym = soup.find('td', attrs={'width': '90%'}).find_all('a')
        synonym = [x.getText() for x in synonym]
        return [kata] + synonym[:5]
    except:
        return [kata]

def filter_sinonim_semantik(kata, sinonim_list, query_words, model, top_n=3):
    """Filter sinonim berdasarkan semantic similarity"""
    if model is None:
        return sinonim_list[:top_n]
    
    query_asli = " ".join(query_words)
    
    try:
        emb_asli = model.encode([query_asli])
        
        hasil = [(kata, 1.0)]
        idx_kata = query_words.index(kata) if kata in query_words else 0
        
        for s in sinonim_list:
            if s == kata:
                continue
            
            query_variasi = query_words.copy()
            query_variasi[idx_kata] = s
            q_variasi_str = " ".join(query_variasi)
            
            emb_variasi = model.encode([q_variasi_str])
            skor = cosine_similarity(emb_asli, emb_variasi)[0][0]
            
            if skor >= 0.6:
                hasil.append((s, skor))
        
        hasil = sorted(hasil, key=lambda x: x[1], reverse=True)
        return [x[0] for x in hasil[:top_n]]
    except:
        return sinonim_list[:top_n]

def query_expansion(query_words, stopword, stemmer, model=None):
    """Query expansion otomatis"""
    list_synonym = []
    
    for kata in query_words:
        sinonim = get_sinonim(kata)
        if model:
            sinonim = filter_sinonim_semantik(kata, sinonim, query_words, model)
        else:
            sinonim = sinonim[:3]
        list_synonym.append(sinonim)
    
    qs = set()
    
    # Kombinasi sinonim
    for i, kata in enumerate(query_words):
        for s in list_synonym[i]:
            kombinasi = query_words.copy()
            kombinasi[i] = s
            qs.add(' '.join(kombinasi))
    
    qs.add(' '.join(query_words))
    
    return qs

def search_documents(query, df, vectorizer, tfidf_matrix, processed_paper, stopword, stemmer, model=None):
    """Fungsi pencarian utama"""
    # Preprocessing query
    processed_query = preprocess_query(query, stopword, stemmer)
    
    if not processed_query:
        return []
    
    # Query expansion
    expanded_queries = query_expansion(processed_query, stopword, stemmer, model)
    
    # Pencarian dengan semua query yang sudah diexpand
    all_results = []
    
    for q in expanded_queries:
        try:
            q_vec = vectorizer.transform([q])
            result = cosine_similarity(tfidf_matrix, q_vec)
            
            for i in range(len(result)):
                score = result[i][0]
                if score > 0.05:  # Threshold rendah untuk lebih banyak hasil
                    all_results.append({
                        'index': i,
                        'score': score,
                        'query_used': q
                    })
        except:
            continue
    
    # Hapus duplikat dan urutkan berdasarkan score
    unique_results = {}
    for item in all_results:
        if item['index'] not in unique_results or unique_results[item['index']]['score'] < item['score']:
            unique_results[item['index']] = item
    
    sorted_results = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)
    
    # Ambil data dokumen
    results = []
    for item in sorted_results:
        i = item['index']
        if i < len(df):
            results.append({
                'judul': df['Judul'].iloc[i],
                'url': df['URL'].iloc[i],
                'score': item['score'],
                'snippet': df['Text'].iloc[i][:300] + '...' if len(df['Text'].iloc[i]) > 300 else df['Text'].iloc[i]
            })
    
    return results

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🔍 Mesin Pencari Berita</h1>
        <p style="color: white; margin: 0.5rem 0 0 0;">Temukan berita terbaru dengan mudah dan cepat</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data dan komponen
    with st.spinner("Memuat sistem pencarian..."):
        df, vectorizer, tfidf_matrix, processed_paper = load_data()
        stopword, stemmer = load_preprocessing_components()
        semantic_model = load_semantic_model()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-content">
            <h3>📊 Statistik</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if df is not None:
            st.metric("Jumlah Berita", len(df))
            st.metric("Topik Berita", df['Judul'].nunique())
        
        st.markdown("---")
        st.markdown("""
        <div class="sidebar-content">
            <h3>💡 Tips Pencarian</h3>
            <ul>
                <li>Gunakan kata kunci yang spesifik</li>
                <li>Sistem akan otomatis mencari sinonim</li>
                <li>Hasil diurutkan berdasarkan relevansi</li>
                <li>Semakin panjang query, semakin akurat</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption("© 2024 Mesin Pencari Berita | Dibangun dengan Streamlit")
    
    # Main content
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        
        # Search box
        query = st.text_input(
            "🔎 Masukkan kata kunci pencarian",
            placeholder="Contoh: harga emas, politik indonesia, teknologi terbaru...",
            label_visibility="collapsed"
        )
        
        search_button = st.button("Cari Berita", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Proses pencarian
    if search_button and query:
        if df is not None and vectorizer is not None:
            with st.spinner("🔍 Mencari berita yang relevan..."):
                time.sleep(0.5)
                results = search_documents(query, df, vectorizer, tfidf_matrix, processed_paper, stopword, stemmer, semantic_model)
            
            if results:
                st.success(f"✅ Ditemukan {len(results)} berita relevan")
                
                # Tampilkan hasil
                for idx, result in enumerate(results, 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="result-title">{idx}. {result['judul']}</div>
                            <div class="result-url">🔗 {result['url']}</div>
                            <div class="result-snippet">{result['snippet']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Tombol untuk membuka link
                        if st.button(f"Baca Selengkapnya", key=f"btn_{idx}"):
                            st.markdown(f'<a href="{result["url"]}" target="_blank">Klik di sini untuk membaca berita lengkap</a>', unsafe_allow_html=True)
            else:
                st.warning("😔 Tidak ditemukan berita yang sesuai. Coba kata kunci lain.")
        else:
            st.error("❌ Data belum tersedia. Silakan upload file dataset terlebih dahulu.")
    
    elif search_button and not query:
        st.warning("⚠️ Silakan masukkan kata kunci pencarian terlebih dahulu.")
    
    # Tampilkan beberapa berita terbaru jika belum mencari
    if not search_button and df is not None:
        st.markdown("### 📰 Berita Terbaru")
        
        cols = st.columns(2)
        for idx, (_, row) in enumerate(df.head(6).iterrows()):
            with cols[idx % 2]:
                with st.container():
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-title">{row['Judul'][:100]}...</div>
                        <div class="result-url">🔗 {row['URL'][:80]}...</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Baca", key=f"preview_{idx}"):
                        st.markdown(f'<a href="{row["URL"]}" target="_blank">Klik di sini untuk membaca</a>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
