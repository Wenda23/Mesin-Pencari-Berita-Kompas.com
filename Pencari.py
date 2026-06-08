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
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

# ==================== KONFIGURASI ====================
st.set_page_config(
    page_title="Mesin Pencari Berita Kompas",
    page_icon="🔍",
    layout="centered"
)

# RAW URL DARI GITHUB
RAW_GITHUB_URL = "https://raw.githubusercontent.com/Wenda23/Mesin-Pencari-Berita-Kompas.com/main/dataset_berita.xlsx"

# ==================== SESSION STATE ====================
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

# ==================== FUNGSI PREPROCESSING ====================
def clean_text(text):
    """Membersihkan teks"""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def get_stopword_stemmer():
    """Mendapatkan objek stopword dan stemmer"""
    factory_stop = StopWordRemoverFactory()
    stopword = factory_stop.create_stop_word_remover()
    factory_stem = StemmerFactory()
    stemmer = factory_stem.create_stemmer()
    return stopword, stemmer

def process_text_series(series, stopword, stemmer):
    """Proses text series menjadi clean text"""
    cleaned = series.apply(clean_text)
    cleaned = cleaned.apply(lambda x: stopword.remove(x))
    cleaned = cleaned.apply(lambda x: stemmer.stem(x))
    return cleaned

# ==================== FUNGSI LOAD DATASET ====================
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
        
        # Proses teks
        stopword, stemmer = get_stopword_stemmer()
        df["clean_text"] = process_text_series(df["Text"], stopword, stemmer)
        
        # TF-IDF
        processed_paper = df["clean_text"].tolist()
        vectorizer = TfidfVectorizer(use_idf=True)
        tfidf_matrix = vectorizer.fit_transform(processed_paper)
        
        return df, vectorizer, tfidf_matrix, True
        
    except Exception as e:
        st.error(f"Gagal load dataset: {str(e)}")
        return None, None, None, False

def process_uploaded_file(uploaded_file):
    """Proses file yang diupload manual"""
    try:
        df_temp = pd.read_excel(uploaded_file)
        required = ['No', 'URL', 'Judul', 'Text']
        
        if not all(col in df_temp.columns for col in required):
            st.error(f"Kolom yang dibutuhkan: {', '.join(required)}")
            return None, None, None, False
        
        # Proses teks
        stopword, stemmer = get_stopword_stemmer()
        df_temp["clean_text"] = process_text_series(df_temp["Text"], stopword, stemmer)
        
        # TF-IDF
        processed_paper = df_temp["clean_text"].tolist()
        vectorizer = TfidfVectorizer(use_idf=True)
        tfidf_matrix = vectorizer.fit_transform(processed_paper)
        
        return df_temp, vectorizer, tfidf_matrix, True
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None, None, False

# ==================== FUNGSI QUERY EXPANSION ====================
@st.cache_data
def get_synonyms(word):
    """Mencari sinonim kata dari web"""
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
    """Memperluas query dengan sinonim"""
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

# ==================== FUNGSI PENCARIAN ====================
def search_documents(query, df, vectorizer, tfidf_matrix):
    """Mencari dokumen yang relevan dengan query"""
    stopword, stemmer = get_stopword_stemmer()
    
    # Preprocessing query
    query_lower = query.lower()
    query_clean = re.sub(r'[^a-zA-Z\s]', '', query_lower)
    query_no_stop = stopword.remove(query_clean)
    query_words = [stemmer.stem(w) for w in query_no_stop.split() if w.strip()]
    
    if not query_words:
        return []
    
    # Query expansion
    expanded_queries = expand_query(query_words)
    
    # Cari dengan semua query expansion
    all_results = {}
    
    for eq in expanded_queries:
        try:
            q_vec = vectorizer.transform([eq])
            similarities = cosine_similarity(tfidf_matrix, q_vec)
            
            for i in range(len(similarities)):
                score = similarities[i][0]
                if score > 0.05:
                    if i not in all_results or all_results[i] < score:
                        all_results[i] = score
        except:
            continue
    
    # Urutkan hasil
    sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
    
    # Ambil data dokumen
    results = []
    for doc_id, score in sorted_results[:30]:
        row = df.iloc[doc_id]
        results.append({
            'judul': row['Judul'],
            'url': row['URL'],
            'score': score,
            'snippet': str(row['Text'])[:250] + '...' if len(str(row['Text'])) > 250 else str(row['Text'])
        })
    
    return results

# ==================== TAMPILAN UTAMA ====================
st.title("🔍 Mesin Pencari Berita Kompas.com")
st.markdown("---")

# AUTO LOAD DATASET DARI GITHUB
if not st.session_state.data_loaded:
    with st.spinner("📡 Mengunduh dan memproses dataset dari GitHub..."):
        df, vectorizer, tfidf_matrix, success = load_and_process_dataset()
        
        if success and df is not None:
            st.session_state.df = df
            st.session_state.vectorizer = vectorizer
            st.session_state.tfidf_matrix = tfidf_matrix
            st.session_state.data_loaded = True
            st.session_state.total_berita = len(df)
            st.rerun()

# ==================== SIDEBAR (DIPERKECIL/DIHAPUS) ====================
with st.sidebar:
    st.header("⚙️ Pengaturan")
    
    if st.session_state.data_loaded:
        # Hanya tampilkan jumlah berita saja (tanpa tips)
        st.metric("📊 Jumlah Berita", f"{st.session_state.total_berita}")
        
        st.markdown("---")
        
        # Tombol reset data
        if st.button("🔄 Reset Data", use_container_width=True):
            st.session_state.data_loaded = False
            st.cache_resource.clear()
            st.rerun()
    
    st.markdown("---")
    st.caption("© 2024 Mesin Pencari Berita")

# ==================== MAIN CONTENT ====================
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
            results = search_documents(
                query, 
                st.session_state.df, 
                st.session_state.vectorizer, 
                st.session_state.tfidf_matrix
            )
        
        if results:
            st.success(f"✅ Ditemukan {len(results)} berita relevan untuk: **{query}**")
            st.markdown("---")
            
            # Tampilkan hasil
            for idx, result in enumerate(results, 1):
                with st.container():
                    # Warna skor berdasarkan relevansi
                    if result['score'] >= 0.3:
                        score_color = "🟢"
                    elif result['score'] >= 0.15:
                        score_color = "🟡"
                    else:
                        score_color = "🟠"
                    
                    st.markdown(f"**{idx}. {result['judul']}**")
                    st.caption(f"🔗 {result['url']}")
                    st.markdown(f"{score_color} **Skor relevansi:** `{result['score']:.4f}`")
                    st.markdown(f"📝 {result['snippet']}")
                    st.markdown(f"[📖 Baca selengkapnya]({result['url']})", unsafe_allow_html=True)
                    st.markdown("---")
        else:
            st.warning(f"😔 Tidak ada berita yang cocok dengan '{query}'. Coba kata kunci lain.")
    
    elif search_clicked and not query:
        st.warning("⚠️ Masukkan kata kunci pencarian terlebih dahulu")
    
    # Tampilkan beberapa berita terbaru jika belum mencari
    if not search_clicked:
        st.markdown("### 📰 Berita Terbaru")
        st.markdown("---")
        
        for idx, (_, row) in enumerate(st.session_state.df.head(5).iterrows()):
            with st.container():
                judul_text = str(row['Judul'])
                judul_display = judul_text[:100] + "..." if len(judul_text) > 100 else judul_text
                st.markdown(f"**{idx+1}. {judul_display}**")
                st.caption(f"🔗 {str(row['URL'])[:80]}...")
                st.markdown(f"[📖 Baca berita]({row['URL']})", unsafe_allow_html=True)
                st.markdown("---")

else:
    # Jika auto-load gagal, tampilkan opsi upload manual
    st.warning("⚠️ Auto-load dari GitHub gagal. Silakan upload dataset manual.")
    
    st.markdown("### 📋 Format File Excel yang Dibutuhkan:")
    st.markdown("""
    File Excel harus memiliki **4 kolom**:
    
    | No | URL | Judul | Text |
    |----|-----|-------|------|
    | 1 | https://... | Judul berita | Isi berita lengkap... |
    | 2 | https://... | Judul berita | Isi berita lengkap... |
    """)
    
    # Tombol download contoh file
    sample_df = pd.DataFrame({
        'No': [1, 2],
        'URL': ['https://kompas.com/berita1', 'https://kompas.com/berita2'],
        'Judul': ['Contoh Judul Berita 1', 'Contoh Judul Berita 2'],
        'Text': ['Isi berita lengkap contoh 1...', 'Isi berita lengkap contoh 2...']
    })
    
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        sample_df.to_excel(writer, index=False)
    excel_data = output.getvalue()
    
    st.download_button(
        label="📥 Download Contoh File Excel",
        data=excel_data,
        file_name="contoh_dataset_berita.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    st.markdown("---")
    st.markdown("### 📂 Upload Dataset Manual:")
    
    uploaded_file = st.file_uploader(
        "Pilih file Excel",
        type=['xlsx', 'xls'],
        help="Upload file Excel dengan kolom: No, URL, Judul, Text"
    )
    
    if uploaded_file is not None:
        with st.spinner("Memproses data..."):
            df, vectorizer, tfidf_matrix, success = process_uploaded_file(uploaded_file)
            
            if success and df is not None:
                st.session_state.df = df
                st.session_state.vectorizer = vectorizer
                st.session_state.tfidf_matrix = tfidf_matrix
                st.session_state.data_loaded = True
                st.session_state.total_berita = len(df)
                st.success(f"✅ Berhasil! {len(df)} berita siap dicari")
                st.rerun()
