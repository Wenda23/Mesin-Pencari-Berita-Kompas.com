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
import os

# Setup halaman
st.set_page_config(
    page_title="Mesin Pencari Berita",
    page_icon="🔍",
    layout="centered"
)

# Title
st.title("🔍 Mesin Pencari Berita")
st.markdown("---")

# Inisialisasi session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'tfidf_matrix' not in st.session_state:
    st.session_state.tfidf_matrix = None
if 'processed_paper' not in st.session_state:
    st.session_state.processed_paper = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Konfigurasi dataset dari GitHub
# GANTI URL INI dengan raw URL dataset Anda dari GitHub
GITHUB_DATASET_URL = "https://raw.githubusercontent.com/username/repo/main/dataset_berita.xlsx"
# Atau jika paket CSV
GITHUB_CSV_URL = "https://raw.githubusercontent.com/username/repo/main/dataset_berita.csv"

@st.cache_data
def load_dataset_from_github():
    """Muat dataset dari GitHub"""
    try:
        # Coba load dari Excel
        df = pd.read_excel(GITHUB_DATASET_URL)
        return df
    except:
        try:
            # Coba load dari CSV
            df = pd.read_csv(GITHUB_CSV_URL)
            return df
        except Exception as e:
            st.error(f"Gagal load dataset dari GitHub: {str(e)}")
            return None

def process_dataframe(df):
    """Proses dataframe menjadi index pencarian"""
    with st.spinner("Memproses data..."):
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
        
        return df, vectorizer, tfidf_matrix, processed_paper

# Fungsi query expansion sederhana
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
            return [word] + [w.getText() for w in words[:3]]
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

# Sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan")
    
    # Pilihan sumber data
    data_source = st.radio(
        "Sumber Data",
        ["GitHub (Auto)", "Upload Manual"],
        help="Pilih sumber dataset"
    )
    
    if data_source == "GitHub (Auto)":
        st.info("📡 Menggunakan dataset dari GitHub")
        
        # Input URL GitHub
        github_url = st.text_input(
            "URL Dataset (Excel/CSV):",
            value="https://raw.githubusercontent.com/username/repo/main/dataset_berita.xlsx",
            help="Masukkan raw URL dari file dataset di GitHub"
        )
        
        if st.button("🔄 Load dari GitHub", use_container_width=True):
            try:
                # Load dataset
                if github_url.endswith('.csv'):
                    df_temp = pd.read_csv(github_url)
                else:
                    df_temp = pd.read_excel(github_url)
                
                # Validasi kolom
                required = ['No', 'URL', 'Judul', 'Text']
                if all(col in df_temp.columns for col in required):
                    st.success(f"✓ Loaded {len(df_temp)} berita dari GitHub")
                    
                    # Proses data
                    df, vectorizer, tfidf_matrix, processed_paper = process_dataframe(df_temp)
                    
                    st.session_state.df = df
                    st.session_state.vectorizer = vectorizer
                    st.session_state.tfidf_matrix = tfidf_matrix
                    st.session_state.processed_paper = processed_paper
                    st.session_state.data_loaded = True
                    
                    st.success("✅ Index berhasil dibuat!")
                    st.balloons()
                else:
                    st.error(f"Kolom yang dibutuhkan: {', '.join(required)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Pastikan URL adalah RAW URL dari GitHub (bukan halaman repository)")
    
    else:  # Upload Manual
        st.markdown("**Upload file Excel:**")
        st.markdown("""
        **Format kolom:**
        - `No` (nomor)
        - `URL` (link)
        - `Judul` (judul)
        - `Text` (isi)
        """)
        
        uploaded_file = st.file_uploader(
            "Pilih file Excel",
            type=['xlsx', 'xls'],
            key="manual_upload"
        )
        
        if uploaded_file is not None:
            try:
                df_temp = pd.read_excel(uploaded_file)
                
                required = ['No', 'URL', 'Judul', 'Text']
                if all(col in df_temp.columns for col in required):
                    st.success(f"✓ File valid: {len(df_temp)} berita")
                    
                    if st.button("🚀 Proses Data", use_container_width=True):
                        df, vectorizer, tfidf_matrix, processed_paper = process_dataframe(df_temp)
                        
                        st.session_state.df = df
                        st.session_state.vectorizer = vectorizer
                        st.session_state.tfidf_matrix = tfidf_matrix
                        st.session_state.processed_paper = processed_paper
                        st.session_state.data_loaded = True
                        
                        st.success(f"✅ Berhasil! {len(df_temp)} berita siap dicari")
                        st.balloons()
                else:
                    st.error(f"Kolom yang dibutuhkan: {', '.join(required)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    
    # Contoh format data
    with st.expander("📋 Lihat Contoh Format"):
        sample_df = pd.DataFrame({
            'No': [1, 2],
            'URL': ['https://kompas.com/berita1', 'https://kompas.com/berita2'],
            'Judul': ['Contoh Judul Berita 1', 'Contoh Judul Berita 2'],
            'Text': ['Isi berita lengkap contoh 1...', 'Isi berita lengkap contoh 2...']
        })
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            sample_df.to_excel(writer, index=False)
        excel_data = output.getvalue()
        
        st.download_button(
            label="📥 Download Contoh File",
            data=excel_data,
            file_name="contoh_berita.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # Status data
    if st.session_state.data_loaded:
        st.markdown("---")
        st.subheader("📊 Status")
        st.info(f"✅ {len(st.session_state.df)} berita terindex")
        st.caption(f"Terakhir: {st.session_state.df['Judul'].iloc[0][:40]}...")

# Main content
if not st.session_state.data_loaded:
    st.info("👈 **Pilih sumber data di sidebar**")
    st.markdown("""
    ### Cara Memasukkan Data:
    
    **Opsi 1: Auto dari GitHub (Rekomendasi)**
    1. Upload file `dataset_berita.xlsx` ke repository GitHub
    2. Dapatkan raw URL (klik Raw → copy URL)
    3. Paste URL di kolom yang tersedia
    4. Klik "Load dari GitHub"
    
    **Opsi 2: Upload Manual**
    1. Siapkan file Excel dengan format yang benar
    2. Upload melalui file browser
    3. Klik "Proses Data"
    
    **Format file Excel:**
    | No | URL | Judul | Text |
    |----|-----|-------|------|
    | 1 | https://... | Judul berita | Isi berita... |
    """)
else:
    # Search box
    st.markdown("### 🔎 Cari Berita")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Kata kunci",
            placeholder="Contoh: harga emas, politik, teknologi, kriminal...",
            label_visibility="collapsed"
        )
    with col2:
        search_clicked = st.button("🔍 Cari", type="primary", use_container_width=True)
    
    if search_clicked and query:
        with st.spinner("Mencari..."):
            # Preprocessing query
            factory_stop = StopWordRemoverFactory()
            stopword = factory_stop.create_stop_word_remover()
            factory_stem = StemmerFactory()
            stemmer = factory_stem.create_stemmer()
            
            # Proses query
            query_lower = query.lower()
            query_clean = re.sub(r'[^a-zA-Z\s]', '', query_lower)
            query_no_stop = stopword.remove(query_clean)
            query_words = [stemmer.stem(w) for w in query_no_stop.split()]
            
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
                st.success(f"✅ Ditemukan {len(sorted_results)} berita relevan")
                st.markdown("---")
                
                # Tampilkan hasil
                for idx, (doc_id, score) in enumerate(sorted_results[:30], 1):
                    row = st.session_state.df.iloc[doc_id]
                    
                    with st.container():
                        st.markdown(f"**{idx}. {row['Judul']}**")
                        st.caption(f"🔗 {row['URL']}")
                        st.markdown(f"📊 **Skor:** `{score:.4f}`")
                        
                        # Snippet
                        snippet = str(row['Text'])[:200] + "..." if len(str(row['Text'])) > 200 else str(row['Text'])
                        st.markdown(f"📝 {snippet}")
                        st.markdown(f"[📖 Baca selengkapnya]({row['URL']})", unsafe_allow_html=True)
                        st.markdown("---")
            else:
                st.warning("😔 Tidak ada berita yang cocok. Coba kata kunci lain.")
    
    elif search_clicked and not query:
        st.warning("⚠️ Masukkan kata kunci terlebih dahulu")
    
    # Tampilkan beberapa berita terbaru
    if not search_clicked:
        st.markdown("### 📰 Berita Terbaru dalam Database")
        
        for idx, (_, row) in enumerate(st.session_state.df.head(5).iterrows()):
            with st.container():
                st.markdown(f"**{idx+1}. {row['Judul'][:100]}**")
                st.caption(f"🔗 {row['URL'][:80]}...")
                st.markdown(f"[📖 Baca berita]({row['URL']})", unsafe_allow_html=True)
                st.markdown("---")
