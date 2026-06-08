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
import io
import os

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
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
    }
    @media (max-width: 768px) {
        .result-title {
            font-size: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

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

@st.cache_resource
def load_preprocessing_components():
    """Load komponen preprocessing"""
    try:
        factory_stop = StopWordRemoverFactory()
        stopword = factory_stop.create_stop_word_remover()
        
        factory_stem = StemmerFactory()
        stemmer = factory_stem.create_stemmer()
        
        return stopword, stemmer
    except Exception as e:
        st.error(f"Error loading preprocessing components: {str(e)}")
        return None, None

@st.cache_resource
def load_semantic_model():
    """Load model semantic"""
    try:
        model = SentenceTransformer('firqaaa/indo-sentence-bert-base')
        return model
    except Exception as e:
        st.warning(f"Semantic model tidak dapat dimuat: {str(e)}")
        return None

def process_dataset(df):
    """Proses dataset untuk indexing"""
    with st.spinner("Memproses dataset..."):
        try:
            # Preprocessing teks
            def clean_text(text):
                text = str(text).lower()
                text = re.sub(r'http\S+', '', text)
                text = re.sub(r'[^a-zA-Z\s]', '', text)
                text = re.sub(r'\s+', ' ', text)
                return text
            
            stopword, stemmer = load_preprocessing_components()
            
            if stopword is None or stemmer is None:
                st.error("Gagal memuat komponen preprocessing")
                return None, None, None, None
            
            # Bersihkan teks
            df["clean_text"] = df["Text"].apply(clean_text)
            df["clean_text"] = df["clean_text"].apply(lambda x: stopword.remove(x))
            df["clean_text"] = df["clean_text"].apply(lambda x: stemmer.stem(x))
            
            # Buat TF-IDF matrix
            processed_paper = df["clean_text"].tolist()
            vectorizer = TfidfVectorizer(use_idf=True, max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(processed_paper)
            
            return df, vectorizer, tfidf_matrix, processed_paper
        except Exception as e:
            st.error(f"Error processing dataset: {str(e)}")
            return None, None, None, None

def preprocess_query(query, stopword, stemmer):
    """Preprocessing query"""
    try:
        query = query.lower()
        query = re.sub(r'[^a-zA-Z\s]', '', query)
        query = stopword.remove(query)
        query = query.split()
        query = [stemmer.stem(w) for w in query]
        return query
    except Exception as e:
        st.error(f"Error preprocessing query: {str(e)}")
        return []

def get_sinonim(kata):
    """Ambil sinonim dari web"""
    try:
        data = {"q": kata}
        encoded_data = urllib.parse.urlencode(data).encode("utf-8")
        req = urllib.request.Request(
            "http://www.sinonimkata.com/search.php", 
            data=encoded_data,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        content = urllib.request.urlopen(req, timeout=5)
        soup = BeautifulSoup(content, 'html.parser')
        synonym = soup.find('td', attrs={'width': '90%'})
        if synonym:
            synonym = synonym.find_all('a')
            synonym = [x.getText() for x in synonym]
            return [kata] + synonym[:5]
        return [kata]
    except Exception as e:
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
            if idx_kata < len(query_variasi):
                query_variasi[idx_kata] = s
                q_variasi_str = " ".join(query_variasi)
                
                emb_variasi = model.encode([q_variasi_str])
                skor = cosine_similarity(emb_asli, emb_variasi)[0][0]
                
                if skor >= 0.6:
                    hasil.append((s, skor))
        
        hasil = sorted(hasil, key=lambda x: x[1], reverse=True)
        return [x[0] for x in hasil[:top_n]]
    except Exception as e:
        return sinonim_list[:top_n]

def query_expansion(query_words, stopword, stemmer, model=None):
    """Query expansion otomatis"""
    try:
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
                if i < len(kombinasi):
                    kombinasi[i] = s
                    qs.add(' '.join(kombinasi))
        
        qs.add(' '.join(query_words))
        
        return qs
    except Exception as e:
        return {' '.join(query_words)}

def search_documents(query, df, vectorizer, tfidf_matrix, processed_paper, stopword, stemmer, model=None):
    """Fungsi pencarian utama"""
    try:
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
                    if score > 0.05:
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
        for item in sorted_results[:50]:  # Batasi 50 hasil teratas
            i = item['index']
            if i < len(df):
                results.append({
                    'judul': df['Judul'].iloc[i],
                    'url': df['URL'].iloc[i],
                    'score': item['score'],
                    'snippet': str(df['Text'].iloc[i])[:300] + '...' if len(str(df['Text'].iloc[i])) > 300 else str(df['Text'].iloc[i])
                })
        
        return results
    except Exception as e:
        st.error(f"Error searching documents: {str(e)}")
        return []

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🔍 Mesin Pencari Berita</h1>
        <p style="color: white; margin: 0.5rem 0 0 0;">Temukan berita terbaru dengan mudah dan cepat</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load komponen preprocessing
    stopword, stemmer = load_preprocessing_components()
    semantic_model = load_semantic_model()
    
    # Sidebar untuk upload dataset
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-content">
            <h3>📂 Upload Dataset</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Upload file Excel
        uploaded_file = st.file_uploader(
            "Pilih file dataset berita (format Excel)",
            type=['xlsx', 'xls'],
            help="File harus memiliki kolom: No, URL, Judul, Text"
        )
        
        if uploaded_file is not None:
            try:
                # Baca file Excel
                df_temp = pd.read_excel(uploaded_file)
                
                # Cek kolom yang diperlukan
                required_columns = ['No', 'URL', 'Judul', 'Text']
                if all(col in df_temp.columns for col in required_columns):
                    if st.button("🔄 Proses Dataset", use_container_width=True):
                        # Proses dataset
                        df, vectorizer, tfidf_matrix, processed_paper = process_dataset(df_temp)
                        
                        if df is not None:
                            # Simpan ke session state
                            st.session_state.df = df
                            st.session_state.vectorizer = vectorizer
                            st.session_state.tfidf_matrix = tfidf_matrix
                            st.session_state.processed_paper = processed_paper
                            st.session_state.data_loaded = True
                            
                            st.success(f"✅ Berhasil memproses {len(df)} berita!")
                            st.balloons()
                else:
                    st.error(f"❌ File harus memiliki kolom: {', '.join(required_columns)}")
                    st.info("Contoh format: No, URL, Judul, Text")
            
            except Exception as e:
                st.error(f"Error membaca file: {str(e)}")
        
        st.markdown("---")
        
        # Tampilkan statistik jika data sudah loaded
        if st.session_state.data_loaded and st.session_state.df is not None:
            st.markdown("""
            <div class="sidebar-content">
                <h3>📊 Statistik Dataset</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Jumlah Berita", len(st.session_state.df))
            st.metric("Topik Unik", st.session_state.df['Judul'].nunique())
            
            # Preview dataset
            with st.expander("📋 Preview Dataset"):
                st.dataframe(st.session_state.df[['No', 'Judul']].head(5), use_container_width=True)
        
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
        # Cek apakah data sudah diupload
        if not st.session_state.data_loaded:
            st.info("📌 **Selamat datang!** Silakan upload dataset berita di sidebar kiri untuk memulai pencarian.")
            
            # Tampilkan contoh format
            with st.expander("📖 Lihat Contoh Format Dataset"):
                st.markdown("""
                File Excel harus memiliki kolom:
                - **No**: Nomor urut berita
                - **URL**: Link lengkap berita
                - **Judul**: Judul berita
                - **Text**: Isi/konten berita lengkap
                
                Contoh:
                | No | URL | Judul | Text |
                |----|-----|-------|------|
                | 1 | https://... | Contoh Judul | Isi berita lengkap... |
                """)
                
                # Tombol download contoh - FIXED VERSION
                sample_df = pd.DataFrame({
                    'No': [1, 2],
                    'URL': ['https://example.com/berita1', 'https://example.com/berita2'],
                    'Judul': ['Contoh Berita 1', 'Contoh Berita 2'],
                    'Text': ['Ini adalah isi berita contoh 1', 'Ini adalah isi berita contoh 2']
                })
                
                # Convert to Excel in memory
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    sample_df.to_excel(writer, index=False)
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📥 Download Contoh Dataset",
                    data=excel_data,
                    file_name="contoh_dataset_berita.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            # Search box
            st.markdown('<div class="search-container">', unsafe_allow_html=True)
            
            query = st.text_input(
                "🔎 Masukkan kata kunci pencarian",
                placeholder="Contoh: harga emas, politik indonesia, teknologi terbaru...",
                label_visibility="collapsed"
            )
            
            search_button = st.button("Cari Berita", type="primary", use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Proses pencarian
            if search_button and query:
                with st.spinner("🔍 Mencari berita yang relevan..."):
                    time.sleep(0.5)
                    results = search_documents(
                        query, 
                        st.session_state.df, 
                        st.session_state.vectorizer, 
                        st.session_state.tfidf_matrix,
                        st.session_state.processed_paper,
                        stopword, 
                        stemmer, 
                        semantic_model
                    )
                
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
                            
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                if st.button(f"📖 Baca", key=f"btn_{idx}"):
                                    st.markdown(f'<a href="{result["url"]}" target="_blank">Klik di sini untuk membaca berita lengkap</a>', unsafe_allow_html=True)
                else:
                    st.warning("😔 Tidak ditemukan berita yang sesuai. Coba kata kunci lain.")
            
            elif search_button and not query:
                st.warning("⚠️ Silakan masukkan kata kunci pencarian terlebih dahulu.")
            
            # Tampilkan beberapa berita terbaru jika belum mencari
            if not search_button and st.session_state.df is not None:
                st.markdown("### 📰 Berita Terbaru dalam Dataset")
                
                cols = st.columns(2)
                for idx, (_, row) in enumerate(st.session_state.df.head(6).iterrows()):
                    with cols[idx % 2]:
                        with st.container():
                            judul_singkat = row['Judul'][:100] + '...' if len(str(row['Judul'])) > 100 else row['Judul']
                            st.markdown(f"""
                            <div class="result-card">
                                <div class="result-title">{judul_singkat}</div>
                                <div class="result-url">🔗 {str(row['URL'])[:80]}...</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if st.button(f"Baca", key=f"preview_{idx}"):
                                st.markdown(f'<a href="{row["URL"]}" target="_blank">Klik di sini untuk membaca</a>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
