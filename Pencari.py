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

# Sidebar untuk upload data
with st.sidebar:
    st.header("📂 Manajemen Data")
    
    st.subheader("1. Upload File Berita")
    st.markdown("""
    **Format file Excel (.xlsx) dengan kolom:**
    - `No` (nomor urut)
    - `URL` (link berita)
    - `Judul` (judul berita)
    - `Text` (isi berita)
    """)
    
    uploaded_file = st.file_uploader(
        "Pilih file Excel",
        type=['xlsx'],
        help="Upload file berita dalam format Excel"
    )
    
    if uploaded_file is not None:
        try:
            df_temp = pd.read_excel(uploaded_file)
            
            # Validasi kolom
            required = ['No', 'URL', 'Judul', 'Text']
            if all(col in df_temp.columns for col in required):
                st.success(f"✓ File valid: {len(df_temp)} berita")
                
                if st.button("🚀 Proses Data", use_container_width=True):
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
                        df_temp["clean_text"] = df_temp["Text"].apply(clean_text)
                        df_temp["clean_text"] = df_temp["clean_text"].apply(lambda x: stopword.remove(x))
                        df_temp["clean_text"] = df_temp["clean_text"].apply(lambda x: stemmer.stem(x))
                        
                        # TF-IDF
                        processed_paper = df_temp["clean_text"].tolist()
                        vectorizer = TfidfVectorizer(use_idf=True)
                        tfidf_matrix = vectorizer.fit_transform(processed_paper)
                        
                        # Simpan ke session
                        st.session_state.df = df_temp
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
        st.subheader("📊 Status Data")
        st.info(f"✓ {len(st.session_state.df)} berita terload")
        st.caption(f"Contoh: {st.session_state.df['Judul'].iloc[0][:50]}...")

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
        for syn in synonyms[:2]:  # Ambil max 2 sinonim
            if syn != word:
                new_query = query_words.copy()
                new_query[i] = syn
                expanded.add(' '.join(new_query))
    
    return expanded

# Main search area
if not st.session_state.data_loaded:
    st.info("👈 **Silakan upload file berita di sidebar kiri untuk memulai**")
    st.markdown("""
    ### Cara Memasukkan Data:
    
    1. **Siapkan file Excel** dengan kolom:
       - `No` (nomor urut)
       - `URL` (link berita)
       - `Judul` (judul berita)
       - `Text` (isi berita lengkap)
    
    2. **Klik "Browse files"** di sidebar kiri
    
    3. **Pilih file Excel** Anda
    
    4. **Klik "Proses Data"** untuk mengindex berita
    
    5. **Mulai mencari!** 🔍
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
                        if score > 0.05:  # Threshold minimal
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
                    
                    # Card hasil
                    with st.container():
                        st.markdown(f"**{idx}. {row['Judul']}**")
                        st.caption(f"🔗 {row['URL']}")
                        st.markdown(f"📊 **Skor relevansi:** `{score:.4f}`")
                        
                        # Snippet
                        snippet = str(row['Text'])[:200] + "..." if len(str(row['Text'])) > 200 else str(row['Text'])
                        st.markdown(f"📝 {snippet}")
                        
                        # Tombol baca
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
                st.markdown("[📖 Baca berita]({})".format(row['URL']), unsafe_allow_html=True)
                st.markdown("---")
