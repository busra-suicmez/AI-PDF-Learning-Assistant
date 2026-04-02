import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader  # <-- EKSİK OLAN VE BU HATAYI VEREN SATIR!
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# ... (Geri kalan kodlar: load_dotenv, TOKEN kontrolü vb.)
# .env dosyasının tam yolunu bul (Daha garantidir)
env_path = Path('.') / '.env'
load_dotenv(override=True) 

TOKEN = os.getenv("HF_TOKEN")
if not TOKEN or not TOKEN.startswith("hf_"):
    st.error("⚠️ Token Okunamadı! .env dosyasını ve ismini kontrol edin.")
    st.info(f"Sistem şu an bu klasörde çalışıyor: {os.getcwd()}")
    st.stop()
# 3. Modelleri başlatırken artık bu güvenli TOKEN'ı kullanıyoruz
client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token=TOKEN)

# Sayfa Ayarları
st.set_page_config(page_title="Pro Ders Asistanı", page_icon="🤖")
st.title("🤖 Pro AI Ders Asistanı (RAG + Memory)")


# Vektörleştirme Modeli (Hafif ve Hızlı)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()
# --- HAFIZA YÖNETİMİ ---
if "messages" not in st.session_state:
    st.session_state.messages = [] # Sohbet geçmişini tutar

# --- PDF İŞLEME VE VEKTÖR VERİTABANI ---
uploaded_file = st.file_uploader("Ders notunu (PDF) yükle", type="pdf")

if uploaded_file:
    # PDF'i oku ve parçala
    reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text()
    
    # Metni küçük parçalara böl (AI'nın okuyabileceği boyutlar)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(raw_text)
    
    # Vektör veritabanı oluştur (Hızlı arama için)
    vector_db = FAISS.from_texts(chunks, embeddings)
    st.success("PDF Analiz Edildi ve Vektör Veritabanı Hazır!")

    # --- SOHBET ARAYÜZÜ ---
    # Geçmiş mesajları ekrana bas
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Kullanıcıdan soru al
    if prompt := st.chat_input("Notlarla ilgili bir şey sor..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # RAG: Soruyla en alakalı metin parçalarını bul
        docs = vector_db.similarity_search(prompt, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        # AI'ya gönderilecek sistem talimatı ve geçmiş
        system_msg = f"Sana verilen şu notlara göre cevap ver:\n{context}\n\nTürkçe cevap ver."
        
        with st.chat_message("assistant"):
            with st.spinner("Düşünüyorum..."):
                messages = [{"role": "system", "content": system_msg}]
                # Hafızayı AI'ya gönderiyoruz
                messages.extend(st.session_state.messages[-5:]) # Son 5 mesajı hatırlasın
                
                response = client.chat_completion(messages=messages, max_tokens=500)
                full_response = response.choices[0].message.content
                st.markdown(full_response) 

                
                with st.expander("🔍 AI Bu Bilgiyi Nereden Buldu? (Kaynak Metinler)"):
                    for i, doc in enumerate(docs):
                        st.info(f"**Kaynak {i+1}:**\n{doc.page_content}")
                # -----------------------------------

        st.session_state.messages.append({"role": "assistant", "content": full_response})