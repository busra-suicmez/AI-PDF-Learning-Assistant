# 🎓 AI PDF Ders Asistanı (RAG & Memory)

Bu proje, Llama-3 ve LangChain kullanarak geliştirilmiş, PDF dokümanları üzerinden soru-cevap yapabilen bir yapay zeka asistanıdır.

## 🛠️ Özellikler
- **RAG Mimarisi:** Dokümanları FAISS vektör veritabanında saklar.
- **Hafıza:** Konuşma geçmişini hatırlar.
- **Güvenlik:** `.env` mimarisi ile API anahtarlarını korur.

## 🚀 Kurulum
1. Klasörü indirin.
2. `pip install -r requirements.txt` ile kütüphaneleri yükleyin.
3. `.env` dosyası oluşturup `HF_TOKEN` değerini girin.
4. `streamlit run app.py` ile çalıştırın.
