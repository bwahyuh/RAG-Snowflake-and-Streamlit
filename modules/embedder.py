import streamlit as st
import voyageai
from PIL import Image
import io

# Init Client
client = voyageai.Client(api_key=st.secrets["voyage"]["api_key"])

def get_text_embedding(text_query):
    """
    Generate embedding untuk teks.
    """
    try:
        result = client.multimodal_embed(
            inputs=[[text_query]], 
            model="voyage-multimodal-3", 
            input_type="query"
        )
        return result.embeddings[0]
    except Exception as e:
        st.error(f"❌ Text Embedding Error: {e}")
        return []

def get_image_embedding_from_bytes(uploaded_file):
    """
    Generate embedding untuk gambar.
    Menggunakan PIL Image langsung agar diterima oleh Voyage SDK.
    """
    try:
        # 1. Pastikan pointer file ada di awal (Penting untuk Streamlit)
        uploaded_file.seek(0)
        
        # 2. Convert file upload Streamlit menjadi PIL Image
        # Ini format yang diminta oleh error message tadi ("PIL images")
        pil_image = Image.open(uploaded_file)
        
        # 3. Kirim ke Voyage
        # Format: inputs=[ [content_1, content_2] ]
        # Kita kirim [[pil_image]] karena ini single multimodal query
        result = client.multimodal_embed(
            inputs=[[pil_image]], 
            model="voyage-multimodal-3",
            input_type="query" 
        )
        
        return result.embeddings[0]
        
    except Exception as e:
        st.error(f"❌ Image Embedding Error: {e}")
        return []