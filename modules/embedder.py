import streamlit as st
import voyageai
import base64
from PIL import Image
import io

client = voyageai.Client(api_key=st.secrets["voyage"]["api_key"])

def get_text_embedding(text_query):
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

def get_image_embedding_from_bytes(image_bytes):
    try:
        base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        result = client.multimodal_embed(
            inputs=[[{"type": "base64", "media": base64_image}]],
            model="voyage-multimodal-3",
            input_type="query" 
        )
        return result.embeddings[0]
    except Exception as e:
        st.error(f"❌ Image Embedding Error: {e}")
        return []