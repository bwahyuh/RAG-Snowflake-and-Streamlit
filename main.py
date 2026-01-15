import streamlit as st
import pandas as pd
import json
import time
import snowflake.connector
import io
import os
import tempfile

# LANGCHAIN IMPORTS (THE FULL ARSENAL)
from modules.database import search_products_by_vector
from modules.embedder import get_text_embedding, get_image_embedding_from_bytes
from modules.llm import get_llm_cortex
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ==========================================
# 0. CONFIG
# ==========================================
STAGE_PATH = "@RETAIL_GENAI_DB.PUBLIC.PRODUCT_IMAGES_STAGE"

st.set_page_config(
    page_title="SoleMate AI",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; padding: 12px; font-size: 0.95rem; }
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] { gap: 0.5rem; }
    .product-title-small { font-weight: 600; font-size: 0.85rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 5px; color: #333; }
    .price-tag-small { color: #2e7d32; font-weight: 700; font-size: 0.9rem; }
    .brand-tag-small { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
    div[data-testid="stImage"] { display: flex; justify_content: center; background-color: #f9f9f9; border-radius: 8px; }
    div[data-testid="stImage"] > img { max-height: 140px; object-fit: contain; padding: 5px; }
</style>
""", unsafe_allow_html=True)

st.title("👟 SoleMate AI")
st.caption("Your personal footwear concierge. Powered by Snowflake Cortex & Voyage AI.")

# ==========================================
# 1. LANGCHAIN MEMORY SETUP (NATIVE)
# ==========================================
# Kita simpan history object LangChain di Session State biar tidak hilang saat rerun
if "langchain_store" not in st.session_state:
    st.session_state.langchain_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Fungsi wajib buat RunnableWithMessageHistory.
    Mengambil object history berdasarkan session_id.
    """
    if session_id not in st.session_state.langchain_store:
        st.session_state.langchain_store[session_id] = ChatMessageHistory()
    return st.session_state.langchain_store[session_id]

# Initial UI Messages (Hanya untuk Tampilan Streamlit)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm SoleMate. Looking for specific shoes? Ask away!", "type": "text"}
    ]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_image_from_snowflake(filename):
    if not filename: return None
    conn = None
    try:
        conn = snowflake.connector.connect(**st.secrets["snowflake"])
        cursor = conn.cursor()
        with tempfile.TemporaryDirectory() as tmpdirname:
            safe_tmp_path = tmpdirname.replace(os.sep, '/')
            sql = f"GET {STAGE_PATH}/{filename} file://{safe_tmp_path}"
            cursor.execute(sql)
            downloaded_path = os.path.join(tmpdirname, filename)
            possible_gz_path = downloaded_path + ".gz"
            final_path = downloaded_path if os.path.exists(downloaded_path) else (possible_gz_path if os.path.exists(possible_gz_path) else None)
            
            if final_path:
                with open(final_path, "rb") as f: return f.read()
            return None
    except Exception: return None
    finally:
        if conn: conn.close()

def render_product_image(filename, use_container_width=True):
    img_bytes = fetch_image_from_snowflake(filename)
    if img_bytes:
        st.image(img_bytes, use_container_width=use_container_width)
    else:
        st.image("https://via.placeholder.com/300x200?text=No+Image", use_container_width=use_container_width)

def classify_intent(user_query):
    # Router tetap pakai LLM biasa karena dia stateless (gak butuh memory)
    system_prompt = """
    You are an Intent Classifier.
    Classify query into: 'SEARCH' (looking for products) or 'CHAT' (greeting, history reference, general talk).
    USER QUERY: {query}
    OUTPUT: 'SEARCH' or 'CHAT' only.
    """
    try:
        prompt = ChatPromptTemplate.from_template(system_prompt)
        llm = get_llm_cortex() 
        chain = prompt | llm | StrOutputParser()
        intent = chain.invoke({"query": user_query}).strip().upper()
        return "SEARCH" if "SEARCH" in intent else "CHAT"
    except Exception:
        return "SEARCH"

@st.dialog("✨ Product Details")
def show_product_popup(product):
    render_product_image(product['IMAGE_FILENAME'], use_container_width=True)
    st.markdown(f"### {product['TITLE']}")
    st.caption(f"Brand: {product['BRAND']}")
    col_price, col_btn = st.columns([1,1])
    with col_price: st.markdown(f"### {product['PRICE']}")
    with col_btn: st.button("🛒 Add to Cart", key=f"cart_pop_{product['TITLE'][:10]}", type="primary", use_container_width=True)
    st.divider()
    st.markdown("**Description & Features:**")
    st.info(product['PRODUCT_DETAILS_CLEAN'])

def format_context_json(df):
    if df.empty: return "[]"
    products = []
    for _, row in df.iterrows():
        products.append({
            "product_name": row['TITLE'],
            "brand": row['BRAND'],
            "price": row['PRICE'],
            "features": row['PRODUCT_DETAILS_CLEAN'][:200]
        })
    return json.dumps(products)

# ==========================================
# 3. UI RENDERER
# ==========================================
with st.sidebar:
    st.header("📸 Visual Match")
    st.write("Snap a pic, find the fit.")
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Your Snap", use_container_width=True)

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg.get("thought"):
            with st.expander("🧠 See AI Reasoning", expanded=False):
                st.markdown(f"```text\n{msg['thought']}\n```")
        st.markdown(msg["content"])
        if msg.get("role") == "assistant" and msg.get("data") is not None and not msg["data"].empty:
            st.markdown("---")
            st.caption("👟 Top Picks:")
            df_products = msg["data"]
            cols = st.columns(3)
            for idx, row in df_products.iterrows():
                with cols[idx % 3]:
                    with st.container(border=True):
                        render_product_image(row['IMAGE_FILENAME'], use_container_width=True)
                        short_title = (row['TITLE'][:22] + '..') if len(row['TITLE']) > 22 else row['TITLE']
                        st.markdown(f"<div class='product-title-small' title='{row['TITLE']}'>{short_title}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='brand-tag-small'>{row['BRAND']}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='price-tag-small'>{row['PRICE']}</div>", unsafe_allow_html=True)
                        if st.button("View", key=f"hist_btn_{i}_{idx}", use_container_width=True):
                            show_product_popup(row)

# ==========================================
# 4. MAIN LOGIC (WITH LANGCHAIN MEMORY)
# ==========================================
input_text = st.chat_input("I need comfy running shoes...")

if input_text or uploaded_file:
    trigger = False
    content_text = ""
    if input_text:
        content_text = input_text
        trigger = True
    elif uploaded_file:
        content_text = "🖼️ [Image Uploaded]"
        trigger = True
    if trigger:
        st.session_state.messages.append({"role": "user", "content": content_text})
        st.rerun()

if st.session_state.messages[-1]["role"] == "user":
    last_msg = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        with st.status("🧠 SoleMate is thinking...", expanded=True) as status:
            
            # --- ROUTER ---
            intent = "SEARCH"
            if "Image Uploaded" in last_msg:
                intent = "SEARCH"
                st.write("📸 Visual Search Detected.")
            else:
                st.write("🚦 Checking Intent...")
                intent = classify_intent(last_msg)
                st.write(f"✅ Intent: **{intent}**")

            # --- SEARCH ---
            products_df = pd.DataFrame()
            context_str = "[]"
            
            if intent == "SEARCH":
                st.write("🔍 Searching Inventory...")
                vector = []
                if "Image Uploaded" in last_msg:
                    vector = get_image_embedding_from_bytes(uploaded_file)
                else:
                    vector = get_text_embedding(last_msg)
                
                if len(vector) > 0:
                    products_df = search_products_by_vector(vector, limit=6)
                    st.write(f"✅ Found {len(products_df)} items.")
                context_str = format_context_json(products_df)
            else:
                st.write("⏭️ Checking Memory (Chat Mode).")

            # --- GENERATION (WITH NATIVE HISTORY) ---
            st.write("🤔 Drafting response...")
            
            # 1. SETUP PROMPT DENGAN MEMORY PLACEHOLDER
            # Perhatikan: Tidak ada {history} manual. Kita pakai MessagesPlaceholder.
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are SoleMate, a footwear expert. Respond in STRICT JSON.
                
                CONTEXT FROM DB: {context}
                
                JSON OUTPUT STRUCTURE:
                {{
                    "classification": "greeting" | "off_topic" | "recommendation" | "no_result",
                    "thought": "Reasoning (English). Refer to history if needed.",
                    "response_text": "Natural friendly response (English).",
                    "recommended_products": []
                }}
                """),
                MessagesPlaceholder(variable_name="chat_history"), # <--- INI KUNCINYA!
                ("human", "{input}")
            ])
            
            llm = get_llm_cortex()
            
            # 2. RAKIT CHAIN DENGAN HISTORY WRAPPER
            runnable_chain = prompt | llm | StrOutputParser()
            
            chain_with_history = RunnableWithMessageHistory(
                runnable_chain,
                get_session_history, # Fungsi pengambil history yg kita buat di atas
                input_messages_key="input",
                history_messages_key="chat_history"
            )
            
            try:
                # 3. INVOKE DENGAN SESSION ID
                raw_res = chain_with_history.invoke(
                    {
                        "input": last_msg, 
                        "context": context_str
                    },
                    config={"configurable": {"session_id": "user_session_1"}} # Session ID tetap
                )
                
                result_json = json.loads(raw_res.replace("```json", "").replace("```", "").strip())
                status.update(label="✅ Ready!", state="complete", expanded=False)
            
            except Exception as e:
                status.update(label="❌ Error", state="error")
                st.error(f"Error: {e}")
                st.stop()

        # --- RENDER FINAL ---
        if result_json.get('thought'):
            with st.expander("🧠 See AI Reasoning", expanded=True):
                st.write(result_json.get('thought'))
        
        final_resp = result_json.get("response_text")
        st.markdown(final_resp)
        
        if result_json.get('classification') == 'recommendation' and not products_df.empty:
            st.markdown("---")
            cols = st.columns(3)
            for idx, row in products_df.iterrows():
                with cols[idx % 3]:
                    with st.container(border=True):
                        render_product_image(row['IMAGE_FILENAME'], use_container_width=True)
                        short_title = (row['TITLE'][:22] + '..') if len(row['TITLE']) > 22 else row['TITLE']
                        st.markdown(f"<div class='product-title-small'>{short_title}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='price-tag-small'>{row['PRICE']}</div>", unsafe_allow_html=True)
                        if st.button("View", key=f"now_btn_{idx}", use_container_width=True):
                            show_product_popup(row)

        st.session_state.messages.append({
            "role": "assistant", 
            "content": final_resp,
            "thought": result_json.get('thought'),
            "data": products_df if result_json.get('classification') == 'recommendation' else None
        })
        st.rerun()