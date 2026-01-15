import streamlit as st
import pandas as pd
import json
import time
import snowflake.connector
import io
import os
import tempfile
import base64
from PIL import Image

# LANGCHAIN IMPORTS
from modules.database import search_products_by_vector
from modules.embedder import get_text_embedding, get_image_embedding_from_bytes
from modules.llm import get_llm_cortex
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
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
    .reasoning-box {
        background-color: #f8f9fa; border-left: 4px solid #7c4dff; padding: 15px;
        margin-bottom: 15px; border-radius: 4px; font-family: 'Source Code Pro', monospace;
        font-size: 0.85rem; color: #333;
    }
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] { gap: 0.3rem; }
    .product-title-small { font-weight: 700; font-size: 0.8rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 5px; color: #2c3e50; }
    .price-tag-small { color: #27ae60; font-weight: 800; font-size: 0.85rem; }
    .brand-tag-small { font-size: 0.7rem; color: #95a5a6; text-transform: uppercase; letter-spacing: 0.5px; }
    div[data-testid="stImage"] { display: flex; justify_content: center; background-color: #fff; border-radius: 6px; border: 1px solid #eee; }
    div[data-testid="stImage"] > img { max-height: 120px; object-fit: contain; padding: 8px; }
    button[kind="primary"] { font-size: 0.8rem; padding: 0px 10px; }
</style>
""", unsafe_allow_html=True)

st.title("👟 SoleMate AI")
st.caption("Your smart footwear concierge. Powered by Snowflake Cortex & Voyage AI.")

# ==========================================
# 1. STATE & MEMORY
# ==========================================
if "langchain_store" not in st.session_state:
    st.session_state.langchain_store = {}
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm SoleMate. Upload a photo or describe what you're looking for!", "type": "text"}
    ]
if "image_cache" not in st.session_state:
    st.session_state.image_cache = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.langchain_store:
        st.session_state.langchain_store[session_id] = ChatMessageHistory()
    return st.session_state.langchain_store[session_id]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def fetch_images_batch(filenames):
    missing_files = [f for f in filenames if f and f not in st.session_state.image_cache]
    if not missing_files: return 
    conn = None
    try:
        conn = snowflake.connector.connect(**st.secrets["snowflake"])
        cursor = conn.cursor()
        for filename in missing_files:
            try:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    safe_tmp_path = tmpdirname.replace(os.sep, '/')
                    sql = f"GET {STAGE_PATH}/{filename} file://{safe_tmp_path}"
                    cursor.execute(sql)
                    downloaded_path = os.path.join(tmpdirname, filename)
                    possible_gz_path = downloaded_path + ".gz"
                    final_path = downloaded_path if os.path.exists(downloaded_path) else (possible_gz_path if os.path.exists(possible_gz_path) else None)
                    if final_path:
                        with open(final_path, "rb") as f: st.session_state.image_cache[filename] = f.read()
                    else: st.session_state.image_cache[filename] = None 
            except Exception: st.session_state.image_cache[filename] = None
    except Exception: pass
    finally:
        if conn: conn.close()

def render_product_image(filename, use_container_width=True):
    data = st.session_state.image_cache.get(filename)
    if data: st.image(data, use_container_width=use_container_width)
    else: st.image("https://via.placeholder.com/300x200?text=Image+Not+Found", use_container_width=use_container_width)

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

# =======================================================
# 3. VISION & ROUTER MODULE (NATIVE CORTEX)
# =======================================================

def analyze_image_with_cortex(image_file):
    """
    Versi Fix Upload: Menangani path Windows dan File Locking dengan ketat.
    """
    conn = None
    # Nama file unik
    temp_filename = f"temp_{int(time.time())}.jpg"
    tmp_path = None
    
    try:
        conn = snowflake.connector.connect(**st.secrets["snowflake"])
        cursor = conn.cursor()

        # 1. SIMPAN FILE LOKAL DENGAN AMAN
        image_file.seek(0)
        # delete=False supaya file fisik tetap ada untuk dibaca PUT
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", mode='wb') as tmp:
            tmp.write(image_file.read())
            tmp_path = tmp.name
            # PENTING: File otomatis tertutup saat keluar dari block 'with'
            # Ini melepas 'Lock' dari Windows supaya driver Snowflake bisa membacanya
        
        # 2. NORMALISASI PATH (ANTI-WINDOWS BUG)
        # Ambil path absolut dan ubah backslash jadi forward slash
        abs_path = os.path.abspath(tmp_path).replace("\\", "/")
        
        # 3. EKSEKUSI PUT
        # 'file://' + path absolut
        put_query = f"PUT 'file://{abs_path}' {STAGE_PATH}/temp_vision/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
        
        # print(f"DEBUG PUT: {put_query}") # Cek di terminal kalau penasaran
        cursor.execute(put_query)
        
        # 4. CEK HASIL PUT (JANGAN ASUMSI SUKSES)
        put_result = cursor.fetchone()
        # Struktur result PUT biasanya: [source, target, source_size, target_size, source_compression, target_compression, status, message]
        # Index 6 adalah Status (UPLOADED, SKIPPED, dll)
        upload_status = put_result[6]
        
        if upload_status != 'UPLOADED':
            return f"ERROR_VISION: Upload Failed. Status: {upload_status}. Path: {abs_path}"

        # 5. PANGGIL CORTEX (NATIVE TO_FILE)
        # File di stage akan bernama sama dengan nama file lokal (basename)
        target_filename = os.path.basename(abs_path)
        stage_file_path = f"temp_vision/{target_filename}"
        
        model = 'claude-3-5-sonnet'
        prompt = "Describe this image in detail. Is it footwear (shoes, sandals, boots)? If yes, describe color and style. If no, say what object it is."
        
        sql_query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{model}', 
            '{prompt}', 
            TO_FILE('{STAGE_PATH}', '{stage_file_path}')
        ) as response
        """
        
        cursor.execute(sql_query)
        result = cursor.fetchone()
        
        if result:
            return str(result[0])
        else:
            return "No description returned from Cortex."

    except Exception as e:
        return f"ERROR_VISION: {str(e)}"
        
    finally:
        # Bersih-bersih file lokal
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass # Kadang Windows masih lock sebentar, biarkan saja
        if conn:
            conn.close()

def smart_router(user_text, image_desc):
    """
    Otak yang menentukan nasib request:
    1. Apakah ini Sepatu? (Kalo Ferrari -> Tolak)
    2. Apakah User mau Belanja atau Ngobrol?
    """
    system_prompt = """
    You are a Smart Router for a Shoe Store AI.
    
    INPUTS:
    - User Text: "{user_text}"
    - Image Description: "{image_desc}"
    
    RULES:
    1. IS_FOOTWEAR check:
       - Read 'Image Description'. If it describes a car, animal, food, electronic -> is_footwear = FALSE.
       - If it describes shoes, boots, sandals -> is_footwear = TRUE.
       - If Image Description says "No image", check User Text context.
       
    2. INTENT check:
       - If User Text implies searching/buying ("find this", "price?", "in blue") -> SEARCH.
       - If User Text implies context ("This is my old shoe") -> SEARCH (looking for replacement).
       - If User Text is pure chat ("Cool right?", "Hello") -> CHAT.
       - If is_footwear = FALSE -> CHAT (Discuss the non-shoe object).

    OUTPUT JSON:
    {{
        "is_footwear": true/false,
        "intent": "SEARCH" or "CHAT"
    }}
    """
    try:
        prompt = ChatPromptTemplate.from_template(system_prompt)
        llm = get_llm_cortex() 
        chain = prompt | llm | JsonOutputParser()
        return chain.invoke({"user_text": user_text, "image_desc": image_desc})
    except:
        return {"is_footwear": True, "intent": "SEARCH"}

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

# ==========================================
# 4. UI & SIDEBAR
# ==========================================
with st.sidebar:
    st.header("📸 Visual Match")
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
    analyze_btn = False
    if uploaded_file:
        st.image(uploaded_file, caption="Preview", use_container_width=True)
        analyze_btn = st.button("🔍 Analyze This Image", type="primary", use_container_width=True)

# Preload History
all_history_images = []
for msg in st.session_state.messages:
    if msg.get("role") == "assistant" and msg.get("data") is not None and not msg["data"].empty:
        files = msg["data"]['IMAGE_FILENAME'].tolist()
        all_history_images.extend(files)
if all_history_images:
    fetch_images_batch(list(set(all_history_images)))

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg.get("thought"):
            with st.expander("🧠 AI Reasoning & Context Analysis", expanded=False):
                st.markdown(f"<div class='reasoning-box'>{msg['thought']}</div>", unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg.get("role") == "assistant" and msg.get("data") is not None and not msg["data"].empty:
            st.markdown("---")
            st.caption(f"👟 Found {len(msg['data'])} Options:")
            df_products = msg["data"]
            cols = st.columns(5) 
            for idx, row in df_products.iterrows():
                with cols[idx % 5]:
                    with st.container(border=True):
                        render_product_image(row['IMAGE_FILENAME'], use_container_width=True)
                        short_title = (row['TITLE'][:18] + '..') if len(row['TITLE']) > 18 else row['TITLE']
                        st.markdown(f"<div class='product-title-small' title='{row['TITLE']}'>{short_title}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='price-tag-small'>{row['PRICE']}</div>", unsafe_allow_html=True)
                        if st.button("View", key=f"hist_btn_{i}_{idx}", use_container_width=True):
                            show_product_popup(row)

# ==========================================
# 5. MAIN LOGIC (THE BRAIN)
# ==========================================
input_text = st.chat_input("Ex: Find this in red, or just ask anything...")
trigger = False
content_text = ""

if analyze_btn and uploaded_file:
    content_text = "🖼️ [User sent an image for analysis]"
    trigger = True
elif input_text:
    if uploaded_file:
        content_text = f"🖼️ [Image + Text] {input_text}"
    else:
        content_text = input_text
    trigger = True

if trigger:
    st.session_state.messages.append({"role": "user", "content": content_text})
    st.rerun()

if st.session_state.messages[-1]["role"] == "user":
    last_msg = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        with st.status("🧠 SoleMate is thinking...", expanded=True) as status:
            
            # --- 1. VISION ANALYSIS (NATIVE TO_FILE) ---
            image_desc = "No image uploaded."
            if uploaded_file and "🖼️" in last_msg:
                st.write("👁️ **Step 1:** Cortex Vision Analysis (Native)...")
                image_desc = analyze_image_with_cortex(uploaded_file)
                
                if "ERROR_VISION" in image_desc:
                    st.error(f"System Error: {image_desc}")
                    # Fallback ke asumsi user upload sepatu
                    image_desc = "User uploaded an image (Vision unavailable)."
                else:
                    st.write(f"   → Sees: **{image_desc[:100]}...**")
            
            # --- 2. SMART ROUTING ---
            st.write("🚦 **Step 2:** Router Decision...")
            router_result = smart_router(last_msg, image_desc)
            st.write(f"   → Logic: {router_result}")
            
            intent = router_result.get("intent", "CHAT")
            is_footwear = router_result.get("is_footwear", True)

            # --- 3. EXECUTION ---
            products_df = pd.DataFrame()
            context_str = "[]"
            
            if not is_footwear:
                st.write("   → 🛑 Object is NOT footwear. Switching to CHAT.")
                
            elif intent == "SEARCH":
                st.write("🔍 **Step 3:** Vector Search (Voyage)...")
                vector = []
                if uploaded_file:
                    vector = get_image_embedding_from_bytes(uploaded_file)
                else:
                    search_query = last_msg
                    if "No image" not in image_desc:
                        search_query += f" (Context from image: {image_desc})"
                    vector = get_text_embedding(search_query)

                if len(vector) > 0:
                    products_df = search_products_by_vector(vector, limit=15)
                    st.write(f"   → ✅ Found {len(products_df)} items.")
                    if not products_df.empty:
                        fetch_images_batch(products_df['IMAGE_FILENAME'].tolist())
                        context_str = format_context_json(products_df)

            # --- 4. RESPONSE GENERATION ---
            st.write("🤔 **Step 4:** Final Answer...")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are SoleMate.
                
                CONTEXT FROM DB: {context}
                IMAGE ANALYSIS: "{img_desc}"
                IS FOOTWEAR: {is_footwear}
                
                TASKS:
                1. If IS_FOOTWEAR is False, talk about the object in 'IMAGE ANALYSIS' (e.g. "Nice car!") but say you only sell shoes.
                2. If intent is SEARCH, recommend Top 3 items from CONTEXT. Use the IMAGE ANALYSIS to connect user's image with results.
                3. If intent is CHAT, just chat friendly.
                
                OUTPUT JSON:
                {{
                    "classification": "recommendation" | "greeting" | "off_topic",
                    "thought": "Reasoning...",
                    "response_text": "Markdown...",
                    "recommended_products": []
                }}
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
            
            llm = get_llm_cortex()
            runnable_chain = prompt | llm | JsonOutputParser()
            
            chain_with_history = RunnableWithMessageHistory(
                runnable_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="response_text"
            )
            
            try:
                result_json = chain_with_history.invoke(
                    {
                        "input": last_msg, 
                        "context": context_str,
                        "img_desc": image_desc,
                        "is_footwear": is_footwear
                    },
                    config={"configurable": {"session_id": "user_session_1"}}
                )
                status.update(label="✅ Done!", state="complete", expanded=False)
            
            except Exception as e:
                st.error(f"Response Error: {e}")
                st.stop()

        # --- FINAL RENDER ---
        if result_json.get('thought'):
            with st.expander("🧠 AI Reasoning & Context Analysis", expanded=True):
                st.markdown(f"""
                **Thinking Process:**
                {result_json.get('thought')}
                """)
        
        final_resp = result_json.get("response_text")
        st.markdown(final_resp)
        
        show_grid = False
        if result_json.get('classification') == 'recommendation' and not products_df.empty:
            show_grid = True

        if show_grid:
            st.markdown("---")
            st.markdown("### 🛍️ Explore All 15 Options")
            cols = st.columns(5)
            for idx, row in products_df.iterrows():
                with cols[idx % 5]:
                    with st.container(border=True):
                        render_product_image(row['IMAGE_FILENAME'], use_container_width=True)
                        short_title = (row['TITLE'][:18] + '..') if len(row['TITLE']) > 18 else row['TITLE']
                        st.markdown(f"<div class='product-title-small' title='{row['TITLE']}'>{short_title}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='price-tag-small'>{row['PRICE']}</div>", unsafe_allow_html=True)
                        if st.button("View", key=f"now_btn_{idx}", use_container_width=True):
                            show_product_popup(row)

        st.session_state.messages.append({
            "role": "assistant", 
            "content": final_resp,
            "thought": result_json.get('thought'),
            "data": products_df if show_grid else None
        })
        st.rerun()