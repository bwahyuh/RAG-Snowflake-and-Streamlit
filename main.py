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
# 0. CONFIG & SAFETY CHECK
# ==========================================
STAGE_PATH = "@RETAIL_GENAI_DB.PUBLIC.PRODUCT_IMAGES_STAGE"

st.set_page_config(
    page_title="SoleMate AI",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "snowflake" not in st.secrets or "voyage" not in st.secrets:
    st.error("🚨 Missing Secrets! Please configure .streamlit/secrets.toml.")
    st.stop()

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; padding: 12px; font-size: 0.95rem; }
    .reasoning-box {
        background-color: #f8f9fa; border-left: 4px solid #7c4dff; padding: 15px;
        margin-bottom: 15px; border-radius: 4px; font-family: 'Source Code Pro', monospace;
        font-size: 0.85rem; color: #333;
    }
    .product-card-title { 
        font-weight: 700; font-size: 0.85rem; 
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis; 
        margin-top: 8px; color: #2c3e50; 
    }
    .product-card-price { color: #27ae60; font-weight: 800; font-size: 0.9rem; }
    div[data-testid="stImage"] { 
        display: flex; justify_content: center; 
        background-color: #fff; border-radius: 6px; border: 1px solid #eee; 
    }
    div[data-testid="stImage"] > img { 
        max-height: 150px; object-fit: contain; padding: 10px; 
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.logo("logo.png", size="large", link=None)

# ==========================================
# 1. STATE MANAGEMENT
# ==========================================
if "page" not in st.session_state:
    st.session_state.page = "home"
if "langchain_store" not in st.session_state:
    st.session_state.langchain_store = {}
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm SoleMate. Upload a photo or describe what you're looking for!", "type": "text"}
    ]
if "image_cache" not in st.session_state:
    st.session_state.image_cache = {}

def switch_page(page_name):
    st.session_state.page = page_name
    st.rerun()

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
            "features": row['PRODUCT_DETAILS_CLEAN'][:300] 
        })
    return json.dumps(products)

# ==========================================
# 3. CORE AI MODULES
# ==========================================
def analyze_image_with_cortex(image_file):
    conn = None
    tmp_path = None
    try:
        conn = snowflake.connector.connect(**st.secrets["snowflake"])
        cursor = conn.cursor()
        image_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", mode='wb') as tmp:
            tmp.write(image_file.read())
            tmp_path = tmp.name
        abs_path = os.path.abspath(tmp_path).replace("\\", "/")
        put_query = f"PUT 'file://{abs_path}' {STAGE_PATH}/temp_vision/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
        cursor.execute(put_query)
        put_result = cursor.fetchone()
        if put_result[6] != 'UPLOADED': return f"ERROR_VISION: Upload Failed."
        target_filename = os.path.basename(abs_path)
        stage_file_path = f"temp_vision/{target_filename}"
        model = 'claude-3-5-sonnet'
        prompt = "Describe this image in detail. Is it footwear? If yes, describe color, material, and style. If no, say what object it is."
        sql_query = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{model}', '{prompt}', TO_FILE('{STAGE_PATH}', '{stage_file_path}'))"
        cursor.execute(sql_query)
        result = cursor.fetchone()
        return str(result[0]) if result else "No description returned."
    except Exception as e: return f"ERROR_VISION: {str(e)}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass 
        if conn: conn.close()

def smart_router(user_text, image_desc):
    """
    Optimized Intent Classification
    """
    # Quick Keyword Check (Bypass LLM for speed)
    greetings = ['hi', 'hello', 'hey', 'p', 'halo', 'test', 'pagi', 'siang', 'malam']
    if len(user_text.split()) < 3 and user_text.lower().strip() in greetings:
        return {"is_footwear": True, "intent": "CHAT"}

    system_prompt = """
    You are a Smart Router.
    INPUTS: User Text: "{user_text}", Image Desc: "{image_desc}"
    RULES:
    1. IS_FOOTWEAR: False if image describes car/animal/food. True if shoes or No image.
    2. INTENT: 
       - SEARCH: User explicitly asks to FIND, BUY, SHOW, RECOMMEND products (e.g. "red shoes", "price?", "boots").
       - CHAT: Greetings ("Hi"), General talk ("How are you"), Questions NOT about buying ("Who made you?").
       - If is_footwear=False -> CHAT.
    OUTPUT JSON: {{"is_footwear": true/false, "intent": "SEARCH" or "CHAT"}}
    """
    try:
        prompt = ChatPromptTemplate.from_template(system_prompt)
        llm = get_llm_cortex() 
        chain = prompt | llm | JsonOutputParser()
        return chain.invoke({"user_text": user_text, "image_desc": image_desc})
    except:
        # Fallback Logic: If text is short, assume CHAT. If long, assume SEARCH.
        if len(user_text) < 15:
            return {"is_footwear": True, "intent": "CHAT"}
        return {"is_footwear": True, "intent": "SEARCH"}

@st.dialog("✨ Product Details")
def show_product_popup(product):
    render_product_image(product['IMAGE_FILENAME'], use_container_width=True)
    st.markdown(f"### {product['TITLE']}")
    st.caption(f"Brand: {product['BRAND']}")
    col_price, col_btn = st.columns([1,1])
    with col_price: st.markdown(f"### {product['PRICE']}")
    with col_btn: 
        if st.button("🛒 Add to Cart", key=f"cart_pop_{product['TITLE'][:10]}", type="primary", use_container_width=True):
            st.toast(f"✅ **{product['TITLE']}** added to cart!", icon="🛒")
    st.divider()
    st.markdown("**Description & Features:**")
    st.info(product['PRODUCT_DETAILS_CLEAN'])

# ==========================================
# 4. SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.markdown("### 👟 About SoleMate")
    st.markdown(
        "SoleMate AI is your intelligent shopping assistant powered by Snowflake Cortex and Voyage AI. "
        "We combine Visual Search and Smart Chat to help you find the perfect footwear instantly."
    )
    st.divider()

    if st.session_state.page == "home":
        if st.button("💬 Ask SoleMate", type="primary", use_container_width=True):
            switch_page("chatbot")
    else:
        if st.button("🏠 Back to Home", type="secondary", use_container_width=True):
            switch_page("home")
        st.divider()
        with st.popover("📎 Attach Image", use_container_width=True):
            st.markdown("### Upload Photo")
            uploaded_file_sidebar = st.file_uploader("Choose a file...", type=["jpg", "png", "jpeg"], key="sidebar_uploader")
            if uploaded_file_sidebar:
                st.image(uploaded_file_sidebar, caption="Ready to analyze!", use_container_width=True)
                st.caption("ℹ️ *Type in chat to submit.*")
        if st.button("🔄 Reset Chat", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi! I'm SoleMate. Upload a photo or describe what you're looking for!", "type": "text"}
            ]
            st.session_state.image_cache = {}
            st.rerun()

# ==========================================
# 5. PAGE 1: HOME (PRODUCT GALLERY)
# ==========================================
if st.session_state.page == "home":
    st.title("🛍️ Featured Collection")
    st.caption("Explore our latest arrivals. Click 'Ask SoleMate' to find something specific!")
    
    @st.cache_data(ttl=600)
    def get_home_products():
        vector = get_text_embedding("stylish footwear sneakers boots") 
        df = search_products_by_vector(vector, limit=50) 
        if not df.empty: return df.sample(n=min(20, len(df)), random_state=int(time.time()))
        return pd.DataFrame()

    home_products = get_home_products()
    
    if not home_products.empty:
        fetch_images_batch(home_products['IMAGE_FILENAME'].tolist())
        cols = st.columns(5)
        for idx, (index, row) in enumerate(home_products.iterrows()):
            with cols[idx % 5]:
                with st.container(border=True):
                    render_product_image(row['IMAGE_FILENAME'], use_container_width=True)
                    st.markdown(f"<div class='product-card-title' title='{row['TITLE']}'>{row['TITLE']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='product-card-price'>{row['PRICE']}</div>", unsafe_allow_html=True)
                    if st.button("View", key=f"home_btn_{idx}", use_container_width=True):
                        show_product_popup(row)
    else:
        st.info("Loading collection...")

# ==========================================
# 6. PAGE 2: CHATBOT INTERFACE
# ==========================================
elif st.session_state.page == "chatbot":
    st.title("💬 Chat with SoleMate")
    
    # 1. Load History Images
    all_history_images = []
    for msg in st.session_state.messages:
        if msg.get("role") == "assistant" and msg.get("data") is not None and not msg["data"].empty:
            files = msg["data"]['IMAGE_FILENAME'].tolist()
            all_history_images.extend(files)
    if all_history_images: fetch_images_batch(list(set(all_history_images)))

    # 2. Render Chat History
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg.get("thought"):
                with st.expander("🧠 AI Reasoning (Thinking Process)", expanded=False):
                    st.markdown(f"<div class='reasoning-box'>{msg['thought']}</div>", unsafe_allow_html=True)
            st.markdown(msg["content"])
            if msg.get("role") == "assistant" and msg.get("data") is not None and not msg["data"].empty:
                st.markdown("---")
                st.caption(f"👟 Found {len(msg['data'])} Recommendations:")
                df_products = msg["data"]
                cols = st.columns(5) 
                for idx, row in df_products.iterrows():
                    with cols[idx % 5]:
                        with st.container(border=True):
                            render_product_image(row['IMAGE_FILENAME'], use_container_width=True)
                            st.markdown(f"<div class='product-card-title' title='{row['TITLE']}'>{row['TITLE'][:18]}..</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='product-card-price'>{row['PRICE']}</div>", unsafe_allow_html=True)
                            if st.button("View", key=f"chat_btn_{i}_{idx}", use_container_width=True):
                                show_product_popup(row)

    # 3. Input Logic
    uploaded_file = st.session_state.get("sidebar_uploader")
    input_text = st.chat_input("Ask SoleMate")

    if input_text:
        content_text = input_text
        if uploaded_file: content_text = f"🖼️ [Image Attached] {input_text}"
        st.session_state.messages.append({"role": "user", "content": content_text})
        st.rerun()

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        last_msg = st.session_state.messages[-1]["content"]
        
        with st.chat_message("assistant"):
            with st.status("🧠 SoleMate is thinking...", expanded=True) as status:
                
                # Vision
                image_desc = "No image uploaded."
                if uploaded_file and "🖼️" in last_msg:
                    st.write("👁️ Analyzing Image...")
                    image_desc = analyze_image_with_cortex(uploaded_file)
                    st.write("✅ Image Analyzed.")
                
                # Routing (OPTIMIZED)
                st.write("🚦 Routing Intent...")
                router_res = smart_router(last_msg, image_desc)
                intent = router_res.get("intent", "CHAT")
                is_footwear = router_res.get("is_footwear", True)
                st.write(f"   → Intent: {intent}")

                # Logic Flow
                products_df = pd.DataFrame()
                context_str = "[]"
                
                if not is_footwear or intent == "CHAT":
                    # SKIP SEARCHING DB IF CHAT
                    st.write("💬 Conversational Mode (Skipping Search)")
                
                elif intent == "SEARCH":
                    st.write("🔍 Searching Database...")
                    vector = []
                    if uploaded_file: vector = get_image_embedding_from_bytes(uploaded_file)
                    else:
                        search_query = last_msg
                        if "No image" not in image_desc: search_query += f" ({image_desc})"
                        vector = get_text_embedding(search_query)

                    if len(vector) > 0:
                        products_df = search_products_by_vector(vector, limit=15)
                        if not products_df.empty:
                            fetch_images_batch(products_df['IMAGE_FILENAME'].tolist())
                            context_str = format_context_json(products_df)
                            st.write(f"✅ Found {len(products_df)} items.")

                # Generation
                st.write("✍️ Drafting Response...")
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """
                    You are SoleMate, an expert footwear consultant.
                    
                    CONTEXT: {context}
                    IMAGE_DESC: "{img_desc}"
                    INTENT: {intent}
                    IS_FOOTWEAR: {is_footwear}
                    
                    INSTRUCTIONS:
                    1. If IS_FOOTWEAR=False, politely explain you only deal with shoes.
                    2. If INTENT=CHAT, reply naturally. Do not hallucinate products.
                    3. If INTENT=SEARCH:
                       - Select Top 3 products from CONTEXT.
                       - For EACH product, write:
                         * **Why it fits:** Specific reason based on request.
                         * **Key Features:** Material/Tech.
                         * **Best For:** Ideal occasion.
                    
                    OUTPUT JSON: {{ "classification": "recommendation"|"chat", "thought": "Step-by-step reasoning...", "response_text": "Markdown string...", "recommended_products": [] }}
                    """),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}")
                ])
                
                llm = get_llm_cortex()
                chain = prompt | llm | JsonOutputParser()
                chain_with_history = RunnableWithMessageHistory(
                    chain, get_session_history, input_messages_key="input", history_messages_key="chat_history", output_messages_key="response_text"
                )
                
                try:
                    res = chain_with_history.invoke(
                        {"input": last_msg, "context": context_str, "img_desc": image_desc, "is_footwear": is_footwear, "intent": intent},
                        config={"configurable": {"session_id": "user_session_1"}}
                    )
                    status.update(label="✅ Done!", state="complete", expanded=False)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()
            
            # FINAL RENDER (Outside Status)
            if res.get('thought'):
                with st.expander("🧠 AI Reasoning (Thinking Process)", expanded=False):
                    st.markdown(f"**Thought Process:**\n{res.get('thought')}")
            
            st.markdown(res.get("response_text"))
            
            show_grid = False
            if res.get('classification') == 'recommendation' and not products_df.empty:
                show_grid = True
                
            if show_grid:
                st.markdown("---")
                st.markdown("### 🛍️ Explore Recommendations")
                cols = st.columns(5)
                for idx, row in products_df.iterrows():
                    with cols[idx % 5]:
                        with st.container(border=True):
                            render_product_image(row['IMAGE_FILENAME'], use_container_width=True)
                            st.markdown(f"<div class='product-card-title' title='{row['TITLE']}'>{row['TITLE'][:18]}..</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='product-card-price'>{row['PRICE']}</div>", unsafe_allow_html=True)
                            if st.button("View", key=f"now_btn_{idx}", use_container_width=True):
                                show_product_popup(row)

            st.session_state.messages.append({
                "role": "assistant", 
                "content": res.get("response_text"),
                "thought": res.get("thought"),
                "data": products_df if show_grid else None
            })