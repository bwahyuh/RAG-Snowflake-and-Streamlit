import streamlit as st
import snowflake.connector
import pandas as pd
import json

def get_db_connection():
    """
    Establishes a connection to Snowflake using credentials from secrets.toml.
    """
    return snowflake.connector.connect(
        **st.secrets["snowflake"]
    )

def search_products_by_vector(query_vector, limit=5):
    """
    Searches for similar products using Snowflake's VECTOR_COSINE_SIMILARITY function.
    Returns a DataFrame containing the top N most similar products.
    """
    conn = get_db_connection()
    try:
        # FIX APPLIED:
        # 1. We dump the list to a JSON string in Python to be safe.
        # 2. In SQL, we use PARSE_JSON() before casting to VECTOR.
        # This resolves the "Unsupported data type 'FIXED'" error.
        
        vector_json = json.dumps(query_vector)
        
        sql = f"""
        SELECT 
            TITLE, 
            BRAND, 
            PRICE, 
            PRODUCT_DETAILS_CLEAN,
            IMAGE_FILENAME,
            VECTOR_COSINE_SIMILARITY(VECTOR_TEXT, CAST(PARSE_JSON(%s) AS VECTOR(FLOAT, 1024))) as SIMILARITY_SCORE
        FROM PRODUCTS_FINAL
        ORDER BY SIMILARITY_SCORE DESC
        LIMIT {limit}
        """
        
        # We pass the JSON string, not the raw list
        df = pd.read_sql(sql, conn, params=[vector_json])
        return df
        
    except Exception as e:
        st.error(f"❌ Database Error: {e}")
        return pd.DataFrame() 
        
    finally:
        conn.close()