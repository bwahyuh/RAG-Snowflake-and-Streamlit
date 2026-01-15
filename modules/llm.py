import streamlit as st
from langchain_community.chat_models import ChatSnowflakeCortex
from modules.database import get_db_connection

def get_llm_cortex():
    
    creds = st.secrets["snowflake"]

    chat = ChatSnowflakeCortex(
        model="claude-3-5-sonnet",
        cortex_function="complete",
        temperature=0.7,
        snowflake_account=creds["account"],
        snowflake_username=creds["user"],
        snowflake_password=creds["password"],
        snowflake_database=creds["database"],
        snowflake_schema=creds["schema"],
        snowflake_warehouse=creds["warehouse"],
        snowflake_role=creds["role"]
    )
    
    return chat