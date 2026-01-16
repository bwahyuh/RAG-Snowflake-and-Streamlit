# 👟 SoleMate AI

<div align="center">

![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Snowflake](https://img.shields.io/badge/Snowflake-29B5E8?style=for-the-badge&logo=Snowflake&logoColor=white)
![Voyage AI](https://img.shields.io/badge/Voyage%20AI-Embeddings-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Next-Gen Footwear Concierge powered by Multimodal RAG**

---

## 📖 Overview

**SoleMate AI** is an intelligent e-commerce assistant designed to revolutionize how users discover footwear. Unlike traditional keyword search, SoleMate combines **Visual Search**, **Semantic Understanding**, and **Generative AI** to provide a human-like shopping experience.

Users can upload a photo of a shoe they see on the street, or describe what they want in natural language (e.g., *"Comfortable sneakers for marathon training"*), and SoleMate will find the closest matches from the inventory using state-of-the-art vector similarity.

## ✨ Key Features

* **👁️ Native Visual Discovery**: Leverages **Snowflake Cortex Vision** (via `TO_FILE`) to analyze uploaded images directly within the Data Cloud, identifying style, color, and material without external processing.
* **🧠 Smart Router**: An intelligent routing system that classifies user intent (Search vs. Chat) and distinguishes between footwear-related queries and general chitchat to optimize performance.
* **🔍 Multimodal RAG**: Powered by **Voyage AI (`voyage-multimodal-3`)** embeddings, allowing seamless searching across text and images simultaneously.
* **💬 Expert Consultation**: Uses **Claude 3.5 Sonnet** via Snowflake Cortex to act as a domain expert, providing detailed reasoning on *why* a specific product fits the user's needs.
* **🛒 Interactive UI**: Features a dynamic Home Gallery, Chatbot interface, and interactive "Add to Cart" notifications.

## 🏗️ Architecture

SoleMate AI is built on a modern **Serverless RAG** architecture:

```mermaid
graph TD
    User[User] -->|Uploads Image/Text| UI[Streamlit Interface]
    UI -->|Router Logic| Router{Smart Router}
    
    Router -->|Chat Intent| LLM[Cortex LLM]
    Router -->|Search Intent| Embedder[Voyage AI Client]
    
    Embedder -->|Generate Vector| Vector[Multimodal Embedding]
    Vector -->|Cosine Similarity| DB[(Snowflake Database)]
    
    DB -->|Top K Results| Context[RAG Context]
    Context -->|Augment| LLM
    
    LLM -->|Final Response| UI


🛠️ Tech Stack
Frontend Framework: Streamlit

LLM & Vision: Snowflake Cortex (Claude 3.5 Sonnet)

Embeddings: Voyage AI (Multimodal-3)

Vector Database: Snowflake (Vector Data Type)

Orchestration: LangChain

Data Connector: snowflake-connector-python

📂 Project Structure
Plaintext

SoleMate-AI/
├── modules/
│   ├── database.py       # Snowflake connection & Vector Search logic
│   ├── embedder.py       # Voyage AI Client for multimodal embeddings
│   └── llm.py            # Cortex Chat Model initialization
├── .gitignore            # Git ignore rules
├── LICENSE               # MIT License
├── README.md             # Documentation
├── logo.png              # App branding asset
├── main.py               # Main application entry point
└── requirements.txt      # Python dependencies
🚀 Getting Started
Prerequisites
Python 3.10+

A Snowflake account with Cortex enabled.

A Voyage AI API Key.

A Snowflake Stage created for image storage (@RETAIL_GENAI_DB.PUBLIC.PRODUCT_IMAGES_STAGE).

Installation
Clone the repository

Bash

git clone [https://github.com/yourusername/SoleMate-AI.git](https://github.com/yourusername/SoleMate-AI.git)
cd SoleMate-AI
Create a virtual environment

Bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

Bash

pip install -r requirements.txt
Configure Secrets Create a .streamlit/secrets.toml file in the root directory:

Ini, TOML

[snowflake]
user = "YOUR_USER"
password = "YOUR_PASSWORD"
account = "YOUR_ACCOUNT_LOCATOR"
warehouse = "YOUR_WAREHOUSE"
database = "RETAIL_GENAI_DB"
schema = "PUBLIC"
role = "YOUR_ROLE"

[voyage]
api_key = "YOUR_VOYAGE_API_KEY"
Run the application

Bash

streamlit run main.py
☁️ Deployment
To deploy on Streamlit Community Cloud:

Push your code to GitHub.

Login to share.streamlit.io.

Click New App and select your repository.

Go to Advanced Settings within the deployment dashboard.

Copy the contents of your local secrets.toml and paste them into the Secrets field.

Click Deploy! 🚀

📜 License
Distributed under the MIT License. See LICENSE for more information.

👤 Author
Bagas Wahyu Herdiansyah

Copyright: © 2026
