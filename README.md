# Local RAG Assistant (Streamlit + FAISS + Ollama)

A lightweight **local Retrieval-Augmented Generation (RAG) system** that runs entirely on your machine.  
It processes uploaded documents (PDF, DOCX, PPTX, TXT), generates embeddings locally, stores them as `.pkl` files, builds a FAISS vector index, and uses **Ollama** (Gemma, Mistral, etc.) to generate answers **strictly grounded** in your documents.

No external API calls — everything runs offline.

 ## Requirements

### Install dependencies:

pip install requirements.txt

### ⚠️ NOTE — Important

The embedding model and LLM model must be downloaded manually.

Specifically:

#### 1️⃣ Download embedding model

This project expects the folder:

./bge-large-en-v1.5/

 Download it from HuggingFace:

🔗 https://huggingface.co/BAAI/bge-large-en-v1.5

Extract it into the project directory.

#### 2️⃣ Install Ollama + the LLM model

Install Ollama:

https://ollama.com/download

Then pull the model:

ollama pull gemma:2b

#### 3️⃣ How to run:

streamlit run app.py

