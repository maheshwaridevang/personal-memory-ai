# ingest.py
import os
import docx
from PyPDF2 import PdfReader
from utils.chunking import chunk_text
from utils.embedding import embed_text
import chromadb
from chromadb.config import Settings

DATA_DIR = "data"
DB_DIR = "db"
COLLECTION_NAME = "memory"

# Use in-memory Chroma client on Streamlit Cloud
if os.environ.get("IS_STREAMLIT_CLOUD", "false").lower() == "true":
    client = chromadb.Client()
else:
    from chromadb import PersistentClient
    client = PersistentClient(path=DB_DIR)

collection = client.get_or_create_collection(name=COLLECTION_NAME)

def ingest_single_file(filename, text):
    chunks = chunk_text(text)
    if len(chunks) == 0:
        print(f"⚠️ No chunks found in {filename}")
        return

    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    existing = collection.get(ids=ids)["ids"]
    if existing:
        print(f"⏩ Skipping {filename} (already ingested)")
        return

    embeddings = embed_text(chunks)
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    print(f"✅ Ingested {len(chunks)} chunks from {filename}")
