"""
ingest.py — Semantic Chunking + ChromaDB Indexing

This script:
1. Loads files from DATA_PATH
2. Extracts text (PDF/DOCX via unstructured, others via plain read)
3. Splits text using SEMANTIC CHUNKING (sentence embeddings)
4. Embeds chunks using SentenceTransformer
5. Stores them in Chroma with metadata + feedback_score = 0
"""

import os
import glob
import shutil
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "support_docs")

MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", 350))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.65))

# -----------------------------
# NLTK Sentence Tokenizer
# -----------------------------
import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt", quiet=True)


# -----------------------------
# File Reading Utilities
# -----------------------------
def read_text_file(file_path):
    """Fallback raw text loader."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except:
        return ""


def extract_text(file_path):
    """
    Extracts text using unstructured for PDFs / DOCX.
    Falls back to plain text reader otherwise.
    """
    try:
        from unstructured.partition.auto import partition
        elements = partition(filename=file_path)
        text = "\n".join([str(e).strip() for e in elements if str(e).strip()])
        if text:
            return text
    except:
        pass

    return read_text_file(file_path)


# -----------------------------
# Semantic Chunking
# -----------------------------
def semantic_chunk_text(text, embedder):
    """
    Semantic chunking using:
    - Sentence tokenization
    - Embedding similarity
    - Token-length limits
    """
    if not text or len(text.strip()) == 0:
        return []

    sentences = sent_tokenize(text)

    if len(sentences) == 0:
        return []

    sentence_embeddings = embedder.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]
    current_tokens = len(sentences[0].split())
    last_embedding = sentence_embeddings[0]

    for i in range(1, len(sentences)):
        sentence = sentences[i]
        emb = sentence_embeddings[i]
        similarity = np.dot(emb, last_embedding) / (np.linalg.norm(emb) * np.linalg.norm(last_embedding))

        # Start a new chunk if:
        # - semantic similarity is low
        # - chunk would get too long
        if similarity < SIMILARITY_THRESHOLD or current_tokens + len(sentence.split()) > MAX_CHUNK_TOKENS:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_tokens = len(sentence.split())
        else:
            current_chunk.append(sentence)
            current_tokens += len(sentence.split())

        last_embedding = emb

    chunks.append(" ".join(current_chunk))
    return chunks


# -----------------------------
# Main ingestion pipeline
# -----------------------------
def build_embeddings_and_index():
    print("\n Starting Semantic Ingestion...")

    # Ensure data folder exists
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f" Created DATA_PATH at '{DATA_PATH}'. Add your files and run again.")
        return

    # Cleanup existing Chroma DB
    if os.path.exists(CHROMA_PATH):
        print(f" Removing old Chroma database at '{CHROMA_PATH}'...")
        shutil.rmtree(CHROMA_PATH)

    # Load embedder
    print(f"\n Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    # Get list of files
    files = glob.glob(os.path.join(DATA_PATH, "*"))
    files = [f for f in files if os.path.isfile(f) and not os.path.basename(f).startswith(".")]

    if not files:
        print(f" No files found in '{DATA_PATH}'. Add files and try again.")
        return

    docs = []
    metadatas = []
    ids = []

    # Process each file
    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"\n Processing: {filename}")

        text = extract_text(file_path)
        if not text:
            print(f" No text extracted. Skipping {filename}")
            continue

        chunks = semantic_chunk_text(text, embedder)
        print(f" Created {len(chunks)} semantic chunks")

        for i, chunk in enumerate(chunks):
            docs.append(chunk)
            metadatas.append({
                "source": filename,
                "chunk_index": i,
                "feedback_score": 0
            })
            ids.append(f"{filename}--{i}")

    if not docs:
        print(" No documents to index.")
        return

    print("\n Encoding embeddings for all chunks...")
    embeddings = embedder.encode(docs, batch_size=32, show_progress_bar=True)

    # Initialize Chroma
    print(f"\n Creating ChromaDB at: {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print(f"\n Upserting {len(docs)} chunks into Chroma collection '{COLLECTION_NAME}'...")
    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print("\n Ingestion complete!")
    print(f" Chunks stored: {len(docs)}")
    print(f" Chroma path: {CHROMA_PATH}")


# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    build_embeddings_and_index()
