# ingest.py
"""
Ingest documents from the DATA_PATH into a Chroma vector DB.
- Deletes existing CHROMA_PATH (safe rebuild)
- Adds feedback_score: 0 metadata to each chunk
- Uses SentenceTransformer (EMBED_MODEL) to embed chunks
- Stores documents + metadata in Chroma collection "support_docs"
"""

import os
import shutil
import glob
from dotenv import load_dotenv
from config import CHROMA_PATH, DATA_PATH, EMBED_MODEL
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ---------- Parameters ----------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))        # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))   # overlap between chunks
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "support_docs")

# ---------- Load environment ----------
load_dotenv()

# ---------- Utility functions ----------
def read_text_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def safe_load_document(path):
    """
    Try to load using unstructured if available for PDFs/DOCX,
    otherwise fallback to raw text read.
    """
    try:
        from unstructured.partition.auto import partition
        elems = partition(filename=path)
        text = "\n\n".join([str(el).strip() for el in elems if str(el).strip()])
        if text:
            return text
    except Exception:
        try:
            return read_text_file(path)
        except Exception:
            return ""

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Simple character-based chunker with overlap.
    Returns list of text chunks.
    """
    if not text:
        return []
    text = text.replace("\r\n", "\n")
    length = len(text)
    chunks = []
    start = 0
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    # remove empty chunks and duplicates
    cleaned = []
    seen = set()
    for c in chunks:
        if not c:
            continue
        key = c[:80]
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(c)
    return cleaned

# ---------- Main ingestion ----------
def build_embeddings_and_index():
    print("Starting ingestion...")

    # ensure data folder exists
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created DATA_PATH at '{DATA_PATH}'. Place files to ingest and run again.")
        return

    # remove old chroma DB if exists
    if os.path.exists(CHROMA_PATH):
        print(f"Deleting existing Chroma DB at '{CHROMA_PATH}' ...")
        shutil.rmtree(CHROMA_PATH)

    # initialize embedder
    print(f"Loading embedder model: {EMBED_MODEL} ...")
    embedder = SentenceTransformer(EMBED_MODEL)

    # gather files
    patterns = ["*.*"]  # all files in data
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(DATA_PATH, p)))
    files = [f for f in files if os.path.isfile(f) and not os.path.basename(f).startswith(".")]

    if not files:
        print(f"No files found in '{DATA_PATH}'. Place text/PDF/docx files and run again.")
        return

    docs = []
    metadatas = []
    ids = []

    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"\nLoading: {filename}")
        text = safe_load_document(filepath)
        if not text:
            print(f" - Skipped (no text extracted): {filename}")
            continue

        chunks = chunk_text(text)
        print(f" - Extracted {len(chunks)} chunks from {filename}")

        for i, c in enumerate(chunks):
            doc_id = f"{filename}--{i}"
            docs.append(c)
            metadatas.append({
                "source": filename,
                "chunk_index": i,
                # default feedback score for new chunks
                "feedback_score": 0
            })
            ids.append(doc_id)

    if not docs:
        print("No chunks to index.")
        return

    # encode all embeddings in batches
    print("\nEncoding embeddings...")
    embeddings = embedder.encode(docs, show_progress_bar=True, batch_size=32)

    # ensure chroma client persists to CHROMA_PATH
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
    except Exception:
        client = chromadb.Client(Settings(persist_directory=CHROMA_PATH))

    # create or get collection
    try:
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception:
        try:
            collection = client.create_collection(name=COLLECTION_NAME)
        except Exception as e:
            raise RuntimeError(f"Unable to create/get Chroma collection: {e}")

    print(f"\nUpserting {len(docs)} documents into Chroma collection '{COLLECTION_NAME}' ...")
    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings
    )

    # persist
    try:
        client.persist()
    except Exception:
        pass

    print("\nIngestion complete. Chroma DB at:", CHROMA_PATH)
    print(f"Total documents indexed: {len(docs)}")

if __name__ == "__main__":
    build_embeddings_and_index()
