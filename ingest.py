# ingest.py
import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load .env
load_dotenv()

# Paths
DATA_PATH = os.getenv("DATA_PATH", "data")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")

# OpenRouter Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "google/text-embedding-004")  # Gemini embedding model

def load_documents():
    """Load documents from /data folder"""
    print(f"\nLoading documents from '{DATA_PATH}'...")

    docs = []
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"DATA folder was missing — created '{DATA_PATH}'. Place files and run again.")
        return []

    for filename in os.listdir(DATA_PATH):
        filepath = os.path.join(DATA_PATH, filename)
        if os.path.isfile(filepath) and not filename.startswith("."):
            loader = UnstructuredFileLoader(filepath)
            file_docs = loader.load()
            for d in file_docs:
                d.metadata["source"] = filename
            docs.extend(file_docs)
            print(f" Loaded {len(file_docs)} chunks from {filename}")

    print(f"Total text chunks extracted: {len(docs)}")
    return docs


def split_documents_semantic(docs):
    """Semantic chunker (MiniLM for chunking logic)"""
    print("\nSplitting documents into semantic chunks...")

    hf_embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    chunker = SemanticChunker(hf_embedder)

    chunks = chunker.split_documents(docs)
    print(f"Created {len(chunks)} meaningful chunks")
    return chunks


def index_documents(chunks):
    """Create embeddings using Gemini via OpenRouter & store in Chroma"""
    print("\nGenerating embeddings via Gemini (OpenRouter)...")

    # Remove old DB
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Old vector DB deleted.")

    # Embedding engine (Gemini via OpenRouter)
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"Indexed {len(chunks)} chunks into Chroma DB at '{CHROMA_PATH}'")


if __name__ == "__main__":
    print("\nStarting Knowledge Base Builder...")

    documents = load_documents()
    if documents:
        chunks = split_documents_semantic(documents)
        index_documents(chunks)
        print("\nKnowledge Base successfully created! RAG is ready.")
