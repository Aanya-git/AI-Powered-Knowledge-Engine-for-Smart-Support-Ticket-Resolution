# ingest.py
import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

def load_documents():
    """Load all documents from DATA_PATH and extract structured content."""
    print(f"\n Loading documents from '{DATA_PATH}'...")

    if not os.path.exists(DATA_PATH):
        print(f"  Data directory '{DATA_PATH}' not found. Please create it and add files.")
        return []

    documents = []
    for filename in os.listdir(DATA_PATH):
        filepath = os.path.join(DATA_PATH, filename)
        if os.path.isfile(filepath) and not filename.startswith('.'):
            try:
                loader = UnstructuredFileLoader(filepath)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                documents.extend(docs)
                print(f" Loaded {len(docs)} segments from {filename}")
            except Exception as e:
                print(f" Error loading {filename}: {e}")
    print(f" Total documents loaded: {len(documents)}")
    return documents


def split_documents_semantic(documents):
    """Split documents semantically using embeddings similarity."""
    print("\n Splitting documents semantically...")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=90)
    # You can tune breakpoint_threshold_amount (lower = smaller chunks, higher = larger semantic chunks)

    chunks = text_splitter.split_documents(documents)
    print(f" Created {len(chunks)} semantically coherent chunks.")
    return chunks


def index_documents(chunks):
    """Embed text and store it in ChromaDB."""
    print("\n Creating and storing embeddings in ChromaDB...")

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(" Old ChromaDB removed for fresh indexing.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH,
    )

    print(f" Indexed {len(chunks)} chunks in ChromaDB at '{CHROMA_PATH}'.")


if __name__ == "__main__":
    print("\n Starting Semantic Document Ingestion Pipeline...")
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f" Created '{DATA_PATH}'. Add your documents and rerun.")
    else:
        docs = load_documents()
        if docs:
            chunks = split_documents_semantic(docs)
            index_documents(chunks)
            print("\n Ingestion complete! Vector DB ready for RAG use.")
