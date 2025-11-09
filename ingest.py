import os
import shutil
from dotenv import load_dotenv

# --- RAG Core Libraries ---
# Used for document loading and advanced splitting
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ChromaDB is your Vector Database
from langchain_community.vectorstores import Chroma
# sentence-transformers is the recommended local Embedding Model
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Configuration ---
load_dotenv()
DATA_PATH = "data"         # Source folder for documents
CHROMA_PATH = "chroma_db"  # Destination folder for the vector database

# --- 1. Multimodal Document Loading Function ---
def load_documents():
    """
    Loads all documents from the DATA_PATH directory using UnstructuredFileLoader.
    This handles complex files like PDFs and attempts OCR for embedded images/scans.
    """
    print(f"Loading documents from {DATA_PATH}...")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data directory '{DATA_PATH}' not found. Please create it and add your documents.")
        return []

    documents = []
    # Loop through every file in the data folder
    for filename in os.listdir(DATA_PATH):
        filepath = os.path.join(DATA_PATH, filename)
        if os.path.isfile(filepath) and not filename.startswith('.'): # Ignore hidden files
            try:
                # UnstructuredFileLoader is great for multimodal data
                # This corresponds to the 'preprocessing' and 'data cleaning' box in your diagram
                loader = UnstructuredFileLoader(filepath)
                docs = loader.load()
                
                # Add file name as metadata for source tracking
                for doc in docs:
                    doc.metadata['source_file'] = filename
                documents.extend(docs)
                print(f"  -> Loaded {len(docs)} segments from {filename}")
            except Exception as e:
                print(f"  -> ERROR loading {filename}: {e}")
                
    print(f"Total document segments loaded: {len(documents)}")
    return documents

# --- 2. Semantic Chunking Function ---
def split_documents(documents: list):
    """
    Splits documents into semantically meaningful chunks using a combination of
    structural and semantic boundaries while maintaining context.
    """
    # Use a semantic text splitter that respects sentence and paragraph boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,          # Smaller chunks for more semantic coherence
        chunk_overlap=50,        # Minimal overlap to maintain context
        length_function=len,     # Standard length calculation
        separators=[            # Ordered by semantic significance
            "\n## ",           # Section headings
            "\n### ",          # Subsection headings
            "\n\n",           # Paragraphs
            "\n",             # Line breaks
            ". ",             # Sentences
            "? ",             # Questions
            "! ",             # Exclamations
            ", ",             # Clauses
            " ",              # Words
            ""                # Characters
        ],
        is_separator_regex=False
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} semantic chunks for embedding.")
    return chunks

# --- 3. Embedding and Indexing Function ---
def index_documents(chunks):
    """
    Converts text chunks into vectors and stores them in ChromaDB (Vector Database).
    (Corresponds to 'data ingestion pipeline' -> 'Store in Vector DB' in your diagram)
    """
    # Clean up any previous database to ensure freshness
    print("Deleting old ChromaDB content if it exists...")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        
    print("Creating new vector store and generating embeddings...")
    
    # HuggingFaceEmbeddings uses the sentence-transformers model you installed
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create and persist the Vector Store (ChromaDB)
    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print(f"Successfully indexed {len(chunks)} chunks in ChromaDB at '{CHROMA_PATH}'")

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created data directory '{DATA_PATH}'. Please add your PDF files here.")
    else:
        documents = load_documents()
        if documents:
            chunks = split_documents(documents)
            index_documents(chunks)
            print("\nData ingestion complete! Your vector database is ready.")