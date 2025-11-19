import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
from groq import Groq

# -------------------------------------------------
# LOAD ENVIRONMENT
# -------------------------------------------------
load_dotenv()
st.set_page_config(page_title=" BookMyShow AI Assistant", layout="wide")

# -------------------------------------------------
# PAGE HEADER
# -------------------------------------------------
st.title(" BookMyShow AI Knowledge Assistant")
st.caption("Built with Groq + Streamlit + ChromaDB + Semantic Chunking")

# -------------------------------------------------
# SIDEBAR SETTINGS
# -------------------------------------------------
st.sidebar.header(" Configuration")
api_key = st.sidebar.text_input(" Enter your GROQ API Key", type="password", value=os.getenv("GROQ_API_KEY"))
collection_name = st.sidebar.text_input(" Collection Name", "bookmyshow_semantic_docs")

# -------------------------------------------------
# INITIALIZE CLIENTS
# -------------------------------------------------
chroma_client = chromadb.Client(Settings(persist_directory="./chroma_db"))
collection = chroma_client.get_or_create_collection(collection_name)

if not api_key:
    st.sidebar.warning("Please enter your GROQ API key to continue.")
else:
    groq_client = Groq(api_key=api_key)

# -------------------------------------------------
# PDF UPLOAD SECTION
# -------------------------------------------------
st.subheader(" Upload BookMyShow PDF Documents")

uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
os.makedirs("docs", exist_ok=True)

# Initialize embedding model for semantic chunking
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
semantic_splitter = SemanticChunker(embedding_model)

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

def add_to_chroma(text, source):
    st.info(f" Creating semantic chunks for {source}...")
    chunks = semantic_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            metadatas=[{"source": source}],
            ids=[f"{source}_{i}"]
        )
    st.success(f" Indexed {len(chunks)} semantic chunks from {source}")

if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join("docs", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        text = extract_text_from_pdf(file)
        add_to_chroma(text, file.name)

# -------------------------------------------------
# USER QUERY SECTION
# -------------------------------------------------
st.markdown("---")
st.subheader(" Ask Your AI Assistant")

query = st.text_area("Type your question about BookMyShow:", height=100)

if st.button(" Get AI Answer"):
    if not api_key:
        st.error("Please enter your Groq API key first!")
    elif not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Analyzing your question... "):
            try:
                # Retrieve context from ChromaDB
                results = collection.query(query_texts=[query], n_results=3)
                docs = results["documents"][0] if results["documents"] else []
                context = "\n\n".join(docs) if docs else "No relevant context found."

                # Call Groq AI
                response = groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": "You are an AI assistant specialized in BookMyShow support queries. Use the provided context when answering."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                    ]
                )

                answer = response.choices[0].message.content
                st.success(" AI Answer:")
                st.markdown(f"### {answer}")

                with st.expander(" Retrieved Context"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**{i+1}.** {doc}")

            except Exception as e:
                st.error(f" Error: {e}")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption(" Powered by Groq + Streamlit + Semantic Chunking + ChromaDB")
