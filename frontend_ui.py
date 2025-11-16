import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# LangChain components (Python 3.8 compatible versions installed)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

import chromadb
from chromadb.config import Settings

# Groq LLM Client
from groq import Groq

# -------------------------------------------------
# LOAD ENVIRONMENT
# -------------------------------------------------
load_dotenv()
st.set_page_config(page_title="BookMyShow AI Assistant", layout="wide")

# -------------------------------------------------
# PAGE HEADER
# -------------------------------------------------
st.title(" BookMyShow AI Knowledge Assistant")

# -------------------------------------------------
# SIDEBAR SETTINGS
# -------------------------------------------------
st.sidebar.header(" Configuration")

default_api_key = os.getenv("GROQ_API_KEY") or ""

api_key = st.sidebar.text_input(
    "Enter your GROQ API Key",
    type="password",
    value=default_api_key
)

collection_name = st.sidebar.text_input("Collection Name", "bookmyshow_semantic_docs")

# -------------------------------------------------
# INITIALIZE ChromaDB
# -------------------------------------------------
chroma_client = chromadb.Client(Settings(persist_directory="./chroma_db"))
collection = chroma_client.get_or_create_collection(collection_name)

# Initialize Groq only when key exists
groq_client = None
if api_key.strip() != "":
    groq_client = Groq(api_key=api_key)
else:
    st.sidebar.warning(" Enter your Groq API key to enable AI responses.")

# -------------------------------------------------
# PDF UPLOAD SECTION
# -------------------------------------------------
st.subheader(" Upload BookMyShow PDF Documents")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# Ensure docs directory exists
if not os.path.exists("docs"):
    os.makedirs("docs")

# Load embedding model (Python 3.8 compatible)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

semantic_splitter = SemanticChunker(embedding_model)


def extract_text_from_pdf(file):
    """Extract text from a PDF file safely."""
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        content = page.extract_text()
        if content:
            text += content
    return text


def add_to_chroma(text, source):
    """Add semantic chunks to ChromaDB."""
    st.info(" Creating semantic chunks for " + source + "...")
    try:
        chunks = semantic_splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            doc_id = source.replace(" ", "_") + "_" + str(i)
            collection.add(
                documents=[chunk],
                metadatas=[{"source": source}],
                ids=[doc_id]
            )

        st.success(" Indexed " + str(len(chunks)) + " chunks from " + source)

    except Exception as e:
        st.error("Error adding to ChromaDB: " + str(e))


# Process uploaded PDFs
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

query = st.text_area("Type your question about BookMyShow:", height=120)

if st.button("Get AI Answer"):
    if api_key.strip() == "":
        st.error(" Please enter your Groq API Key first!")
    elif query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner(" Processing your question..."):

            try:
                # Retrieve contextual documents
                results = collection.query(
                    query_texts=[query],
                    n_results=3
                )

                docs = results.get("documents", [[]])[0]
                context = "\n\n".join(docs) if docs else "No relevant context found."

                # LLM Query to Groq
                response = groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an AI assistant specialized in BookMyShow support. "
                                "Use only the provided context to answer questions."
                            )
                        },
                        {
                            "role": "user",
                            "content": "Context:\n" + context + "\n\nQuestion: " + query
                        }
                    ]
                )

                # Python 3.8 safe dictionary access
                answer = response.choices[0].message["content"]

                st.success(" AI Answer:")
                st.markdown("### " + answer)

                with st.expander(" Retrieved Context"):
                    if docs:
                        for i, doc in enumerate(docs):
                            st.markdown("**" + str(i + 1) + ".** " + doc)
                    else:
                        st.write("No context available.")

            except Exception as e:
                st.error(" Error: " + str(e))

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption(" Powered by Groq + Streamlit + ChromaDB + Semantic Chunking (Python 3.8 Compatible)")
