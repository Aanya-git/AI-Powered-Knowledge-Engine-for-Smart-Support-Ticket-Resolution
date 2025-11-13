# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Groq (xAI) API key for Groq endpoints
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Paths and embedding model
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
DATA_PATH = os.getenv("DATA_PATH", "data")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FEEDBACK_DB = os.getenv("FEEDBACK_DB", "feedback.db")

# Groq model names (centralized)
GROQ_CLASSIFY_MODEL = os.getenv("GROQ_CLASSIFY_MODEL", "llama-3.3-8b-instant")
GROQ_RAG_MODEL = os.getenv("GROQ_RAG_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.3-8b-instant")
