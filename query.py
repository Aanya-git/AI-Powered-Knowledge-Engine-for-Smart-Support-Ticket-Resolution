# query.py
from config import CHROMA_PATH, EMBED_MODEL, GROQ_RAG_MODEL
import chromadb
from sentence_transformers import SentenceTransformer
from llm_utils import groq_chat_completion

embedder = SentenceTransformer(EMBED_MODEL)

# init chroma client (same approach as rag_app)
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
except Exception:
    from chromadb.config import Settings
    chroma_client = chromadb.Client(Settings(persist_directory=CHROMA_PATH))

collection = chroma_client.get_or_create_collection(name="support_docs")

def ask_query(query: str, k: int = 5):
    q_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=k, include=["documents", "metadatas"])
    docs = []
    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]

    if not docs:
        return "No relevant information found in the knowledge base."

    context = "\n\n".join([d for d in docs if isinstance(d, str)])
    prompt = f"""You are a BookMyShow support assistant.
Use the following knowledge base context to answer the user’s question.
Context:
{context}

Question: {query}

Answer:"""

    messages = [
        {"role": "system", "content": "You are a helpful customer support agent."},
        {"role": "user", "content": prompt}
    ]

    return groq_chat_completion(messages, model="llama-3.3-70b-versatile", temperature=0.2, max_tokens=512)

if __name__ == "__main__":
    while True:
        q = input("\nAsk your question (or 'exit'): ").strip()
        if q.lower() == "exit":
            break
        print("\nAnswer:\n", ask_query(q))
