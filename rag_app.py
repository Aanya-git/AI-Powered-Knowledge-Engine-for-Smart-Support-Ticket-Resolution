# rag_app.py
import os
import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_PATH, EMBED_MODEL, GROQ_RAG_MODEL, GROQ_FALLBACK_MODEL
from llm_utils import groq_chat_completion

# Embedding model for query embeddings
embedder = SentenceTransformer(EMBED_MODEL)

# Initialize Chroma client (support different chromadb versions)
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
except Exception:
    from chromadb.config import Settings
    chroma_client = chromadb.Client(Settings(persist_directory=CHROMA_PATH))

collection = chroma_client.get_or_create_collection(name="support_docs")

def retrieve_docs(query: str, k: int = 5, alpha: float = 0.2):
    """
    Feedback-aware retrieval.
    1) Fetch top-N by embedding similarity (n_results=10)
    2) Convert distances -> similarity (1 - distance)
    3) final_score = similarity + alpha * feedback_score
    4) return top-k documents (strings)
    """
    query_emb = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_emb,
        n_results=10,
        include=["documents", "metadatas", "distances"]
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    re_ranked = []
    for doc, meta, dist in zip(docs, metas, dists):
        # convert distance to similarity safely
        try:
            similarity = 1.0 - float(dist)
        except Exception:
            similarity = 0.0
        feedback_score = meta.get("feedback_score", 0) if isinstance(meta, dict) else 0
        final_score = similarity + alpha * feedback_score
        re_ranked.append({
            "doc": doc,
            "meta": meta,
            "similarity": similarity,
            "feedback_score": feedback_score,
            "final_score": final_score
        })

    # sort by final_score desc and select top-k
    re_ranked = sorted(re_ranked, key=lambda x: x["final_score"], reverse=True)
    # debug printing (useful during development)
    print("\n--- Retrieval re-ranking debug (top results) ---")
    for i, item in enumerate(re_ranked[:k]):
        src = item["meta"].get("source") if isinstance(item["meta"], dict) else "unknown"
        print(f"{i+1}) final={item['final_score']:.4f} sim={item['similarity']:.4f} fb={item['feedback_score']} src={src}")

    top_docs = [item["doc"] for item in re_ranked[:k]]
    return top_docs

def generate_answer_with_context(user_query: str, context_docs: list):
    """
    Use Groq to generate an answer using retrieved context. If Groq RAG model fails,
    fallback to smaller instant model.
    """
    context_text = "\n\n---\n\n".join(context_docs) if context_docs else ""
    system_content = (
        "You are a concise and helpful BookMyShow customer support assistant. "
        "Use the provided context ONLY to answer. If the context doesn't contain an answer, reply exactly: 'Let me connect you to support.' "
        "Keep the answer short and actionable (max 200 words). If you mention a policy or step, cite the source file name if available."
    )

    user_prompt = f"User Query: {user_query}\n\nRelevant Knowledge:\n{context_text}\n\nAnswer:"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt}
    ]

    # Try primary RAG model first, fallback to smaller instant model on failure
    try:
        answer = groq_chat_completion(messages, model=GROQ_RAG_MODEL, max_tokens=512, temperature=0.1)
        return answer
    except Exception as e:
        print(f"[WARN] Primary RAG model failed: {e}. Trying fallback...")
        try:
            answer = groq_chat_completion(messages, model=GROQ_FALLBACK_MODEL, max_tokens=400, temperature=0.1)
            return answer + "\n\n(Note: Answer generated using fallback model.)"
        except Exception as e2:
            print(f"[ERROR] Fallback model also failed: {e2}")
            return "Let me connect you to support."

def ai_support_agent(user_input: str) -> str:
    """
    Full RAG pipeline: retrieve context and generate grounded answer.
    """
    docs = retrieve_docs(user_input, k=5, alpha=0.2)
    answer = generate_answer_with_context(user_input, docs)
    return answer

# CLI quick test
if __name__ == "__main__":
    print("RAG Agent (feedback-aware)")
    while True:
        q = input("\nAsk (type 'exit'): ").strip()
        if q.lower() == "exit":
            break
        print("\nResponse:\n", ai_support_agent(q))
