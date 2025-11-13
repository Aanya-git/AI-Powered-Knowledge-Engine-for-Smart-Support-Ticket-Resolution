# feedback_trainer.py
"""
Read feedback entries from the SQLite feedback DB and adjust Chroma metadata
('feedback_score') for documents that are most similar to the user's query.

- Positive feedback ("Yes") => +1 for matched docs
- Negative feedback ("No") => -1 for matched docs

Run this periodically or after collecting a batch of feedback.
"""

import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
from config import FEEDBACK_DB, CHROMA_PATH, EMBED_MODEL
from dotenv import load_dotenv

load_dotenv()

# init embedder
embedder = SentenceTransformer(EMBED_MODEL)

# init chroma client
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
except Exception:
    from chromadb.config import Settings
    chroma_client = chromadb.Client(Settings(persist_directory=CHROMA_PATH))

collection = chroma_client.get_or_create_collection(name="support_docs")

def adjust_vector_ranking():
    conn = sqlite3.connect(FEEDBACK_DB)
    c = conn.cursor()
    rows = c.execute("SELECT id, query, response, feedback FROM feedback").fetchall()
    if not rows:
        print("No feedback rows found.")
        return

    pos = []
    neg = []
    for _id, q, r, fb in rows:
        if not fb:
            continue
        fb_norm = fb.strip().lower()
        if fb_norm in ("yes", "y", "helpful"):
            pos.append(( _id, q ))
        elif fb_norm in ("no", "n", "not helpful"):
            neg.append(( _id, q ))

    print(f"Total feedback rows: {len(rows)} | Positive: {len(pos)} | Negative: {len(neg)}")

    # Helper to process queries
    def process_list(lst, delta):
        for _id, query in lst:
            print(f"\nProcessing feedback id {_id} (delta={delta}) for query: {query}")
            q_emb = embedder.encode([query]).tolist()
            # find top matching docs
            results = collection.query(query_embeddings=q_emb, n_results=5, include=["ids", "metadatas"])
            ids = results.get("ids", [[]])[0] or []
            metas = results.get("metadatas", [[]])[0] or []
            if not ids:
                print(" - No matching docs found for this query.")
                continue
            for doc_id, meta in zip(ids, metas):
                try:
                    current = collection.get(ids=[doc_id])
                    if not current or "metadatas" not in current:
                        current_meta = {}
                    else:
                        current_meta = current["metadatas"][0] or {}
                    score = current_meta.get("feedback_score", 0)
                except Exception as e:
                    print(f" - Failed to fetch current metadata for {doc_id}: {e}")
                    score = 0

                new_score = score + delta
                # Update metadata (we preserve other metadata by copying existing and replacing feedback_score)
                updated_meta = dict(current_meta)
                updated_meta["feedback_score"] = new_score
                try:
                    collection.update(ids=[doc_id], metadatas=[updated_meta])
                    print(f" - Updated doc {doc_id}: {score} -> {new_score}")
                except Exception as e:
                    print(f" - Failed to update metadata for {doc_id}: {e}")

    # Apply positive feedback (+1)
    process_list(pos, delta=1)
    # Apply negative feedback (-1)
    process_list(neg, delta=-1)

    # Persist changes
    try:
        chroma_client.persist()
    except Exception:
        pass

    print("\nFeedback adjustment complete.")

if __name__ == "__main__":
    adjust_vector_ranking()
