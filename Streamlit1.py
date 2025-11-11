import os
import chromadb
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load .env
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Paths and setup
CHROMA_PATH = "embeddings/"
embedder = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection("support_docs_semantic")

def query_with_groq(ticket_query):
    # Get embeddings for query
    query_embedding = embedder.encode(ticket_query).tolist()

    # Search similar chunks from ChromaDB
    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    context = "\n\n".join(results["documents"][0])

    prompt = f"""
    You are a helpful AI support assistant.
    Use the following support documents to answer the query.
    Include citations (source and page).

    Context:
    {context}

    Query:
    {ticket_query}

    Answer:
    """

    # Send to Groq LLM
    completion = client.chat.completions.create(
        model="llama3-70b-8192",  # You can also use mixtral-8x7b or gemma-7b
        messages=[{"role": "user", "content": prompt}],
    )

    print(" AI Response:")
    print(completion.choices[0].message.content)

if __name__ == "__main__":
    while True:
        query = input("\nAsk your support question (type 'exit' to stop): ")
        if query.lower() == "exit":
            break
        query_with_groq(query)
