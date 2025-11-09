from classify import classify_ticket  #classifier
from ingest import Chroma, HuggingFaceEmbeddings  #DB + embeddings
from dotenv import load_dotenv
import os

load_dotenv()

#Load Vector DB
CHROMA_PATH = "chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

def retrieve_docs(query):
    """ RAG retriever """
    results = db.similarity_search(query, k=4)
    return "\n\n".join([doc.page_content for doc in results])

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(user_query, context):
    """ Generate RAG answer using context """
    prompt = f"""
You are a BookMyShow Support AI.

User Query: {user_query}

Relevant Knowledge:
{context}

If the context contains answer, use it.
If not, say "Let me connect you to support".

Provide helpful & short reply.
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"system","content":prompt}]
    )
    return response.choices[0].message.content.strip()

def ai_support_agent(user_input):
    """ Classifier + RAG pipeline connection """

    # Step 1 — Classify ticket intent
    category = classify_ticket(user_input)
    print(f"\n Detected Category: {category}")

    # Step 2 — Decide if RAG should answer
    rag_categories = [
        "Payment Issue", 
        "Refund Request", 
        "Ticket Booking Issue"
    ]

    if category in rag_categories:
        # Retrieve relevant KB docs
        context = retrieve_docs(user_input)

        # Generate RAG response
        answer = generate_answer(user_input, context)
        return f"{answer}"
    
    else:  
        return f"This looks like a '{category}' — forwarding to human support."

# CLI chat loop
if __name__ == "__main__":
    print("\nBookMyShow AI Support Assistant")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Thank you! Goodbye.")
            break

        reply = ai_support_agent(user_input)
        print(reply)
