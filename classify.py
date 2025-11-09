import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Ticket categories
INTENT_CATEGORIES = [
    "Payment Issue",
    "Ticket Booking Issue",
    "Refund Request",
    "Event Cancellation Query",
    "Account or Login Issue",
    "General Query / Other"
]

def classify_ticket(text):
    prompt = f"""
You are a BookMyShow support ticket intent classifier.
Classify the message into one of these categories:

{INTENT_CATEGORIES}

User Query: "{text}"

Return ONLY the category name.
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    print("BookMyShow Ticket Classification System")
    print("Type 'exit' to stop\n")

    while True:
        text = input("Enter customer ticket text: ")

        if text.lower() == "exit":
            print("\n Exiting classifier...")
            break
        
        category = classify_ticket(text)
        print(f" Predicted Category: {category}\n")
