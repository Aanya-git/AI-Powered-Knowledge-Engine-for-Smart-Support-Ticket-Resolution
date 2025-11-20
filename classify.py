# classify.py
from llm_utils import groq_chat_completion
from config import GROQ_CLASSIFY_MODEL
from typing import List

# Predefined categories for BookMyShow support
INTENT_CATEGORIES: List[str] = [
    "Payment Issue",
    "Ticket Booking Issue",
    "Refund Request",
    "Event Cancellation Query",
    "Account or Login Issue",
    "General Query / Other"
]

def normalize_category(cat: str) -> str:
    """
    Normalize small variations the model may output.
    """
    if not cat:
        return "General Query / Other"

    cat = cat.strip().replace('"', '')

    # Convert plural to singular
    if cat.lower() in ["general query / others", "general query/others"]:
        return "General Query / Other"

    return cat


def classify_ticket(text: str) -> str:
    """
    Classify the ticket text into one category from INTENT_CATEGORIES.
    Returns the category name (string).
    """
    if not text or not text.strip():
        return "General Query / Other"

    system_prompt = (
        "You are a BookMyShow support ticket intent classifier. "
        "Classify the user's message into ONE of the categories below and RETURN ONLY the category name.\n\n"
        f"{INTENT_CATEGORIES}\n\n"
        "If none match exactly, return 'General Query / Other'. Do not add extra text or explanation."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

    try:
        resp = groq_chat_completion(
            messages,
            model=GROQ_CLASSIFY_MODEL,   # FIXED: using config model
            max_tokens=32,
            temperature=0.0
        )

        category = normalize_category(resp)

        if category not in INTENT_CATEGORIES:
            return "General Query / Other"

        return category

    except Exception:
        return "General Query / Other"


if __name__ == "__main__":
    print("Classifier (Groq)")
    while True:
        t = input("Ticket text ('exit' to quit): ")
        if t.lower() == "exit":
            break
        print("Predicted ->", classify_ticket(t))
