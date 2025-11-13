# app.py
import streamlit as st
from rag_app import ai_support_agent
from classify import classify_ticket
import sqlite3
from config import FEEDBACK_DB

st.set_page_config(page_title="BookMyShow AI Support", layout="centered")

def init_db():
    conn = sqlite3.connect(FEEDBACK_DB, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT,
        response TEXT,
        feedback TEXT,
        comment TEXT,
        ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    return conn

conn = init_db()

st.title("BookMyShow AI Support Assistant")
st.write("Ask anything — powered by your RAG knowledge base.")

with st.form("ask_form"):
    user_input = st.text_area("Enter your support query", height=120)
    submitted = st.form_submit_button("Submit")

if submitted:
    if not user_input.strip():
        st.warning("Enter a query.")
    else:
        with st.spinner("Classifying and fetching response..."):
            # Step 1: Classify ticket
            try:
                category = classify_ticket(user_input)
            except:
                category = "General Query / Other"

            # Step 2: Categories that will be answered using RAG
            rag_allowed = {
                "Payment Issue",
                "Refund Request",
                "Ticket Booking Issue",
                "General Query / Other",
                "Account or Login Issue",
                "Event Cancellation Query"
            }

            # Step 3: If classifier matches → Use RAG. If classifier fails, STILL use RAG.
            if category in rag_allowed:
                try:
                    reply = ai_support_agent(user_input)
                except Exception as e:
                    reply = "Something went wrong. Let me connect you to support."
            else:
                # Worst-case fallback — should rarely happen
                reply = "Let me connect you to support."

        # UI Display
        st.markdown("### Category")
        st.write(category)

        st.markdown("### Response")
        st.write(reply)

        # Feedback UI
        st.markdown("### Feedback")
        col1, col2 = st.columns(2)
        with col1:
            thumbs_up = st.button("Helpful")
        with col2:
            thumbs_down = st.button("Not Helpful")

        # Save Feedback
        if thumbs_up or thumbs_down:
            feedback_value = "Yes" if thumbs_up else "No"
            comment = ""
            if thumbs_down:
                comment = st.text_area("What went wrong? (optional)")
            c = conn.cursor()
            c.execute(
                "INSERT INTO feedback (query, response, feedback, comment) VALUES (?, ?, ?, ?)",
                (user_input, reply, feedback_value, comment),
            )
            conn.commit()
            st.success("Feedback recorded — thank you!")

st.markdown("---")
st.markdown("### Admin Tools")
if st.button("Show last 5 feedback entries"):
    c = conn.cursor()
    rows = c.execute("SELECT query, response, feedback, comment, ts FROM feedback ORDER BY id DESC LIMIT 5").fetchall()
    for r in rows:
        st.markdown(f"[{r[4]}] Feedback: {r[2]}  \nQuery: {r[0]}  \nResponse: {r[1]}  \nComment: {r[3] or '-'}")
