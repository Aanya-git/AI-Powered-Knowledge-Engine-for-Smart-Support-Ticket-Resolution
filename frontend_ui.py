import os
import sqlite3
import pathlib
import uuid
import time
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

# -------------------------
# Config & env
# -------------------------
load_dotenv()
APP_USER = os.getenv("APP_USER", "user")
APP_PASS = os.getenv("APP_PASS", "user123")
SUPPORT_USER = os.getenv("SUPPORT_USER", "support")  # was ADMIN_USER renamed
SUPPORT_PASS = os.getenv("SUPPORT_PASS", "support123")
KB_DIR = os.getenv("KB_DIR", "kb_docs")
DB_PATH = os.getenv("TICKETS_DB", "tickets.db")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # used by your backend

pathlib.Path(KB_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path("attachments").mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="BookMyShow — Smart Support (WhatsApp UI)", layout="wide")

# -------------------------
# Try to import backend RAG functions (best-effort)
# -------------------------
ai_support_agent = None
generate_answer = None
retrieve_docs = None
add_to_chroma = None
connected_backend = None

POSSIBLE_BACKEND_MODULES = ["rag_app", "query", "config", "classify", "llm_utils"]

_import_errors = {}
for mod_name in POSSIBLE_BACKEND_MODULES:
    try:
        module = __import__(mod_name)
        # prefer names commonly used
        if hasattr(module, "ai_support_agent"):
            ai_support_agent = getattr(module, "ai_support_agent")
        if hasattr(module, "generate_answer"):
            generate_answer = getattr(module, "generate_answer")
        if hasattr(module, "retrieve_docs"):
            retrieve_docs = getattr(module, "retrieve_docs")
        if hasattr(module, "add_to_chroma"):
            add_to_chroma = getattr(module, "add_to_chroma")
        connected_backend = mod_name
        break
    except Exception as e:
        _import_errors[mod_name] = str(e)

# -------------------------
# DB helpers (unchanged schema)
# -------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn

def init_db(conn):
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS tickets (
            id TEXT PRIMARY KEY,
            title TEXT,
            description TEXT,
            created_at TEXT,
            status TEXT,
            creator TEXT,
            assigned_to TEXT,
            ai_response TEXT,
            needs_human INTEGER,
            resolver_response TEXT,
            feedback INTEGER,
            saved_to_kb INTEGER
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS attachments (
            id TEXT PRIMARY KEY,
            ticket_id TEXT,
            filename TEXT,
            filepath TEXT
        )
        """
    )
    conn.commit()

conn = get_conn()

# -------------------------
# Session state defaults
# -------------------------
if "auth" not in st.session_state:
    st.session_state.auth = {"logged_in": False, "role": None, "user": None}

# Keep chat caches in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}  # ticket_id -> list of {"sender","text","time"}

# last opened ticket for agent
if "open_ticket" not in st.session_state:
    st.session_state.open_ticket = None

# -------------------------
# Auth helpers
# -------------------------
def login(user_input, pass_input, role):
    role = role.strip().lower()
    if role == "user":
        if user_input == APP_USER and pass_input == APP_PASS:
            st.session_state.auth = {"logged_in": True, "role": "user", "user": user_input}
            return True
        return False
    elif role == "support":
        if user_input == SUPPORT_USER and pass_input == SUPPORT_PASS:
            st.session_state.auth = {"logged_in": True, "role": "support", "user": user_input}
            return True
        return False
    return False

def logout():
    st.session_state.auth = {"logged_in": False, "role": None, "user": None}
    st.rerun()

# -------------------------
# Ticket helpers
# -------------------------
def create_ticket(title, description, creator):
    ticket_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    c = conn.cursor()
    c.execute(
        "INSERT INTO tickets (id,title,description,created_at,status,creator,assigned_to,ai_response,needs_human,resolver_response,feedback,saved_to_kb) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (ticket_id, title, description, created_at, "open", creator, None, None, 1, None, None, 0),
    )
    conn.commit()
    return ticket_id

def add_attachment(ticket_id, uploaded_file):
    attach_id = str(uuid.uuid4())
    filename = uploaded_file.name
    save_dir = pathlib.Path("attachments")
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / f"{attach_id}_{filename}"
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    c = conn.cursor()
    c.execute(
        "INSERT INTO attachments (id, ticket_id, filename, filepath) VALUES (?,?,?,?)",
        (attach_id, ticket_id, filename, str(filepath)),
    )
    conn.commit()
    return str(filepath)

def list_tickets(filter_by=None):
    c = conn.cursor()
    if filter_by == "open":
        q = "SELECT * FROM tickets WHERE status='open' ORDER BY created_at DESC"
        c.execute(q)
    else:
        q = "SELECT * FROM tickets ORDER BY created_at DESC"
        c.execute(q)
    rows = c.fetchall()
    return [dict(r) for r in rows]

def get_ticket(ticket_id):
    c = conn.cursor()
    c.execute("SELECT * FROM tickets WHERE id=?", (ticket_id,))
    row = c.fetchone()
    return dict(row) if row else None

def update_ticket(ticket_id, **kwargs):
    c = conn.cursor()
    sets = []
    params = []
    for k, v in kwargs.items():
        sets.append(f"{k}=?")
        params.append(v)
    params.append(ticket_id)
    sql = f"UPDATE tickets SET {', '.join(sets)} WHERE id=?"
    c.execute(sql, tuple(params))
    conn.commit()

def save_qna_to_kb(title, question, answer):
    fname = f"{int(time.time())}_{title.replace(' ', '_')}.txt"
    path = pathlib.Path(KB_DIR) / fname
    path.write_text(f"Q: {question}\n\nA: {answer}\n")
    if add_to_chroma:
        try:
            # common signature
            add_to_chroma(path.read_text(), fname)
        except Exception:
            try:
                add_to_chroma(str(path.read_text()), fname)
            except Exception:
                pass
    return str(path)

# -------------------------
# AI helper (agent-side calls only)
# -------------------------
def get_ai_suggestion(agent_query, context_docs=None):
    """
    Agent asks for suggestion. The frontend will only call this function (not user).
    We prefer generate_answer(module) if present, else ai_support_agent wrapper.
    """
    try:
        if generate_answer:
            # generate_answer(user_query, context) signature expected by some backends
            ctx = context_docs or ""
            return generate_answer(agent_query, ctx)
        elif ai_support_agent:
            # fallback to ai_support_agent(query) which may do retrieval+answer
            return ai_support_agent(agent_query)
        else:
            return "AI backend not available."
    except Exception as e:
        return f"AI call failed: {e}"

# -------------------------
# UI Helpers - chat rendering (WhatsApp like)
# -------------------------
def render_chat(ticket_id, compact=False):
    messages = st.session_state.chat_history.get(ticket_id, [])
    # render messages in order
    for m in messages:
        sender = m.get("sender")
        text = m.get("text")
        timestamp = m.get("time")
        # Use streamlit's built-in chat components when available
        try:
            if sender == "user":
                st.chat_message("user", avatar="🧑").write(text)
            elif sender == "support":
                st.chat_message("assistant", avatar="🛠️").write(text)
            else:
                # system or agent
                st.markdown(f"**{sender}:** {text}")
        except Exception:
            # fallback simple
            if sender == "user":
                st.markdown(f"<div style='text-align:left; padding:8px; background:#e6f7ff; border-radius:8px;'>{text}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align:right; padding:8px; background:#f1f0f0; border-radius:8px'>{text}</div>", unsafe_allow_html=True)

def append_chat(ticket_id, sender, text):
    m = {"sender": sender, "text": text, "time": datetime.utcnow().isoformat()}
    st.session_state.chat_history.setdefault(ticket_id, []).append(m)

# -------------------------
# Pages
# -------------------------
def login_page():
    st.title("BookMyShow — Support Portal")
    st.write("Login as Customer (User) or Support Agent.")
    with st.form("login_form"):
        col1, col2 = st.columns([2, 1])
        with col1:
            username = st.text_input("Username")
        with col2:
            password = st.text_input("Password", type="password")
        role = st.radio("Login as", ("user", "support"))
        submitted = st.form_submit_button("Login")
        if submitted:
            ok = login(username.strip(), password.strip(), role)
            if ok:
                st.success("Logged in.")
                st.rerun()
            else:
                st.error("Invalid credentials for role: " + role)

    if not connected_backend:
        with st.expander("Backend status (debug)"):
            st.write("AI Backend connected:", connected_backend)
            if _import_errors:
                st.write("Import errors for attempted backend modules:")
                for m, err in _import_errors.items():
                    st.write(f"- {m}: {err}")

def user_dashboard():
    st.sidebar.markdown("### Actions")
    if st.sidebar.button("Logout"):
        logout()
    st.header("Create a query")
    st.info("Your query will be delivered to a Support Agent.")
    with st.form("query_form", clear_on_submit=True):
        title = st.text_input("Short title for your query")
        description = st.text_area("Describe the issue in detail")
        attachments = st.file_uploader("Add attachments (optional)", accept_multiple_files=True, type=None)
        submit = st.form_submit_button("Send Query to Support")

    if submit:
        if not title.strip() or not description.strip():
            st.warning("Please provide a title and description.")
        else:
            ticket_id = create_ticket(title.strip(), description.strip(), st.session_state.auth["user"])
            # save attachments
            if attachments:
                for f in attachments:
                    add_attachment(ticket_id, f)
            # Add initial chat message as user's message
            append_chat(ticket_id, "user", description.strip())
            st.success("Query created and sent to Support. Ticket ID: " + ticket_id)
            st.rerun()

    st.markdown("---")
    st.subheader("Your queries & chat")
    c = conn.cursor()
    c.execute("SELECT id,title,status,created_at,ai_response,needs_human,feedback FROM tickets WHERE creator=? ORDER BY created_at DESC", (st.session_state.auth["user"],))
    rows = c.fetchall()
    if not rows:
        st.info("You have not submitted any queries yet.")
    else:
        for r in rows:
            t_id = r["id"]
            with st.expander(f"{r['title']} — [{r['status']}] — {r['created_at']}"):
                # render chat for this ticket
                render_chat(t_id)
                if r["status"] == "resolved":
                    fb = r["feedback"]
                    if fb is None:
                        col1, col2 = st.columns(2)
                        if col1.button(" Helpful", key=f"helpful_{t_id}"):
                            update_ticket(t_id, feedback=1)
                            st.success("Thanks for your feedback!")
                            st.rerun()
                        if col2.button(" Not helpful", key=f"nothelpful_{t_id}"):
                            update_ticket(t_id, feedback=0)
                            st.info("Thanks — we'll improve.")
                            st.rerun()
                else:
                    st.write("Status:", r["status"])
                    # allow user to add follow-up message
                    follow = st.text_area("Add follow-up message", key=f"follow_{t_id}")
                    if st.button("Send follow-up", key=f"sendfollow_{t_id}"):
                        if follow.strip():
                            # save as new user message and update ticket to open
                            append_chat(t_id, "user", follow.strip())
                            update_ticket(t_id, status="open", needs_human=1)
                            st.success("Follow-up sent to Support.")
                            st.rerun()

def support_dashboard():
    st.sidebar.markdown("### Actions")
    if st.sidebar.button("Logout"):
        logout()

    st.title("Support Agent Console")
    tickets = list_tickets(filter_by="open")
    if not tickets:
        st.info("No open tickets.")
        return

    cols = st.columns([2, 4])
    with cols[0]:
        st.subheader("Open Tickets")
        for t in tickets:
            if st.button(f"{t['title']} — {t['created_at']}", key=f"open_{t['id']}"):
                st.session_state.open_ticket = t["id"]
                # load chat history into session if not present
                if t["id"] not in st.session_state.chat_history:
                    # initialize chat: user initial description
                    append_chat(t["id"], "user", t["description"])
                st.rerun()

    with cols[1]:
        if not st.session_state.open_ticket:
            st.info("Select a ticket to open (left).")
            return
        ticket_id = st.session_state.open_ticket
        ticket = get_ticket(ticket_id)
        st.subheader(f"{ticket['title']} — {ticket['created_at']}")
        # render chat
        render_chat(ticket_id)
        st.markdown("---")
        st.write("AI assistance (agent only):")
        col_a, col_b = st.columns([3,1])
        with col_a:
            # show a context preview (optional)
            context_preview = ""
            if retrieve_docs:
                try:
                    docs = retrieve_docs(ticket["description"])
                    context_preview = "\n\n".join(docs[:3])
                except Exception:
                    context_preview = ""
            st.text_area("Context (auto-retrieved docs preview)", value=context_preview, height=120, key=f"context_{ticket_id}")
        with col_b:
            if st.button("Get AI Suggestion", key=f"ai_suggest_{ticket_id}"):
                # call backend to get suggestion (agent-side only)
                suggestion = get_ai_suggestion(ticket["description"], context_preview)
                # append suggestion as an agent draft (not yet sent)
                st.session_state[f"draft_{ticket_id}"] = suggestion
                st.rerun()

        # show draft (if exists) and allow edit
        draft = st.session_state.get(f"draft_{ticket_id}", "")
        edited = st.text_area("Edit & review AI suggestion (or write your own reply)", value=draft, key=f"reply_area_{ticket_id}", height=150)
        send_col, resolve_col, kb_col = st.columns([2,1,1])
        with send_col:
            if st.button("Send reply to user", key=f"send_{ticket_id}"):
                if edited.strip():
                    append_chat(ticket_id, "support", edited.strip())
                    update_ticket(ticket_id, resolver_response=edited.strip(), assigned_to=st.session_state.auth["user"], status="open", needs_human=0)
                    # optionally: ask user if issue is solved (agent action)
                    append_chat(ticket_id, "support", "Support: Has this resolved your issue? Please reply with 👍 or 👎.")
                    # clear draft
                    st.session_state.pop(f"draft_{ticket_id}", None)
                    st.success("Reply sent to user.")
                    st.rerun()
        with resolve_col:
            if st.button("Resolve ticket", key=f"resolve_{ticket_id}"):
                # mark resolved and send final message
                update_ticket(ticket_id, status="resolved", needs_human=0, assigned_to=st.session_state.auth["user"])
                append_chat(ticket_id, "support", "This ticket has been resolved by Support.")
                st.success("Ticket marked resolved.")
                st.rerun()
        with kb_col:
            if st.button("Save Q&A to KB", key=f"savekb_{ticket_id}"):
                # save current reply or ai suggestion
                reply_text = edited.strip() or st.session_state.get(f"draft_{ticket_id}", "")
                if reply_text:
                    kb_path = save_qna_to_kb(ticket['title'], ticket['description'], reply_text)
                    update_ticket(ticket_id, saved_to_kb=1)
                    st.success(f"Saved Q&A to KB: {kb_path}")
                    # optional: notify backend indexer handled in save_qna_to_kb()
                else:
                    st.warning("No reply text to save.")
    # end support panel

# -------------------------
# App main
# -------------------------
def main():
    if not st.session_state.auth["logged_in"]:
        login_page()
        return

    role = st.session_state.auth["role"]
    st.sidebar.markdown(f"Logged in as **{st.session_state.auth['user']}** — role: **{role}**")
    if role == "user":
        user_dashboard()
    elif role == "support":
        support_dashboard()
    else:
        st.error("Unknown role. Please logout and login again.")

if __name__ == "__main__":
    main()
