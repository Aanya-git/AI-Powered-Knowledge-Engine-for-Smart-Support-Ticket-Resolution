# streamlit_frontend_singlefile.py
import os
import sqlite3
import uuid
import pathlib
import time
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

# -------------------------
# Load env and config
# -------------------------
load_dotenv()
APP_USER = os.getenv("APP_USER", "user")
APP_PASS = os.getenv("APP_PASS", "user123")
# Use support constants (replace earlier admin)
SUPPORT_USER = os.getenv("SUPPORT_USER", os.getenv("ADMIN_USER", "support"))
SUPPORT_PASS = os.getenv("SUPPORT_PASS", os.getenv("ADMIN_PASS", "support123"))
KB_DIR = os.getenv("KB_DIR", "kb_docs")
DB_PATH = os.getenv("TICKETS_DB", "tickets.db")
SHOW_DEBUG = os.getenv("SHOW_DEBUG", "false").lower() in ("1", "true", "yes")

pathlib.Path(KB_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path("attachments").mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Smart Support — Chat + Tickets", layout="wide")

# -------------------------
# Try to import backend RAG functions (best-effort)
# -------------------------
ai_support_agent = None
add_to_chroma = None
retrieve_docs = None
generate_answer = None
connected_backend = None

POSSIBLE_BACKEND_MODULES = ["rag_app", "rag", "groq_rag", "rag_service", "ai_rag", "ingest", "query", "llm_utils"]
for mod_name in POSSIBLE_BACKEND_MODULES:
    try:
        module = __import__(mod_name)
        if hasattr(module, "ai_support_agent"):
            ai_support_agent = getattr(module, "ai_support_agent")
        if hasattr(module, "add_to_chroma"):
            add_to_chroma = getattr(module, "add_to_chroma")
        if hasattr(module, "retrieve_docs"):
            retrieve_docs = getattr(module, "retrieve_docs")
        if hasattr(module, "generate_answer"):
            generate_answer = getattr(module, "generate_answer")
        connected_backend = mod_name
        break
    except Exception:
        pass

# -------------------------
# Database helpers (tickets + attachments preserved)
# -------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn

def init_db(conn):
    c = conn.cursor()
    # keep your tickets + attachments table (unchanged)
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS tickets (
            id TEXT PRIMARY KEY,
            title TEXT,
            description TEXT,
            created_at TEXT,
            status TEXT,
            priority TEXT,
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
    # ---------- Add messages table for chat (non-destructive)
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            ticket_id TEXT,
            sender TEXT,
            content TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()

conn = get_conn()

# -------------------------
# Messages helpers
# -------------------------
def add_message(ticket_id, sender, content):
    mid = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (id, ticket_id, sender, content, created_at) VALUES (?,?,?,?,?)",
        (mid, ticket_id, sender, content, created_at),
    )
    conn.commit()
    return mid

def get_messages(ticket_id, limit=200):
    c = conn.cursor()
    c.execute("SELECT sender, content, created_at FROM messages WHERE ticket_id=? ORDER BY created_at ASC LIMIT ?", (ticket_id, limit))
    rows = c.fetchall()
    return [dict(r) for r in rows]

# -------------------------
# Ticket helpers (kept)
# -------------------------
def create_ticket(title, description, creator):
    ticket_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    c = conn.cursor()
    c.execute(
        "INSERT INTO tickets (id,title,description,created_at,status,priority,creator,assigned_to,ai_response,needs_human,resolver_response,feedback,saved_to_kb) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (ticket_id, title, description, created_at, "open", None, creator, None, None, 1, None, None, 0),
    )
    conn.commit()
    # also create an initial system message
    add_message(ticket_id, "system", f"Ticket created: {title}")
    return ticket_id

def list_tickets(filter_by=None):
    c = conn.cursor()
    if filter_by == "open":
        q = "SELECT * FROM tickets WHERE status='open' AND needs_human=1 ORDER BY created_at DESC"
        c.execute(q)
    elif filter_by == "unresolved":
        q = "SELECT * FROM tickets WHERE status!='resolved' ORDER BY created_at DESC"
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
            add_to_chroma(str(path.read_text()), fname)
        except Exception:
            try:
                add_to_chroma(path.read_text(), fname)
            except Exception:
                pass
    return str(path)

# -------------------------
# AI helper (pluggable)
# -------------------------
def call_ai_agent(user_query):
    if ai_support_agent:
        try:
            answer = ai_support_agent(user_query)
            if not answer:
                return ("", False, True)
            lowered = answer.strip().lower()
            if any(x in lowered for x in ["let me connect", "connect you to support", "forwarding to human", "cannot"]):
                return (answer, False, True)
            return (answer, True, False)
        except Exception as e:
            return (f"AI call failed: {e}", False, True)
    else:
        return ("AI backend not available. Please configure rag_app or add ai backend.", False, True)

# -------------------------
# Authentication state and helpers (use support role)
# -------------------------
if "auth" not in st.session_state:
    st.session_state.auth = {"logged_in": False, "role": None, "user": None}

def login(user_input, pass_input, role):
    role = role.lower()
    if role == "user":
        if user_input == APP_USER and pass_input == APP_PASS:
            st.session_state.auth = {"logged_in": True, "role": "user", "user": user_input}
            return True
        return False
    elif role in ("support", "agent", "admin"):
        # support role fallback
        if user_input == SUPPORT_USER and pass_input == SUPPORT_PASS:
            st.session_state.auth = {"logged_in": True, "role": "support", "user": user_input}
            return True
        return False
    return False

def logout():
    st.session_state.auth = {"logged_in": False, "role": None, "user": None}
    st.rerun()

# -------------------------
# UI helpers (whatsapp-like)
# -------------------------
def render_messages(messages, me_label="you", agent_label="support"):
    # simple "chat bubble" rendering using columns
    for m in messages:
        sender = m["sender"]
        content = m["content"]
        ts = m["created_at"][:19].replace("T", " ")
        if sender in ("system", "ai", "agent", "support"):
            # left aligned (agent/system)
            cols = st.columns([1, 4])
            with cols[0]:
                st.markdown(f"**{sender}**")
            with cols[1]:
                st.markdown(f"> {content}  \n*{ts}*")
        else:
            # right aligned (user)
            cols = st.columns([4, 1])
            with cols[0]:
                st.markdown("")
            with cols[1]:
                st.markdown(f"**{sender}**  \n> {content}  \n*{ts}*")

# -------------------------
# Pages
# -------------------------
def login_page():
    st.title("Smart Support — Login")
    st.write("Login as a Customer (user) or Support Agent (support).")
    with st.form("login_form"):
        col1, col2 = st.columns([1,1])
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
                strerun()
            else:
                st.error("Invalid credentials for role: " + role)
    if SHOW_DEBUG:
        st.sidebar.markdown("Debug")
        st.sidebar.write("APP_USER / APP_PASS:", APP_USER, APP_PASS)
        st.sidebar.write("SUPPORT_USER / SUPPORT_PASS:", SUPPORT_USER, SUPPORT_PASS)
        st.sidebar.write("Connected backend:", connected_backend if connected_backend else "None")

def user_dashboard():
    st.sidebar.markdown("### Actions")
    if st.sidebar.button("Logout"):
        logout()

    if connected_backend:
        st.sidebar.success(f"Backend: {connected_backend}")
    else:
        st.sidebar.info("Backend not connected — AI calls unavailable.")

    st.header("Customer Dashboard")
    tab = st.tabs(["Raise Query", "Chat with Support"])

    # ---------------- Raise Query tab ----------------
    with tab[0]:
        st.subheader("Submit a new query")
        with st.form("raise_query_form", clear_on_submit=True):
            title = st.text_input("Issue title")
            description = st.text_area("Describe the issue")
            attachments = st.file_uploader("Add attachments (optional)", accept_multiple_files=True)
            submit = st.form_submit_button("Submit query")
            if submit:
                if not title.strip() or not description.strip():
                    st.warning("Please provide a title and description.")
                else:
                    ticket_id = create_ticket(title.strip(), description.strip(), st.session_state.auth["user"])
                    # attachments handling (keeps same semantics)
                    if attachments:
                        for f in attachments:
                            filename = f.name
                            # save file
                            save_dir = pathlib.Path("attachments")
                            save_dir.mkdir(parents=True, exist_ok=True)
                            filepath = save_dir / f"{str(uuid.uuid4())}_{filename}"
                            with open(filepath, "wb") as fh:
                                fh.write(f.getbuffer())
                            c = conn.cursor()
                            c.execute("INSERT INTO attachments (id, ticket_id, filename, filepath) VALUES (?,?,?,?)",
                                      (str(uuid.uuid4()), ticket_id, filename, str(filepath)))
                            conn.commit()
                    st.success(f"Query created: {ticket_id}")

    # ---------------- Chat tab ----------------
    with tab[1]:
        st.subheader("Chat with Support")
        # show list of user's tickets
        c = conn.cursor()
        c.execute("SELECT id, title, status, created_at FROM tickets WHERE creator=? ORDER BY created_at DESC", (st.session_state.auth["user"],))
        tickets = c.fetchall()
        ticket_options = [{"id": r["id"], "label": f"{r['title']} — [{r['status']}] — {r['created_at']}"} for r in tickets]
        if ticket_options:
            opt_map = {t["label"]: t["id"] for t in ticket_options}
            sel_label = st.selectbox("Select ticket", [t["label"] for t in ticket_options])
            selected_ticket_id = opt_map[sel_label]
            st.markdown("----")
            st.markdown("**Chat history**")
            msgs = get_messages(selected_ticket_id)
            render_messages(msgs, me_label=st.session_state.auth["user"], agent_label="support")
            st.markdown("----")
            # send message form
            with st.form("send_message_form", clear_on_submit=True):
                msg = st.text_area("Write message to support", key=f"user_msg_{selected_ticket_id}")
                send = st.form_submit_button("Send to Support")
                if send:
                    if not msg.strip():
                        st.warning("Please enter a message.")
                    else:
                        add_message(selected_ticket_id, st.session_state.auth["user"], msg.strip())
                        st.success("Message sent to support.")
                        # Also mark ticket as needs human (if AI previously tried)
                        update_ticket(selected_ticket_id, needs_human=1, status="open")
                        st.rerun()
        else:
            st.info("You have no queries yet. Create a query in the 'Raise Query' tab.")

def support_dashboard():
    st.sidebar.markdown("### Actions")
    if st.sidebar.button("Logout"):
        logout()

    st.title("Support Agent Console")
    st.markdown("Top: select ticket on the right to load chat and use AI suggestions to assist.")

    # left column: ticket list
    col_left, col_right = st.columns([2,5])
    with col_left:
        st.subheader("Open Tickets")
        tickets = list_tickets(filter_by="open")
        if not tickets:
            st.info("No open tickets.")
            selected_ticket_id = None
        else:
            ticket_map = {f"{t['title']} — {t['created_at']} ({t['id'][:6]})": t["id"] for t in tickets}
            sel_key = st.selectbox("Select ticket to work on", list(ticket_map.keys()))
            selected_ticket_id = ticket_map[sel_key]

            # quick actions
            if st.button("Get AI Suggestion for selected ticket"):
                # use latest user message or ticket description
                msgs = get_messages(selected_ticket_id)
                last_user_msgs = [m for m in msgs if m["sender"] == st.session_state.auth["user"] or m["sender"] == "user"]
                prompt_text = None
                if last_user_msgs:
                    prompt_text = last_user_msgs[-1]["content"]
                else:
                    tkt = get_ticket(selected_ticket_id)
                    prompt_text = tkt["description"] if tkt else ""

                with st.spinner("Calling AI (if configured)..."):
                    answer, resolved, needs_human = call_ai_agent(prompt_text or "")
                    # insert AI reply as message from 'ai' or 'agent' but mark as suggestion
                    add_message(selected_ticket_id, "ai", answer)
                    st.success("AI suggestion saved to messages.")
                    st.rerun()

    with col_right:
        st.subheader("Conversation / Reply")
        if not selected_ticket_id:
            st.info("Select a ticket from the left to view messages.")
            return
        ticket = get_ticket(selected_ticket_id)
        if not ticket:
            st.error("Ticket not found.")
            return

        st.markdown(f"**Ticket:** {ticket['title']}  —  `{ticket['id']}`")
        st.write("Description:")
        st.write(ticket["description"])
        st.markdown("---")
        messages = get_messages(selected_ticket_id)
        render_messages(messages, me_label=st.session_state.auth["user"], agent_label="support")

        st.markdown("**Reply to user**")
        reply = st.text_area("Write reply to user (agent)", key=f"agent_reply_{selected_ticket_id}")
        save_to_kb = st.checkbox("Save this Q&A to KB", key=f"savekb_{selected_ticket_id}")
        if st.button("Send reply & mark resolved"):
            if not reply.strip():
                st.warning("Reply text required.")
            else:
                add_message(selected_ticket_id, "support", reply.strip())
                update_ticket(selected_ticket_id, resolver_response=reply.strip(), assigned_to=st.session_state.auth["user"], status="resolved", needs_human=0)
                st.success("Reply sent and ticket marked resolved.")
                notify_user_placeholder(selected_ticket_id, "Your ticket has been resolved by support.")
                if save_to_kb:
                    save_qna_to_kb(ticket["title"], ticket["description"], reply.strip())
                st.rerun()

# small placeholder for notification (local only)
def notify_user_placeholder(ticket_id, message):
    # non-invasive: store a system message so user sees it in chat
    add_message(ticket_id, "system", message)

# -------------------------
# Entrypoint
# -------------------------
if not st.session_state.auth["logged_in"]:
    login_page()
else:
    role = st.session_state.auth["role"]
    st.sidebar.markdown(f"Logged in as **{st.session_state.auth['user']}** — role: **{role}**")
    if role == "user":
        user_dashboard()
    elif role == "support":
        support_dashboard()
    else:
        st.error("Unknown role. Please logout and login again.")
