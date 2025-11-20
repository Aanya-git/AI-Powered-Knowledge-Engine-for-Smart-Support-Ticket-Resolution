# streamlit_frontend.py
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
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "admin123")
KB_DIR = os.getenv("KB_DIR", "kb_docs")
DB_PATH = os.getenv("TICKETS_DB", "tickets.db")

pathlib.Path(KB_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path("attachments").mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="BookMyShow — Smart Ticket Resolver (Prototype)", layout="wide")

# -------------------------
# Try to import backend RAG functions (best-effort)
# -------------------------
ai_support_agent = None
add_to_chroma = None
retrieve_docs = None
generate_answer = None
connected_backend = None

POSSIBLE_BACKEND_MODULES = ["rag_app", "rag", "groq_rag", "rag_service", "ai_rag"]

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
# Database helpers
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
    conn.commit()

conn = get_conn()

# -------------------------
# Authentication state and helpers
# -------------------------
if "auth" not in st.session_state:
    st.session_state.auth = {"logged_in": False, "role": None, "user": None}

def login(user_input, pass_input, role):
    if role == "user":
        if user_input == APP_USER and pass_input == APP_PASS:
            st.session_state.auth = {"logged_in": True, "role": "user", "user": user_input}
            return True
        return False
    elif role == "admin":
        if user_input == ADMIN_USER and pass_input == ADMIN_PASS:
            st.session_state.auth = {"logged_in": True, "role": "admin", "user": user_input}
            return True
        return False
    return False

def logout():
    st.session_state.auth = {"logged_in": False, "role": None, "user": None}
    st.experimental_rerun()

# -------------------------
# Ticket & attachment helpers
# -------------------------
def create_ticket(title, description, priority, creator):
    ticket_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    c = conn.cursor()
    c.execute(
        "INSERT INTO tickets (id,title,description,created_at,status,priority,creator,assigned_to,ai_response,needs_human,resolver_response,feedback,saved_to_kb) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (ticket_id, title, description, created_at, "open", priority, creator, None, None, 1, None, None, 0),
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

def get_ticket(ticket_id):
    c = conn.cursor()
    c.execute("SELECT * FROM tickets WHERE id=?", (ticket_id,))
    row = c.fetchone()
    return dict(row) if row else None

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
    # best-effort call into backend indexer
    if add_to_chroma:
        try:
            # try common signatures
            add_to_chroma(str(path.read_text()), fname)
        except TypeError:
            try:
                add_to_chroma(path.read_text(), fname)
            except Exception:
                pass
        except Exception:
            pass
    return str(path)

# -------------------------
# AI helper (pluggable)
# -------------------------
def call_ai_agent(user_query):
    """
    Returns (answer_text, resolved_bool, needs_human_bool)
    """
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
        # No backend -> escalate to human/admin
        return ("AI backend not available. Forwarded to admin for human resolution.", False, True)

# -------------------------
# Small placeholder for notifications (extend as needed)
# -------------------------
def notify_user_placeholder(ticket_id, message):
    # You can extend this to send email/Slack. For now it is a no-op.
    st.info(f"(Notification placeholder) Ticket {ticket_id}: {message}")

# -------------------------
# Pages: login, user dashboard, admin dashboard
# -------------------------
def login_page():
    st.title("BookMyShow — Smart Support (Prototype)")
    st.write("Login as a User (raise tickets) or Admin (resolve tickets). Default demo credentials are shown in the header.")
    st.markdown("**Demo credentials:** User:`user/user123`  •  Admin:`admin/admin123`")
    with st.form("login_form"):
        col1, col2 = st.columns([1, 1])
        with col1:
            username = st.text_input("Username")
        with col2:
            password = st.text_input("Password", type="password")
        role = st.radio("Login as", ("user", "admin"))
        submitted = st.form_submit_button("Login")
        if submitted:
            ok = login(username.strip(), password.strip(), role)
            if ok:
                st.success("Logged in.")
                st.experimental_rerun()
            else:
                st.error("Invalid credentials for role: " + role)

def user_dashboard():
    st.sidebar.markdown("### Actions")
    if st.sidebar.button("Logout"):
        logout()

    # show connected backend status
    if connected_backend:
        st.sidebar.success(f"Backend connected: {connected_backend}")
    else:
        st.sidebar.info("Backend not connected — AI calls will be unavailable.")

    st.header("Raise a ticket")
    with st.form("raise_ticket_form", clear_on_submit=True):
        title = st.text_input("Issue title")
        description = st.text_area("Describe the issue in detail")
        priority = st.selectbox("Priority", ("Low", "Medium", "High"))
        attachments = st.file_uploader("Add attachments (optional)", accept_multiple_files=True, type=None)
        submit = st.form_submit_button("Submit Ticket & Try AI Resolution")
        if submit:
            if not title.strip() or not description.strip():
                st.warning("Please provide a title and description.")
            else:
                ticket_id = create_ticket(title.strip(), description.strip(), priority, st.session_state.auth["user"])
                # save attachments
                if attachments:
                    for f in attachments:
                        add_attachment(ticket_id, f)
                st.info("Ticket created: " + ticket_id)

                # try AI
                with st.spinner("Calling AI agent to auto-resolve..."):
                    answer, resolved, needs_human = call_ai_agent(description)
                    update_ticket(ticket_id, ai_response=answer, needs_human=1 if needs_human else 0, status="resolved" if resolved else "open")
                    if resolved:
                        st.success("AI provided an answer (ticket auto-resolved).")
                        st.markdown("**AI Answer:**")
                        st.write(answer)
                        if st.button("Save this Q&A to KB", key=f"savekb_user_{ticket_id}"):
                            kb_path = save_qna_to_kb(title, description, answer)
                            update_ticket(ticket_id, saved_to_kb=1)
                            st.success(f"Saved to KB: {kb_path}")
                    else:
                        st.warning("AI couldn't confidently solve — ticket forwarded to admin.")
                        st.write(answer)

    st.markdown("---")
    st.subheader("Your Tickets")
    c = conn.cursor()
    c.execute("SELECT id,title,status,created_at,ai_response,needs_human,feedback FROM tickets WHERE creator=? ORDER BY created_at DESC", (st.session_state.auth["user"],))
    rows = c.fetchall()
    if not rows:
        st.info("You have not raised any tickets yet.")
    else:
        for r in rows:
            t_id = r["id"]
            t_title = r["title"]
            t_status = r["status"]
            t_created = r["created_at"]
            t_ai_response = r["ai_response"]
            needs_human = r["needs_human"]
            with st.expander(f"{t_title} — [{t_status}] — {t_created}"):
                st.write("Ticket ID:", t_id)
                st.write("Status:", t_status)
                if t_ai_response:
                    st.markdown("**AI Response:**")
                    st.write(t_ai_response)
                if t_status == "resolved":
                    fb = r["feedback"]
                    if fb is None:
                        st.write("Was this answer helpful?")
                        col1, col2 = st.columns(2)
                        if col1.button("👍 Helpful", key=f"helpful_{t_id}"):
                            update_ticket(t_id, feedback=1)
                            st.success("Thanks for your feedback!")
                        if col2.button("👎 Not helpful", key=f"nothelpful_{t_id}"):
                            update_ticket(t_id, feedback=0)
                            st.info("Thanks — this will help improve future answers.")
                # attachments
                c.execute("SELECT filename, filepath FROM attachments WHERE ticket_id=?", (t_id,))
                attachments = c.fetchall()
                if attachments:
                    st.markdown("Attachments:")
                    for a in attachments:
                        st.write(f"- {a['filename']} (saved at `{a['filepath']}`)")

def admin_dashboard():
    st.sidebar.markdown("### Actions")
    if st.sidebar.button("Logout"):
        logout()

    st.title("Admin Console — Ticket Resolver")
    st.markdown("Open tickets (AI couldn't resolve or pending human).")

    tickets = list_tickets(filter_by="open")
    if not tickets:
        st.info("No open tickets awaiting admin resolution.")
    else:
        for t in tickets:
            with st.expander(f"{t['title']} — priority: {t['priority']} — created: {t['created_at']}"):
                st.write("Ticket ID:", t["id"])
                st.write("Creator:", t["creator"])
                st.write("Description:")
                st.write(t["description"])
                if t["ai_response"]:
                    st.markdown("**AI tried:**")
                    st.write(t["ai_response"])
                # attachments
                c = conn.cursor()
                c.execute("SELECT filename, filepath FROM attachments WHERE ticket_id=?", (t["id"],))
                attachments = c.fetchall()
                if attachments:
                    st.markdown("Attachments:")
                    for a in attachments:
                        st.write(f"- {a['filename']} (saved: {a['filepath']})")
                # actions
                claim_key = f"claim_{t['id']}"
                if st.button("Claim & Reply", key=claim_key):
                    st.session_state[f"claiming_{t['id']}"] = True
                    st.experimental_rerun()
                if st.session_state.get(f"claiming_{t['id']}", False):
                    st.markdown("**Reply & Resolve**")
                    reply = st.text_area("Your response to the user", key=f"reply_{t['id']}")
                    save_to_kb = st.checkbox("Save this Q&A to KB (so AI learns from this answer)", key=f"savekb_{t['id']}")
                    submitted = st.button("Send reply & mark resolved", key=f"submit_resolve_{t['id']}")
                    if submitted:
                        update_ticket(t["id"], resolver_response=reply, assigned_to=st.session_state.auth["user"], status="resolved", needs_human=0)
                        st.success("Ticket marked as resolved and user notified.")
                        notify_user_placeholder(t["id"], "Your ticket has been resolved by the admin.")
                        if save_to_kb:
                            kb_path = save_qna_to_kb(t["title"], t["description"], reply)
                            update_ticket(t["id"], saved_to_kb=1)
                            st.success(f"Saved Q&A to KB at `{kb_path}`")
                        st.session_state.pop(f"claiming_{t['id']}", None)
                        st.experimental_rerun()

    st.markdown("---")
    st.subheader("All tickets (for audit)")
    all_tickets = list_tickets(filter_by="unresolved")
    if not all_tickets:
        st.write("No tickets found.")
    else:
        for t in all_tickets:
            st.write(f"{t['created_at']} — {t['id']} — {t['title']} — status: {t['status']} — needs_human: {t['needs_human']}")

# -------------------------
# App entrypoint
# -------------------------
if not st.session_state.auth["logged_in"]:
    login_page()
else:
    role = st.session_state.auth["role"]
    st.sidebar.markdown(f"Logged in as **{st.session_state.auth['user']}** — role: **{role}**")
    if role == "user":
        user_dashboard()
    elif role == "admin":
        admin_dashboard()
    else:
        st.error("Unknown role. Please logout and login again.")
