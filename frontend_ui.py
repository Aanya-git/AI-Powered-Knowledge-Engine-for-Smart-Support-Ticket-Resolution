import os
import sqlite3
import uuid
import pathlib
import time
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

# -------------------------
# Config / defaults
# -------------------------
load_dotenv()
APP_USER = os.getenv("APP_USER", "user")
APP_PASS = os.getenv("APP_PASS", "user123")
SUPPORT_USER = os.getenv("SUPPORT_USER", "support")
SUPPORT_PASS = os.getenv("SUPPORT_PASS", "admin123")
KB_DIR = os.getenv("KB_DIR", "kb_docs")
DB_PATH = os.getenv("TICKETS_DB", "tickets.db")

pathlib.Path(KB_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path("attachments").mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Smart Support — Frontend (chat UI)", layout="wide")

# -------------------------
# Try to import optional backend RAG functions (best-effort)
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
    return conn

conn = get_conn()

def get_table_columns(conn, table_name):
    """Return list of columns in table (or [] if table missing)."""
    try:
        c = conn.cursor()
        c.execute(f"PRAGMA table_info({table_name})")
        cols = [r["name"] for r in c.fetchall()]
        return cols
    except Exception:
        return []

def find_message_column(conn):
    """Find a suitable column name in messages table that holds text content."""
    columns = get_table_columns(conn, "messages")
    # common choices
    for candidate in ["content", "message", "body", "text", "msg", "content_text"]:
        if candidate in columns:
            return candidate
    # fallback: return first text-like column other than id/ticket_id/sender/created_at
    if columns:
        for c in columns:
            if c.lower() not in ("id", "ticket_id", "sender", "created_at", "createdat"):
                return c
        if len(columns) >= 1:
            return columns[-1]
    return None

MSG_COL = find_message_column(conn)  # will be None if no messages table

# -------------------------
# Utility helpers
# -------------------------
def safe_rerun():
    """Rerun if possible without raising attribute errors across streamlit versions."""
    if hasattr(st, "experimental_rerun"):
        try:
            st.experimental_rerun()
        except Exception:
            pass
    elif hasattr(st, "rerun"):
        try:
            st.rerun()
        except Exception:
            pass

def login_check(user_input, pass_input, role):
    """Return True if credentials match configured demo creds."""
    if role == "user":
        return user_input == APP_USER and pass_input == APP_PASS
    elif role == "support":
        return user_input == SUPPORT_USER and pass_input == SUPPORT_PASS
    return False

def create_ticket(title, description, creator):
    ticket_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    c = conn.cursor()
    # keep same schema as original project if present; try to insert minimally
    try:
        c.execute(
            "INSERT INTO tickets (id,title,description,created_at,status,priority,creator,assigned_to,ai_response,needs_human,resolver_response,feedback,saved_to_kb) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ticket_id, title, description, created_at, "open", "Low", creator,
             None, None, 1, None, None, 0),
        )
    except Exception:
        # if original schema different, try minimal insert columns
        try:
            c.execute(
                "INSERT INTO tickets (id,title,description,created_at,status,creator) "
                "VALUES (?,?,?,?,?,?)",
                (ticket_id, title, description, created_at, "open", creator),
            )
        except Exception as e:
            st.error(f"Failed to create ticket (schema mismatch): {e}")
            return None
    conn.commit()
    return ticket_id

def list_tickets_for_agent():
    c = conn.cursor()
    try:
        c.execute("SELECT * FROM tickets WHERE status='open' ORDER BY created_at DESC")
        rows = c.fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        st.error(f"Listing tickets failed: {e}")
        return []

def list_user_tickets(user):
    c = conn.cursor()
    try:
        c.execute(
            "SELECT id,title,status,created_at,ai_response,needs_human,feedback "
            "FROM tickets WHERE creator=? ORDER BY created_at DESC",
            (user,),
        )
        return [dict(r) for r in c.fetchall()]
    except Exception as e:
        st.error(f"Failed loading your tickets: {e}")
        return []

def add_message_to_db(ticket_id, sender, content):
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    inserted = False
    try:
        c.execute(
            "INSERT INTO messages (id,ticket_id,sender,content,created_at) "
            "VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), ticket_id, sender, content, now),
        )
        inserted = True
    except Exception:
        try:
            cols = get_table_columns(conn, "messages")
            if "ticket_id" in cols and "sender" in cols and MSG_COL:
                q = f"INSERT INTO messages (ticket_id,sender,{MSG_COL},created_at) VALUES (?,?,?,?)"
                c.execute(q, (ticket_id, sender, content, now))
                inserted = True
            else:
                if len(cols) >= 1:
                    q = f"INSERT INTO messages ({cols[0]}) VALUES (?)"
                    c.execute(q, (content,))
                    inserted = True
        except Exception as e:
            st.error(f"Failed to save message to DB: {e}")
    if inserted:
        conn.commit()
    return inserted

def get_messages_for_ticket(ticket_id):
    """Return ordered messages for a ticket as list of dicts {sender, content, created_at}."""
    c = conn.cursor()
    try:
        if MSG_COL:
            q = (
                f"SELECT sender, {MSG_COL} as content, created_at "
                f"FROM messages WHERE ticket_id=? ORDER BY created_at ASC"
            )
            c.execute(q, (ticket_id,))
            rows = c.fetchall()
            return [
                {
                    "sender": r["sender"],
                    "content": r["content"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ]
        else:
            c.execute(
                "SELECT * FROM messages WHERE ticket_id=? ORDER BY created_at ASC",
                (ticket_id,),
            )
            rows = c.fetchall()
            result = []
            for r in rows:
                d = dict(r)
                text_field = None
                for k, v in d.items():
                    if isinstance(v, str) and k not in (
                        "id",
                        "ticket_id",
                        "sender",
                        "created_at",
                    ):
                        text_field = v
                        break
                result.append(
                    {
                        "sender": d.get("sender", "<unknown>"),
                        "content": text_field or str(d),
                        "created_at": d.get("created_at"),
                    }
                )
            return result
    except Exception:
        return []

# -------------------------
# Chat UI helper (WhatsApp-like)
# -------------------------
def render_chat_messages(messages, me, customer_name=None):
    """
    Render messages in a WhatsApp-like style.
    - `me`           : username of the current logged-in user
    - `customer_name`: for agent view, the ticket creator's name (so we can label properly)
    """

    use_chat_api = hasattr(st, "chat_message")

    # small CSS tweaks for HTML fallback
    if not use_chat_api:
        st.markdown(
            """
            <style>
            .chat-bubble {
                max-width: 70%;
                padding: 8px 12px;
                margin: 4px 0;
                border-radius: 12px;
                font-size: 0.9rem;
            }
            .chat-me {
                background-color: #DCF8C6;
                margin-left: auto;
            }
            .chat-other {
                background-color: #FFFFFF;
                border: 1px solid #e5e5e5;
                margin-right: auto;
            }
            .chat-meta {
                font-size: 0.7rem;
                color: #777777;
                margin-top: 3px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    for m in messages:
        sender = m.get("sender", "")
        content = m.get("content", "")
        created_at = m.get("created_at", "")

        is_me = sender == me

        if sender == me:
            label = "You"
        elif customer_name and sender == customer_name:
            label = "Customer"
        else:
            label = sender or "Support"

        if use_chat_api:
            role = "user" if is_me or (customer_name and sender == customer_name) else "assistant"
            with st.chat_message(role):
                st.markdown(f"**{label}**")
                st.markdown(content)
                if created_at:
                    st.markdown(
                        f"<div style='font-size:0.7rem;color:#777;'>{created_at}</div>",
                        unsafe_allow_html=True,
                    )
        else:
            # HTML fallback
            bubble_class = "chat-me" if is_me else "chat-other"
            st.markdown(
                f"""
                <div class="chat-bubble {bubble_class}">
                    <strong>{label}</strong><br/>
                    {content}
                    <div class="chat-meta">{created_at}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# -------------------------
# AI helper (pluggable)
# -------------------------
def call_ai_agent_for_ticket(context_text):
    """Return (answer_text, resolved_bool, needs_human_bool). If no backend, return fallback."""
    if ai_support_agent:
        try:
            ans = ai_support_agent(context_text)
            if not ans:
                return ("", False, True)
            lowered = ans.strip().lower()
            if any(
                x in lowered
                for x in [
                    "let me connect",
                    "forwarding to human",
                    "cannot",
                ]
            ):
                return (ans, False, True)
            return (ans, True, False)
        except Exception as e:
            return (f"AI call failed: {e}", False, True)
    else:
        return ("AI backend not available. Agent should reply manually.", False, True)

# -------------------------
# Authentication & session
# -------------------------
if "auth" not in st.session_state:
    st.session_state.auth = {"logged_in": False, "role": None, "user": None}

def do_logout():
    st.session_state.auth = {"logged_in": False, "role": None, "user": None}
    safe_rerun()

# -------------------------
# UI: login page
# -------------------------
def login_page():
    st.title("Smart Support — Login")
    st.write("Login as **Customer** or **Support Agent**.")
    with st.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        role = st.radio("Login as", ("user", "support"))
        submitted = st.form_submit_button("Login")
        if submitted:
            if login_check(u.strip(), p.strip(), role):
                st.session_state.auth = {
                    "logged_in": True,
                    "role": role,
                    "user": u.strip(),
                }
                st.success("Logged in.")
                safe_rerun()
            else:
                st.error(f"Invalid credentials for role: {role}")

# -------------------------
# UI: user dashboard (customer)
# -------------------------
def user_dashboard():
    st.sidebar.markdown("### Actions")
    if st.sidebar.button("Logout"):
        do_logout()

    if connected_backend:
        st.sidebar.success(f"Backend connected: {connected_backend}")
    else:
        st.sidebar.info("Backend not connected — AI features unavailable")

    st.header("Customer Portal")

    tab_raise, tab_chat = st.tabs(["Raise query", "Chat with Support"])

    # ---- RAISE QUERY TAB ----
    with tab_raise:
        st.subheader("Create a new query")
        st.caption("Your query will be handled by a support agent. AI only assists the agent.")
        with st.form("raise_query_form", clear_on_submit=True):
            title = st.text_input("Issue title")
            description = st.text_area("Describe the issue in detail")
            attachments = st.file_uploader(
                "Add attachments (optional)", accept_multiple_files=True
            )
            submit = st.form_submit_button("Submit query (agent will reply)")
            if submit:
                if not title.strip() or not description.strip():
                    st.warning("Please provide both title and description.")
                else:
                    ticket_id = create_ticket(
                        title.strip(),
                        description.strip(),
                        st.session_state.auth["user"],
                    )
                    if ticket_id:
                        # save attachments to table if it exists
                        for f in attachments:
                            sid = str(uuid.uuid4())
                            save_dir = pathlib.Path("attachments")
                            filepath = save_dir / f"{sid}_{f.name}"
                            with open(filepath, "wb") as fh:
                                fh.write(f.getbuffer())
                            try:
                                c = conn.cursor()
                                c.execute(
                                    "INSERT INTO attachments (id,ticket_id,filename,filepath) "
                                    "VALUES (?,?,?,?)",
                                    (sid, ticket_id, f.name, str(filepath)),
                                )
                                conn.commit()
                            except Exception:
                                pass
                        st.success(
                            f"Query created: {ticket_id}. A support agent will respond shortly."
                        )
                        add_message_to_db(
                            ticket_id,
                            st.session_state.auth["user"],
                            description.strip(),
                        )
                    else:
                        st.error("Query creation failed.")

    # ---- CHAT TAB ----
    with tab_chat:
        st.subheader("Chat with support")

        rows = list_user_tickets(st.session_state.auth["user"])
        if not rows:
            st.info("You have not raised any queries yet.")
            return

        ticket_map = {
            f"{r['title']} — [{r['status']}] — {r['created_at']}": r["id"] for r in rows
        }
        sel_label = st.selectbox("Select a query", options=list(ticket_map.keys()))
        t_id = ticket_map[sel_label]

        st.markdown("---")
        st.markdown("#### Conversation")

        messages = get_messages_for_ticket(t_id)

        # WhatsApp-like chat for customer
        render_chat_messages(messages, me=st.session_state.auth["user"])

        st.markdown("---")
        new_msg = st.text_area(
            "Write your message to support", key=f"usermsg_{t_id}", height=80
        )
        if st.button("Send to support", key=f"send_{t_id}"):
            if new_msg.strip():
                ok = add_message_to_db(
                    t_id, st.session_state.auth["user"], new_msg.strip()
                )
                if ok:
                    st.success("Message sent to support agent.")
                    safe_rerun()
                else:
                    st.error("Failed to send message.")
            else:
                st.warning("Type a message before sending.")

# -------------------------
# UI: agent dashboard (support)
# -------------------------
def agent_dashboard():
    st.sidebar.markdown("### Actions")
    if st.sidebar.button("Logout"):
        do_logout()

    st.title("Support Agent")
    st.markdown("Open queries assigned to support. AI suggestions are only visible to you.")

    tickets = list_tickets_for_agent()
    if not tickets:
        st.info("No open queries awaiting support.")
        return

    ticket_labels = {
        f"{t.get('title','(no title)')} — {t.get('created_at','')}": t["id"]
        for t in tickets
    }
    sel_label = st.selectbox("Select an open query", list(ticket_labels.keys()))
    open_id = ticket_labels[sel_label]

    # Load selected ticket
    c = conn.cursor()
    try:
        c.execute("SELECT * FROM tickets WHERE id=?", (open_id,))
        row = c.fetchone()
        if not row:
            st.error("Ticket not found.")
            return
        t = dict(row)
    except Exception as e:
        st.error(f"Failed to load ticket: {e}")
        return

    st.markdown("---")
    st.subheader(f"Ticket: {t.get('title')} — {open_id}")
    st.write("Creator:", t.get("creator"))
    st.write("Description:")
    st.write(t.get("description"))

    # Conversation panel
    st.markdown("#### Chat with customer")
    messages = get_messages_for_ticket(open_id)

    creator = t.get("creator")
    # WhatsApp-like chat for agent; "me" is support user
    render_chat_messages(messages, me=st.session_state.auth["user"], customer_name=creator)

    # Agent actions
    st.markdown("---")
    st.markdown("#### Agent tools")

    if st.button("Get AI Suggestion for this query"):
        context_text = t.get("description", "") + "\n\nMessages:\n"
        for m in messages:
            context_text += f"{m['sender']}: {m['content']}\n"
        ans, resolved, needs_human = call_ai_agent_for_ticket(context_text)
        st.session_state[f"ai_suggestion_{open_id}"] = ans
        if ans:
            st.success("AI suggestion retrieved.")
        else:
            st.info("AI did not return suggestion.")

    ai_s = st.session_state.get(f"ai_suggestion_{open_id}", "")
    if ai_s:
        st.markdown("**AI Suggestion (preview only for agent)**")
        st.write(ai_s)
        if st.button("Use suggestion as reply"):
            added = add_message_to_db(open_id, st.session_state.auth["user"], ai_s)
            if added:
                st.success("Suggestion added as agent reply.")
                safe_rerun()

    reply_text = st.text_area(
        "Your reply to customer (manual)", key=f"reply_{open_id}", height=80
    )
    if st.button("Send reply to customer"):
        if not reply_text.strip():
            st.warning("Enter reply text.")
        else:
            ok = add_message_to_db(
                open_id, st.session_state.auth["user"], reply_text.strip()
            )
            if ok:
                try:
                    c.execute(
                        "UPDATE tickets SET status=?, needs_human=? WHERE id=?",
                        ("resolved", 0, open_id),
                    )
                    conn.commit()
                except Exception:
                    pass
                st.success("Reply sent and ticket marked resolved.")
                safe_rerun()
            else:
                st.error("Failed to save reply.")

# -------------------------
# Entrypoint
# -------------------------
def main():
    if not st.session_state.auth.get("logged_in", False):
        login_page()
    else:
        role = st.session_state.auth["role"]
        st.sidebar.markdown(
            f"Logged in as **{st.session_state.auth['user']}** — role: **{role}**"
        )
        if role == "user":
            user_dashboard()
        elif role == "support":
            agent_dashboard()
        else:
            st.error("Unknown role. Log out and back in.")
            if st.button("Logout"):
                do_logout()

if __name__ == "__main__":
    main()
