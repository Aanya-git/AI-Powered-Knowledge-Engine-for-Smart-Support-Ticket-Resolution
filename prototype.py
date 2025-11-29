import os
import sqlite3
import uuid
import pathlib
from datetime import datetime

from dotenv import load_dotenv
import streamlit as st

# =========================
# Config / defaults
# =========================
load_dotenv()
APP_USER = os.getenv("APP_USER", "user")
APP_PASS = os.getenv("APP_PASS", "user123")
SUPPORT_USER = os.getenv("SUPPORT_USER", "support")
SUPPORT_PASS = os.getenv("SUPPORT_PASS", "admin123")
KB_DIR = os.getenv("KB_DIR", "kb_docs")
DB_PATH = os.getenv("TICKETS_DB", "tickets.db")

pathlib.Path(KB_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path("attachments").mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Smart Support — Frontend", layout="wide")

# =========================
# Optional backend RAG import
# =========================
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

# =========================
# DB helpers
# =========================
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

conn = get_conn()

def get_table_columns(conn, table_name):
    try:
        c = conn.cursor()
        c.execute(f"PRAGMA table_info({table_name})")
        return [r["name"] for r in c.fetchall()]
    except Exception:
        return []

def find_message_column(conn):
    columns = get_table_columns(conn, "messages")
    for candidate in ["content", "message", "body", "text", "msg", "content_text"]:
        if candidate in columns:
            return candidate
    if columns:
        for c in columns:
            if c.lower() not in ("id", "ticket_id", "sender", "created_at", "createdat"):
                return c
        return columns[-1]
    return None

MSG_COL = find_message_column(conn)

def safe_rerun():
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
    if role == "user":
        return user_input == APP_USER and pass_input == APP_PASS
    elif role == "support":
        return user_input == SUPPORT_USER and pass_input == SUPPORT_PASS
    return False

def create_ticket(title, description, creator):
    ticket_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    c = conn.cursor()
    try:
        c.execute(
            """
            INSERT INTO tickets
            (id,title,description,created_at,status,priority,creator,
             assigned_to,ai_response,needs_human,resolver_response,feedback,saved_to_kb)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                ticket_id,
                title,
                description,
                created_at,
                "open",
                "Low",
                creator,
                None,
                None,
                1,
                None,
                None,   # feedback is NULL initially
                0,
            ),
        )
    except Exception:
        try:
            c.execute(
                """
                INSERT INTO tickets
                (id,title,description,created_at,status,creator)
                VALUES (?,?,?,?,?,?)
                """,
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
        c.execute("SELECT * FROM tickets WHERE status='open' OR status='resolved' ORDER BY created_at DESC")
        return [dict(r) for r in c.fetchall()]
    except Exception as e:
        st.error(f"Listing tickets failed: {e}")
        return []

def list_user_tickets(user):
    c = conn.cursor()
    try:
        c.execute(
            """
            SELECT id,title,status,created_at,ai_response,needs_human,feedback
            FROM tickets
            WHERE creator=?
            ORDER BY created_at DESC
            """,
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
            "INSERT INTO messages (id,ticket_id,sender,content,created_at) VALUES (?,?,?,?,?)",
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
            elif cols:
                q = f"INSERT INTO messages ({cols[0]}) VALUES (?)"
                c.execute(q, (content,))
                inserted = True
        except Exception as e:
            st.error(f"Failed to save message to DB: {e}")
    if inserted:
        conn.commit()
    return inserted

def get_messages_for_ticket(ticket_id):
    c = conn.cursor()
    try:
        if MSG_COL:
            q = f"""
                SELECT sender, {MSG_COL} as content, created_at
                FROM messages
                WHERE ticket_id=?
                ORDER BY created_at ASC
            """
            c.execute(q, (ticket_id,))
            rows = c.fetchall()
            return [
                {"sender": r["sender"], "content": r["content"], "created_at": r["created_at"]}
                for r in rows
            ]
        else:
            c.execute("SELECT * FROM messages WHERE ticket_id=? ORDER BY created_at ASC", (ticket_id,))
            rows = c.fetchall()
            result = []
            for r in rows:
                d = dict(r)
                text_field = None
                for k, v in d.items():
                    if isinstance(v, str) and k not in ("id", "ticket_id", "sender", "created_at"):
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

def set_ticket_status(ticket_id, status=None, feedback=None, needs_human=None):
    c = conn.cursor()
    fields = []
    values = []
    if status is not None:
        fields.append("status=?")
        values.append(status)
    if feedback is not None:
        fields.append("feedback=?")
        values.append(feedback)
    if needs_human is not None:
        fields.append("needs_human=?")
        values.append(needs_human)
    if not fields:
        return
    values.append(ticket_id)
    try:
        c.execute(f"UPDATE tickets SET {', '.join(fields)} WHERE id=?", values)
        conn.commit()
    except Exception as e:
        st.error(f"Failed to update ticket: {e}")

# =========================
# Chat UI helper (simple WhatsApp-like)
# =========================
def render_chat_messages(messages, current_user, creator):
    """
    Simple left/right bubbles using Streamlit columns.
    - Customer messages -> left (blue)
    - Agent / current_user messages -> right (green)
    """
    for m in messages:
        sender = m.get("sender", "")
        content = m.get("content", "")
        created_at = m.get("created_at", "")

        is_me = sender == current_user
        is_customer = sender == creator

        left_col, right_col = st.columns([6, 6])

        if is_me:
            # current logged-in user on the right
            with right_col:
                st.markdown(
                    f"**You**",
                )
                st.success(f"{content}\n\n*{created_at}*")
        else:
            # other side (customer or support) on the left
            label = "Customer" if is_customer else sender or "Support"
            with left_col:
                st.markdown(f"**{label}**")
                st.info(f"{content}\n\n*{created_at}*")

# =========================
# AI helper
# =========================
def call_ai_agent_for_ticket(context_text):
    if ai_support_agent:
        try:
            ans = ai_support_agent(context_text)
            if not ans:
                return ("", False, True)
            lowered = ans.strip().lower()
            if any(x in lowered for x in ["let me connect", "forwarding to human", "cannot"]):
                return (ans, False, True)
            return (ans, True, False)
        except Exception as e:
            return (f"AI call failed: {e}", False, True)
    else:
        return ("AI backend not available. Agent should reply manually.", False, True)

# =========================
# Auth state
# =========================
if "auth" not in st.session_state:
    st.session_state.auth = {"logged_in": False, "role": None, "user": None}

def do_logout():
    st.session_state.auth = {"logged_in": False, "role": None, "user": None}
    safe_rerun()

# =========================
# Login page
# =========================
def login_page():
    st.title("Smart Support — Login")
    st.write("Login as **Customer** or **Support Agent**.")

    with st.form("login_form_unique"):
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

# =========================
# User dashboard (customer)
# =========================
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

    # ---- Raise query ----
    with tab_raise:
        st.subheader("Create a new query")
        st.caption("Your query will be handled by a support agent. AI only assists the agent.")

        with st.form("raise_query_form_unique", clear_on_submit=True):
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
                        for f in attachments:
                            sid = str(uuid.uuid4())
                            save_dir = pathlib.Path("attachments")
                            filepath = save_dir / f"{sid}_{f.name}"
                            with open(filepath, "wb") as fh:
                                fh.write(f.getbuffer())
                            try:
                                c = conn.cursor()
                                c.execute(
                                    "INSERT INTO attachments (id,ticket_id,filename,filepath) VALUES (?,?,?,?)",
                                    (sid, ticket_id, f.name, str(filepath)),
                                )
                                conn.commit()
                            except Exception:
                                pass
                        add_message_to_db(ticket_id, st.session_state.auth["user"], description.strip())
                        st.success(f"Query created: {ticket_id}. A support agent will respond shortly.")
                    else:
                        st.error("Query creation failed.")

    # ---- Chat with support ----
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
        st.markdown("### Conversation")

        messages = get_messages_for_ticket(t_id)
        render_chat_messages(messages, current_user=st.session_state.auth["user"], creator=st.session_state.auth["user"])

        # fetch ticket row to know status + feedback
        c = conn.cursor()
        c.execute("SELECT status, feedback FROM tickets WHERE id=?", (t_id,))
        t_row = c.fetchone()
        status = t_row["status"] if t_row else "open"
        feedback = t_row["feedback"] if t_row else None

        st.markdown("---")
        new_msg = st.text_area("Write your message to support", key=f"usermsg_{t_id}", height=80)
        if st.button("Send to support", key=f"send_{t_id}"):
            if new_msg.strip():
                ok = add_message_to_db(t_id, st.session_state.auth["user"], new_msg.strip())
                if ok:
                    st.success("Message sent to support agent.")
                    # once user sends new message after resolving, ticket can be reopened if you want:
                    set_ticket_status(t_id, status="open", needs_human=1)
                    safe_rerun()
                else:
                    st.error("Failed to send message.")
            else:
                st.warning("Type a message before sending.")

        # ---------- Feedback loop ----------
        # Show feedback buttons only if ticket is resolved and feedback is NULL / None
        if status == "resolved" and feedback is None:
            st.markdown("### Was this conversation helpful?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Yes, helpful", key=f"fb_yes_{t_id}"):
                    set_ticket_status(t_id, feedback=1)
                    st.success("Thanks! You marked this conversation as helpful.")
                    safe_rerun()
            with col2:
                if st.button("👎 Not helpful", key=f"fb_no_{t_id}"):
                    set_ticket_status(t_id, feedback=0, status="open", needs_human=1)
                    st.warning("You marked this conversation as not helpful. Support may follow up.")
                    safe_rerun()
        elif feedback == 1:
            st.info("✅ You marked this conversation as **helpful**.")
        elif feedback == 0:
            st.warning("⚠️ You marked this conversation as **not helpful**. Support may follow up again.")

# =========================
# Agent dashboard (support)
# =========================
def agent_dashboard():
    st.sidebar.markdown("### Actions")
    if st.sidebar.button("Logout"):
        do_logout()

    st.title("Support Agent")
    st.markdown("Open queries assigned to support. AI suggestions are only visible to you.")

    tickets = list_tickets_for_agent()
    if not tickets:
        st.info("No queries found.")
        return

    ticket_labels = {
        f"{t.get('title','(no title)')} — {t.get('created_at','')} — status:{t.get('status','')}"
        : t["id"]
        for t in tickets
    }
    sel_label = st.selectbox("Select a query", list(ticket_labels.keys()))
    open_id = ticket_labels[sel_label]

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
    st.write("Status:", t.get("status"))
    st.write("Description:")
    st.write(t.get("description"))

    st.markdown("### Conversation with customer")
    messages = get_messages_for_ticket(open_id)
    creator = t.get("creator")
    render_chat_messages(messages, current_user=st.session_state.auth["user"], creator=creator)

    st.markdown("---")
    st.markdown("### Agent tools")

    # AI suggestion
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
                # mark as resolved, wait for feedback from user
                set_ticket_status(open_id, status="resolved", needs_human=0)
                st.success("Suggestion added as agent reply. Waiting for user feedback.")
                safe_rerun()

    # Manual reply
    reply_text = st.text_area("Your reply to customer (manual)", key=f"reply_{open_id}", height=80)
    if st.button("Send reply to customer"):
        if not reply_text.strip():
            st.warning("Enter reply text.")
        else:
            ok = add_message_to_db(open_id, st.session_state.auth["user"], reply_text.strip())
            if ok:
                # mark as resolved; user will give feedback
                set_ticket_status(open_id, status="resolved", needs_human=0)
                st.success("Reply sent. Waiting for user feedback.")
                safe_rerun()
            else:
                st.error("Failed to save reply.")

    # Show feedback info if available
    if t.get("feedback") == 1:
        st.success("User marked this conversation as **helpful** ✅")
    elif t.get("feedback") == 0:
        st.warning("User marked this conversation as **not helpful** ❌ — consider following up.")

# =========================
# Entrypoint
# =========================
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