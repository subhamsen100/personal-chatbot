"""
Streamlit UI for the Personal Knowledge Base AI.

Run (after starting the FastAPI backend):
    source chatenv/bin/activate
    streamlit run app.py
"""

import json
import uuid

import httpx
import streamlit as st

API_BASE = "http://localhost:8000/api"

# Human-readable labels for each tool
TOOL_LABELS = {
    "retrieve_from_knowledge_base": "Searching knowledge base",
    "calculate":                    "Computing math expression",
    "get_current_datetime":         "Fetching current date / time",
}


# helpers

def api(method: str, path: str, **kwargs):
    url = f"{API_BASE}{path}"
    try:
        resp = httpx.request(method, url, timeout=120, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error("Cannot reach the API. Is `uvicorn main:app` running on port 8000?")
        st.stop()
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
        return None


def fetch_sessions() -> list[dict]:
    data = api("GET", "/sessions")
    return data if data else []


def create_session() -> str:
    data = api("POST", "/sessions")
    return data["session_id"] if data else str(uuid.uuid4())


def fetch_history(session_id: str) -> list[dict]:
    data = api("GET", f"/sessions/{session_id}/history", params={"last_n": 100})
    return data if data else []


def stream_chat(session_id: str, message: str):
    """
    Generator that yields parsed event dicts from the NDJSON /api/chat stream.
    Handles connection and HTTP errors by yielding an error event.
    """
    url = f"{API_BASE}/chat"
    try:
        with httpx.stream(
            "POST", url,
            json={"session_id": session_id, "message": message},
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.strip():
                    yield json.loads(line)
    except httpx.ConnectError:
        yield {"type": "error", "message": "Cannot reach the API. Is uvicorn running on port 8000?"}
    except httpx.HTTPStatusError as e:
        yield {"type": "error", "message": f"API error {e.response.status_code}: {e.response.text}"}


def ingest_file(file_bytes: bytes, filename: str) -> dict | None:
    return api(
        "POST",
        "/ingest/file",
        files={"file": (filename, file_bytes)},
    )


def kb_stats() -> dict:
    data = api("GET", "/kb/stats")
    return data if data else {}


# page config─

st.set_page_config(
    page_title="Personal KB AI",
    page_icon="🧠",
    layout="wide",
)

# session state defaults

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # local display cache


# sidebar ─

with st.sidebar:
    st.title("🧠 Personal KB AI")
    st.caption("Offline · Grounded · Memory-Aware")
    st.divider()

    # Knowledge base stats
    stats = kb_stats()
    st.metric("KB chunks", stats.get("total_chunks", 0))
    sources = stats.get("sources", [])
    if sources:
        with st.expander(f"Indexed sources ({len(sources)})"):
            for s in sources:
                st.write(f"• {s}")

    st.divider()

    # Document upload
    st.subheader("📄 Upload Document")
    uploaded = st.file_uploader(
        "PDF, DOCX, MD, or TXT",
        type=["pdf", "docx", "md", "markdown", "txt"],
        label_visibility="collapsed",
    )
    if uploaded and st.button("Ingest", use_container_width=True):
        with st.spinner(f"Ingesting {uploaded.name}…"):
            result = ingest_file(uploaded.read(), uploaded.name)
        if result:
            st.success(f"✓ {result['chunks']} chunks indexed from {result['file']}")
            st.rerun()

    st.divider()

    # Session management
    st.subheader("💬 Sessions")

    if st.button("＋ New session", use_container_width=True):
        new_id = create_session()
        st.session_state.active_session_id = new_id
        st.session_state.chat_history = []
        st.rerun()

    sessions = fetch_sessions()
    if sessions:
        labels = {
            s["session_id"]: f"{s['session_id'][:8]}… ({s['updated_at'][:10]})"
            for s in sessions
        }
        chosen = st.selectbox(
            "Select session",
            options=[s["session_id"] for s in sessions],
            format_func=lambda sid: labels[sid],
            index=0,
            label_visibility="collapsed",
        )
        if chosen != st.session_state.active_session_id:
            st.session_state.active_session_id = chosen
            st.session_state.chat_history = fetch_history(chosen)
            st.rerun()

        if st.button("🗑 Delete session", use_container_width=True):
            api("DELETE", f"/sessions/{st.session_state.active_session_id}")
            st.session_state.active_session_id = None
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("No sessions yet. Click **＋ New session** to start.")


# main chat area 

if st.session_state.active_session_id is None:
    st.title("Welcome to your Personal Knowledge Base AI")
    st.markdown("""
    **Get started:**
    1. Upload a document (PDF, DOCX, Markdown, or TXT) in the sidebar.
    2. Click **＋ New session** to open a chat.
    3. Ask questions — the agent will retrieve answers *only* from your documents.

    > The model will say **"I cannot find this information in the knowledge base."**
    > if the answer is not in your documents — no hallucinations.
    """)
    st.stop()

st.title(f"Chat  —  `{st.session_state.active_session_id[:8]}…`")

# Render message history
for msg in st.session_state.chat_history:
    role = msg.get("role", "user")
    avatar = "🧑" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

# Input box
user_input = st.chat_input("Ask something about your documents…")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        answer       = ""
        tools_called = []
        total_ms     = 0

        # Real-time stage display
        with st.status("Processing query...", expanded=True) as status:
            for event in stream_chat(st.session_state.active_session_id, user_input):
                etype = event.get("type")

                if etype == "stage":
                    status.update(label=event["message"])

                elif etype == "tool_start":
                    tool  = event["tool"]
                    label = TOOL_LABELS.get(tool, f"Calling {tool}")
                    elapsed = event.get("elapsed_ms", 0)
                    status.update(label=f"{label}...")
                    st.markdown(f"**{label}** *(+{elapsed} ms)*")
                    args = event.get("args", {})
                    if args:
                        st.json(args, expanded=False)

                elif etype == "tool_done":
                    tool     = event["tool"]
                    tool_ms  = event.get("tool_ms", 0)
                    label    = TOOL_LABELS.get(tool, tool)
                    st.caption(f"{label} finished in {tool_ms} ms")
                    tools_called.append(event)

                elif etype == "answer":
                    answer       = event.get("answer", "")
                    total_ms     = event.get("total_ms", 0)
                    tools_called = event.get("tools_called", tools_called)
                    status.update(
                        label=f"Done in {total_ms} ms",
                        state="complete",
                        expanded=False,
                    )

                elif etype == "error":
                    status.update(label="Error", state="error")
                    st.error(event.get("message", "Unknown error"))
                    with st.expander("Traceback"):
                        st.code(event.get("trace", ""), language="python")
                    answer = f"[Error] {event.get('message', 'Unknown error')}"

        # Answer text (always visible, below the collapsed status)
        st.markdown(answer)

        # Per-tool timing chips
        if tools_called:
            cols = st.columns(len(tools_called))
            for col, t in zip(cols, tools_called):
                col.metric(
                    TOOL_LABELS.get(t["tool"], t["tool"]),
                    f"{t['tool_ms']} ms",
                )

        if total_ms:
            st.caption(f"Total query time: {total_ms} ms")

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
