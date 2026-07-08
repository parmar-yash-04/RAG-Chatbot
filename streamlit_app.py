import os
import time
import streamlit as st
import requests

API_BASE = os.getenv("API_BASE", "https://rag-chatbot-production-3142.up.railway.app")

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="centered")

st.markdown(
    """
<style>
/* ── reset & base ── */
.block-container { padding-top: 1.5rem !important; max-width: 780px !important; }

/* ── chat messages ── */
.stChatMessage {
    border-radius: 18px !important;
    padding: 10px 16px !important;
    margin: 6px 0 !important;
    border: none !important;
}
[data-testid="stChatMessage"]:has(div:empty) { display: none; }

/* user bubble — right aligned */
[data-testid="stChatMessage"][data-testid$="user"] {
    background: #e3f2fd !important;
    margin-left: auto !important;
    max-width: 78% !important;
    border-bottom-right-radius: 4px !important;
}

/* assistant bubble */
[data-testid="stChatMessage"][data-testid$="assistant"] {
    background: #f5f5f5 !important;
    max-width: 100% !important;
    border-bottom-left-radius: 4px !important;
}

/* typing cursor animation */
.typing-cursor::after {
    content: "▊";
    animation: blink 0.7s step-end infinite;
    color: #666;
    margin-left: 2px;
}
@keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
}

/* ── sidebar ── */
[data-testid="stSidebar"] { width: 320px !important; }
[data-testid="stSidebar"] .stHeading { font-size: 1.1rem; }
section[data-testid="stSidebar"] div.st-emotion-cache-1y4p8pa {
    padding-top: 1.5rem;
}

/* source pill */
.source-pill {
    display: inline-block;
    background: #e8e8e8;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin: 2px 3px;
    color: #555;
    cursor: default;
}
.source-pill:hover { background: #ddd; }

/* file list items */
.doc-item {
    background: #fafafa;
    border: 1px solid #eee;
    border-radius: 10px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.85rem;
}
.doc-item:hover { background: #f0f0f0; }

/* ── hide default Streamlit bits ── */
#MainMenu { visibility: hidden; }
footer { display: none; }
</style>
""",
    unsafe_allow_html=True,
)

# ── session state ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0
if "streaming" not in st.session_state:
    st.session_state.streaming = False

# ── helpers ──
def upload_doc(file):
    with st.spinner("Indexing..."):
        files = {"file": (file.name, file.getvalue(), file.type)}
        resp = requests.post(f"{API_BASE}/upload", files=files)
        if resp.ok:
            st.toast(f"✅ {resp.json()['message']}", icon=None)
        else:
            st.error(resp.text)
    st.session_state.upload_key += 1
    st.rerun()


def fetch_docs():
    try:
        resp = requests.get(f"{API_BASE}/documents", timeout=5)
        return resp.json() if resp.ok else []
    except Exception:
        return []


def yield_answer(answer):
    """Generator that yields chunks for st.write_stream — word-level for natural flow."""
    words = answer.split(" ")
    # yield first few words character-by-char for a quick start
    first = words[0] if words else ""
    for c in first:
        yield c
        time.sleep(0.008)
    yield " "
    time.sleep(0.005)
    # yield rest word by word
    for word in words[1:]:
        yield word + " "
        time.sleep(0.018)
# ── end helpers ──

# ──────────────────────────── SIDEBAR ────────────────────────────
with st.sidebar:
    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload",
        type=["txt", "pdf", "docx"],
        key=f"upload_{st.session_state.upload_key}",
        label_visibility="collapsed",
    )
    if uploaded_file:
        col_a, col_b = st.columns([3, 1])
        col_a.caption(uploaded_file.name)
        if col_b.button("📤", use_container_width=True, type="primary"):
            upload_doc(uploaded_file)

    st.divider()
    st.markdown("### 📚 Indexed Documents")
    docs = fetch_docs()
    if not docs:
        st.caption("No documents yet.")
    else:
        for doc in docs:
            cols = st.columns([4, 1])
            cols[0].markdown(
                f'<div class="doc-item">'
                f"<b>{doc['doc_name']}</b><br>"
                f'<span style="color:#888;font-size:0.75rem">{doc["chunk_count"]} chunks</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
            if cols[1].button("🗑️", key=f"del_{doc['doc_id']}", help="Delete"):
                requests.delete(f"{API_BASE}/documents/{doc['doc_id']}")
                st.rerun()

    st.divider()
    st.caption(f"💡 Ask questions about your uploaded documents")

# ──────────────────────────── MAIN CHAT ────────────────────────────
st.title("💬 RAG Chatbot")

# ── chat history ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        body = msg.get("content", "")
        if "error" in msg and msg["error"]:
            st.error(body)
        else:
            st.markdown(body)
        # sources as pills
        if "sources" in msg and msg["sources"]:
            src_html = "".join(
                f'<span class="source-pill" title="{s["doc_name"]} · score {s["score"]:.2f}">'
                f"📎 {s['doc_name']}</span>"
                for s in msg["sources"]
            )
            st.markdown(f'<div style="margin-top:4px">{src_html}</div>', unsafe_allow_html=True)

# ── chat input — disabled while streaming ──
if prompt := st.chat_input(
    "Ask a question...", disabled=st.session_state.get("streaming", False)
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown(
            '<div class="typing-cursor" style="color:#999;font-size:0.9rem">Thinking</div>',
            unsafe_allow_html=True,
        )

        try:
            st.session_state.streaming = True
            resp = requests.post(
                f"{API_BASE}/chat",
                json={"question": prompt, "top_k": 5},
                timeout=60,
            )
            if resp.ok:
                data = resp.json()
                answer = data["answer"]
                sources = data.get("sources", [])

                placeholder.empty()
                placeholder.write_stream(yield_answer(answer))

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )
            else:
                placeholder.error(f"Error: {resp.text}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": resp.text, "error": True}
                )
        except Exception as e:
            placeholder.error(f"Connection error: {e}")
            st.session_state.messages.append(
                {"role": "assistant", "content": str(e), "error": True}
            )
        finally:
            st.session_state.streaming = False
