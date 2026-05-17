"""
app.py
------
Streamlit frontend for the Research Assistant.
Calls the FastAPI backend at API_BASE_URL for all operations.

Layout:
  - Left sidebar  : document ingestion (file upload / URL) + job polling
  - Centre column : chat interface — newest Q&A at top, numbered Q1/A1…
  - Right column  : evidence panel — highlighted PDF page + sources

Run with:
    streamlit run app.py
"""

import json
import time
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


# -------------------------------------------------------------
# api_get / api_post: thin wrappers around requests that return
# None on connection errors so the UI degrades gracefully
# -------------------------------------------------------------
def api_get(path):
    try:
        return requests.get(f"{API_BASE_URL}{path}", timeout=5)
    except requests.exceptions.ConnectionError:
        return None


def api_post(path, **kwargs):
    try:
        return requests.post(f"{API_BASE_URL}{path}", timeout=120, **kwargs)
    except requests.exceptions.ConnectionError:
        return None


# -------------------------------------------------------------
# _stream_query: calls POST /query/stream and renders the answer
# token-by-token using st.write_stream. Returns (answer, sources,
# latency_ms) once the stream is complete.
# Sources arrive in the first SSE event so the evidence panel can
# be pre-populated via st.session_state before the answer finishes.
# -------------------------------------------------------------
def _stream_query(question: str):
    sources = []
    latency_ms = None

    try:
        resp = requests.post(
            f"{API_BASE_URL}/query/stream",
            json={"question": question},
            stream=True,
            timeout=120,
        )
    except requests.exceptions.ConnectionError:
        return "Could not reach the API. Is the backend running?", [], None

    if resp.status_code == 503:
        return "No documents are indexed yet. Upload a PDF or enter a URL in the sidebar first.", [], None
    if resp.status_code != 200:
        return f"Error {resp.status_code}: {resp.text}", [], None

    first_token_seen = [False]

    def _token_generator():
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: "):])

            if event["type"] == "sources":
                sources.extend(event["sources"])
                st.session_state.last_sources = sources

            elif event["type"] == "token":
                if not first_token_seen[0]:
                    thinking_placeholder.empty()
                    first_token_seen[0] = True
                yield event["content"]
                time.sleep(0.02)

            elif event["type"] == "done":
                nonlocal latency_ms
                latency_ms = event.get("latency_ms")

            elif event["type"] == "error":
                if not first_token_seen[0]:
                    thinking_placeholder.empty()
                    first_token_seen[0] = True
                yield f"\n\n[Error: {event.get('detail', 'unknown')}]"

    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("_Thinking..._")
        answer = st.write_stream(_token_generator())

    return answer, sources, latency_ms


# -------------------------------------------------------------
# initialise_session: sets up session state keys on first load
# so the rest of the app can safely read them
# -------------------------------------------------------------
def initialise_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "job_id" not in st.session_state:
        st.session_state.job_id = None
    if "job_label" not in st.session_state:
        st.session_state.job_label = ""
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []


# -------------------------------------------------------------
# render_sidebar: handles document ingestion (file upload and
# URL), shows live job status, and displays index stats
# -------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.title("Research Assistant")
        st.caption("RAG over research papers")

        st.divider()

        # --- API status ---
        resp = api_get("/health")
        if resp and resp.status_code == 200:
            st.success("API connected", icon="✅")
        else:
            st.error("API not reachable", icon="🔴")
            st.stop()

        # --- index stats ---
        resp = api_get("/status")
        if resp and resp.status_code == 200:
            data = resp.json()
            st.caption(f"Chunks indexed in memory: {data['chunks_in_memory']}")

        st.divider()

        # --- document ingestion ---
        st.subheader("Add Documents")

        tab_file, tab_url = st.tabs(["Upload PDF", "URL"])

        with tab_file:
            uploaded = st.file_uploader("Choose a PDF", type="pdf", label_visibility="collapsed")
            if st.button("Ingest PDF", disabled=uploaded is None):
                resp = api_post("/ingest/file", files={"file": (uploaded.name, uploaded, "application/pdf")})
                if resp and resp.status_code == 200:
                    data = resp.json()
                    st.session_state.job_id = data["job_id"]
                    st.session_state.job_label = uploaded.name
                    st.session_state.last_sources = []
                else:
                    st.error("Ingestion failed.")

        with tab_url:
            url_input = st.text_input("PDF or arXiv URL", placeholder="https://arxiv.org/pdf/...")
            if st.button("Ingest URL", disabled=not url_input):
                resp = api_post("/ingest/url", json={"url": url_input})
                if resp and resp.status_code == 200:
                    data = resp.json()
                    st.session_state.job_id = data["job_id"]
                    st.session_state.job_label = url_input
                    st.session_state.last_sources = []
                else:
                    st.error("Ingestion failed.")

        # --- job polling ---
        if st.session_state.job_id:
            st.divider()
            resp = api_get(f"/ingest/status/{st.session_state.job_id}")
            if resp and resp.status_code == 200:
                job = resp.json()

                if job["status"] == "done":
                    st.success(f"Indexed {job['chunks_indexed']} chunks from {st.session_state.job_label}")
                    st.session_state.job_id = None

                elif job["status"] == "failed":
                    st.error(f"Ingestion failed: {job['message']}")
                    st.session_state.job_id = None

                else:
                    st.info(f"⏳ {job['message'] or job['status'].capitalize()}...")
                    time.sleep(2)
                    st.rerun()


# -------------------------------------------------------------
# render_evidence: renders the top-ranked source chunk on its
# PDF page with a red bounding box, then lists all sources below.
# ADE pages are 0-indexed; the evidence endpoint uses 1-indexed.
# -------------------------------------------------------------
def render_evidence(col, sources):
    with col:
        st.subheader("Evidence")

        if not sources:
            st.caption("Ask a question to see the relevant evidence highlighted in the paper.")
            return

        # pick the top-ranked chunk that has a valid page and bounding box
        top = next(
            (c for c in sources
             if c.get("page") is not None
             and all(c.get(k) is not None for k in ("box_top", "box_left", "box_right", "box_bottom"))),
            None,
        )

        if top is not None:
            pdf_page = top["page"] + 1   # convert 0-indexed ADE page to 1-indexed PDF page
            box = {
                "top":    top["box_top"],
                "left":   top["box_left"],
                "right":  top["box_right"],
                "bottom": top["box_bottom"],
            }
            resp = api_post(f"/evidence/page/{pdf_page}", json={"boxes": [box]})
            if resp and resp.status_code == 200:
                st.caption(f"Page {pdf_page}")
                st.image(resp.content, use_container_width=True)
            else:
                detail = ""
                if resp is not None:
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except Exception:
                        detail = resp.text
                st.warning(f"Could not render page {pdf_page}: {detail}" if detail else f"Could not render page {pdf_page}.")
        else:
            st.caption("No bounding box data available for these sources.")

        # --- collapsible sources list below the evidence image ---
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        with st.expander(f"Sources ({len(sources)})"):
            for i, chunk in enumerate(sources):
                page_display = (chunk.get("page") or 0) + 1   # convert to 1-indexed for display
                st.markdown(f"**{chunk['source']}** — page {page_display}")
                st.caption(chunk["content"][:400])
                if i < len(sources) - 1:
                    st.divider()


# -------------------------------------------------------------
# render_chat: question form at the top, Q&A history below it
# newest-first with sequential numbering (Q1/A1, Q2/A2…).
# Latest pair is expanded; all previous pairs are collapsed.
# Sources are shown in the evidence panel, not here.
# st.chat_input is always pinned to the viewport bottom, so we
# use st.form instead to keep the input at the top of the column.
# -------------------------------------------------------------
def render_chat(col):
    with col:
        st.header("Ask your document")

        # --- question input at the top (Enter to submit, button hidden via CSS) ---
        with st.form("chat_form", clear_on_submit=True):
            question = st.text_input(
                "question",
                placeholder="Ask about your document...",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Ask")

        if submitted and question.strip():
            answer, sources, latency_ms = _stream_query(question.strip())
            st.session_state.messages.append({"role": "user", "content": question.strip()})
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "latency_ms": latency_ms,
            })
            st.session_state.last_sources = sources
            st.rerun()

        # --- Q&A history below the input ---
        msgs = st.session_state.messages
        pairs = []
        i = 0
        while i + 1 < len(msgs):
            if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                pairs.append((msgs[i], msgs[i + 1]))
                i += 2
            else:
                i += 1

        total = len(pairs)
        for rev_idx, (user_msg, asst_msg) in enumerate(reversed(pairs)):
            q_num = total - rev_idx
            is_latest = rev_idx == 0

            if is_latest:
                with st.chat_message("user"):
                    st.markdown(f"**Q{q_num}:** {user_msg['content']}")
                with st.chat_message("assistant"):
                    st.markdown(f"**A{q_num}:**")
                    st.markdown(asst_msg['content'])
                    if asst_msg.get("latency_ms"):
                        st.caption(f"Answered in {asst_msg['latency_ms']} ms")
            else:
                with st.expander(f"Q{q_num}: {user_msg['content'][:80]}{'…' if len(user_msg['content']) > 80 else ''}"):
                    with st.chat_message("user"):
                        st.markdown(f"**Q{q_num}:** {user_msg['content']}")
                    with st.chat_message("assistant"):
                        st.markdown(f"**A{q_num}:**")
                        st.markdown(asst_msg['content'])
                        if asst_msg.get("latency_ms"):
                            st.caption(f"Answered in {asst_msg['latency_ms']} ms")


# -------------------------------------------------------------
# main: page config, CSS separator, session init, sidebar, and
# 2-column layout (chat on left, evidence on right)
# -------------------------------------------------------------
st.set_page_config(
    page_title="Research Assistant",
    page_icon="📄",
    layout="wide",
)

# add a visible left border on the evidence column to match the sidebar separation,
# and hide the chat form's submit button (Enter key submits instead)
st.markdown("""
<style>
[data-testid="stColumn"]:nth-child(2) {
    border-left: 1px solid rgba(49, 51, 63, 0.2);
    padding-left: 2rem !important;
}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

initialise_session()
render_sidebar()

chat_col, evidence_col = st.columns([3, 2])
render_chat(chat_col)
render_evidence(evidence_col, st.session_state.last_sources)
