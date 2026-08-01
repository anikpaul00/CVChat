import streamlit as st
import os
import time

from src.pdf_loader import load_pdf
from src.embeddings import embedding_model
from src.vectorstore import create_vectorstore, create_retriever
from src.llm import get_llm
from src.rag import create_rag_chain
from src.utils import save_uploaded_file
from memory import update_chat_summary, summarizer


# ---------- PAGE CONFIG (must be first Streamlit call) ----------
st.set_page_config(
    page_title="CVChat",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed",
)
 
# ---------- SESSION STATE ----------
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_summary" not in st.session_state:
    st.session_state.chat_summary = ""

if "cv_name" not in st.session_state:
    st.session_state.cv_name = None

if "processing_error" not in st.session_state:
    st.session_state.processing_error = None
 
# ---------- STYLE ----------
st.markdown(
    """
    <style>
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    header {visibility:hidden;}
 
    .block-container {
        max-width: 780px;
        padding-top: 3rem;
        padding-bottom: 6rem;
    }
 
    .stApp { background:#ffffff; }
 
    /* Top bar */
    .topbar {
        position: fixed;
        top: 0; left: 0; right: 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 24px;
        background: #ffffff;
        border-bottom: 1px solid #f0f0f0;
        z-index: 100;
    }
    .topbar-title {
        font-weight: 600;
        font-size: 16px;
        color: #111827;
    }
 
    /* Empty-state hero, ChatGPT-style */
    .hero {
        text-align: center;
        margin-top: 18vh;
        font-size: 30px;
        font-weight: 600;
        color: #1f2937;
    }
    .hero-sub {
        text-align: center;
        color: #9ca3af;
        margin-top: 8px;
        font-size: 15px;
    }
 
    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        background: transparent;
        border: none;
        padding: 6px 0px;
    }
    [data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: #f7f7f8;
        border-radius: 14px;
        padding: 12px 16px;
    }
 
    /* Chat input */
    [data-testid="stChatInput"] {
        border-radius: 24px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
 
    .cv-badge {
        display: inline-block;
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        border-radius: 999px;
        padding: 3px 12px;
        font-size: 13px;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
 
# ---------- TOP BAR ----------
col_a, col_b = st.columns([5, 1])
with col_a:
    st.markdown('<div class="topbar-title">📄 CVChat</div>', unsafe_allow_html=True)
with col_b:
    if st.button("New chat", use_container_width=True, disabled=st.session_state.qa_chain is None):
        st.session_state.qa_chain = None
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.chat_summary = ""
        st.session_state.cv_name = None
        st.session_state.processing_error = None
        st.rerun()
 
if st.session_state.cv_name:
    st.markdown(
        f'<span class="cv-badge">✅ {st.session_state.cv_name}</span>',
        unsafe_allow_html=True,
    )
 
# ---------- EMPTY STATE ----------
if not st.session_state.messages and not st.session_state.qa_chain:
    st.markdown('<div class="hero">Attach a CV to get started</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Drop a PDF resume below, then ask about skills, '
        "experience, projects, or fit for a role.</div>",
        unsafe_allow_html=True,
    )
 
# ---------- PROCESSING ERROR (persisted) ----------
if st.session_state.processing_error:
    st.error(st.session_state.processing_error)
 
# ---------- CHAT HISTORY ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
 
# ---------- CHAT INPUT (text + optional PDF attach, ChatGPT-style) ----------
prompt = st.chat_input(
    "Ask anything, or attach a CV to begin",
    accept_file=True,
    file_type=["pdf"],
)
 
if prompt:
    question_text = (prompt.text or "").strip()
    attached_files = prompt.files if prompt.files else []
 
    # --- Handle a newly attached CV first ---
    if attached_files:
        uploaded_file = attached_files[0]
        st.session_state.processing_error = None
        with st.spinner("Reading resume..."):
            try:
                file_path = save_uploaded_file(uploaded_file)
                chunks = load_pdf(file_path)
                embeddings = embedding_model()
                vectorstore = create_vectorstore(chunks, embeddings)
                retriever = create_retriever(vectorstore, k=5)
                llm = get_llm()
                st.session_state.qa_chain = create_rag_chain(retriever, llm)
                st.session_state.cv_name = uploaded_file.name
                st.session_state.messages = []
                st.toast("Resume processed successfully!", icon="✅")
            except Exception as e:
                # Bug fix: original code had no error handling at all, so a
                # bad PDF or embedding/LLM failure crashed the whole app.
                st.session_state.qa_chain = None
                st.session_state.cv_name = None
                st.session_state.processing_error = f"Couldn't process that PDF: {e}"
                st.rerun()
 
    # --- Handle the question, if there is one and a CV is ready ---
    if question_text:
        if st.session_state.qa_chain is None:
            st.session_state.processing_error = "Please attach a CV before asking a question."
        else:
            # st.session_state.messages.append({"role": "user", "content": question_text})
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.chat_history.append({"user": question_text, "assistant:", answer})
            recent_history = st.session_state.chat_history[-3:]
            st.session_state.chat_summary = update_chat_summary(recent_history, summarizer)
            
            with st.chat_message("user"):
                st.write(question_text)
 
            def stream_rag_answer(qa_chain, question):
                try:
                    for chunk in qa_chain.stream(
                        {
                            "input": question,
                            "chat_summary": st.session_state.get("chat_summary", "")
                        }
                    ):
                        token = chunk.get("answer")
                        if token:
                            yield token
                            time.sleep(0.02)
            
                except Exception as e:
                    yield f"Something went wrong while answering: {e}"
 
            with st.chat_message("assistant"):
                answer = st.write_stream(
                    stream_rag_answer(st.session_state.qa_chain, question_text)
                )
 
            st.session_state.messages.append({"role": "assistant", "content": answer})
 
    st.rerun()
