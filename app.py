import streamlit as st
import os

from src.pdf_loader import load_pdf
from src.embeddings import embedding_model
from src.vectorstore import create_vectorstore, create_retriever
from src.llm import get_llm
from src.rag import create_rag_chain
from src.utils import save_uploaded_file


# ---------- UI STYLE ----------

st.markdown(
    """
    <style>

    /* Hide Streamlit default */
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    header {visibility:hidden;}

    /* Main width */
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
    }

    /* Background */
    .stApp {
        background: #ffffff;
    }

    /* Title */
    .title {
        text-align:center;
        font-size:32px;
        font-weight:700;
        margin-bottom:5px;
    }

    .subtitle {
        text-align:center;
        color:#6b7280;
        margin-bottom:30px;
    }


    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: transparent;
        border:none;
        padding:10px 0px;
    }


    /* User message */
    [data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background:#f3f4f6;
        border-radius:12px;
        padding:12px;
    }


    /* Input box */
    [data-testid="stChatInput"] {
        border-radius:20px;
        border:1px solid #ddd;
    }


    /* Upload */
    [data-testid="stFileUploader"] {
        border:1px dashed #ccc;
        padding:15px;
        border-radius:12px;
        background:#fafafa;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# ---------- HEADER ----------

st.markdown(
    """
    <div class="title">
        📄 CVChat
    </div>

    <div class="subtitle">
        Analyze resumes using AI. Ask about skills, experience, projects, and suitability.
    </div>
    """,
    unsafe_allow_html=True
)



# Session memory
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:

    st.title("📂 Resume")

    uploaded_file = st.file_uploader(
        "Upload CV",
        type=["pdf"]
    )


    if st.session_state.qa_chain:

        st.success("CV Ready")

        if st.button("New CV"):
            st.session_state.qa_chain = None
            st.session_state.messages = []
            st.rerun()


# Process CV only once
if uploaded_file and st.session_state.qa_chain is None:

    file_path = save_uploaded_file(uploaded_file)

    with st.spinner("Analyzing resume..."):

        chunks = load_pdf(file_path)

        embeddings = embedding_model()

        vectorstore = create_vectorstore(
            chunks,
            embeddings
        )

        retriever = create_retriever(
            vectorstore,
            k=5
        )

        llm = get_llm()

        st.session_state.qa_chain = create_rag_chain(
            retriever,
            llm
        )

    st.toast("Resume processed successfully!")


# Display conversation history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# Chat
if st.session_state.qa_chain:

    question = st.chat_input(
        "Ask about experience, skills, projects..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)


        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = st.session_state.qa_chain.invoke(
                    {
                        "input": question
                    }
                )

                answer = response["answer"]

                st.write(answer)


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )