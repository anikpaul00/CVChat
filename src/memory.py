from langchain_groq import ChatGroq
import streamlit as st

summarizer = ChatGroq(
        model="llama-3.1-8b-instant", # can also use llama-3.1-8b-instant, openai/gpt-oss-20b, qwen/qwen3.6-27b
        max_completion_tokens=600,
        temperature=0.4,
        api_key=st.secrets["GROQ_API_KEY"]
    )


def update_chat_summary(history, summarizer):

    prompt = f"""
    Summarize this conversation in 2-3 sentences.

    Keep only information needed for future questions.
    Do not add new facts.

    Conversation:
    {history}
    """

    response = summarizer.invoke(prompt)

    return response.content
