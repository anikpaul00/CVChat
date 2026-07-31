from langchain_groq import ChatGroq
import streamlit as st

def get_llm():
    
    llm = ChatGroq(
        model="openai/gpt-oss-20b", # can also use llama-3.1-8b-instant, openai/gpt-oss-20b, qwen/qwen3.6-27b
        max_completion_tokens=1024,
        reasoning_format="hidden",
        reasoning_effort="low", # can degrade performance
        temperature=0.2,
        stream=True,
        api_key=st.secrets["GROQ_API_KEY"]
    )
    
    return llm



