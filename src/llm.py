from langchain_groq import ChatGroq
import streamlit as st

def get_llm():
    
    llm = ChatGroq(
        model="qwen/qwen3.6-27b", # can also use llama-3.1-8b-instant, openai/gpt-oss-20b
        reasoning_format="hidden",
        # reasoning_effort="none", # can degrade performance
        temperature=0.1,
        api_key=st.secrets["GROQ_API_KEY"]
    )
    
    return llm



