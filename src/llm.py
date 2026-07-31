from langchain_groq import ChatGroq
import streamlit as st

def get_llm():
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        api_key=st.secrets["GROQ_API_KEY"]
    )
    
    return llm



