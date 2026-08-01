from langchain_groq import ChatGroq

def update_chat_summary(history, llm):

    prompt = f"""
    Summarize this conversation in 2-3 sentences.

    Keep only information needed for future questions.
    Do not add new facts.

    Conversation:
    {history}
    """

    response = llm.invoke(prompt)

    return response.content
