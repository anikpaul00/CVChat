from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


def create_rag_chain(retriever, llm):
    
    system_prompt = """
        You are an expert CV/resume analyst.

        RULES:
        1. Grounding: Use ONLY the provided Context. Do not speculate.
        2. Missing Info: If missing, reply strictly: "The requested information is unavailable in the provided context."
        3. Single Sentence: If the answer fits in one short sentence, output ONLY that sentence (omit bullets, rating, feedback).
        4. Skill Gap: Provide skill-gap analysis ONLY when explicitly requested by the user.
        
        DEFAULT FORMAT:
        - Summary: 1–2 concise sentences.
        - Details: 2–4 factual bullet points.
        - Rating: Scale of 1–10 (ONLY if explicitly requested).
        - Feedback: 1 concise line (ONLY if explicitly requested).

        Context:
        {context}
        """
    
    prompt = ChatPromptTemplate.from_messages(
        [ ("system", system_prompt), 
          ("human", "{input}"), 
        ] 
        )
    
    combine_docs_chain = create_stuff_documents_chain(
        llm,
        prompt
    )
    
    qa_chain = create_retrieval_chain(
        retriever,
        combine_docs_chain
    )
    
    return qa_chain
