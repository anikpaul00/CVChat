from langchain_community.vectorstores import FAISS

def create_vectorstore(chunks, embeddings):
    
    vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
    )
    
    return vectorstore


def create_retriever(vectorstore, k):
    
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )
    
    return retriever
