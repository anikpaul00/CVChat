from langchain_huggingface import HuggingFaceEmbeddings

def embedding_model(model_name="BAAI/bge-base-en-v1.5"):
    
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name,
        encode_kwargs={"normalize_embeddings": True}
    )
    
    return embeddings
    
    