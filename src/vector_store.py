from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def create_vector_store(docs, embeddings=None):
    """
    Creates a FAISS vector store from a list of documents.
    Uses HuggingFace embeddings (all-MiniLM-L6-v2) by default or the provided embeddings object.
    """
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
    vector_store = FAISS.from_documents(docs, embedding=embeddings)
    return vector_store
