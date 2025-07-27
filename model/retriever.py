from langchain_community.vectorstores import FAISS
import os

def get_retriever(embeddings):
    vectorstore_path = os.path.join("vectorstores", embeddings)
    
    if not os.path.exists(vectorstore_path):
        raise FileNotFoundError(f"Vector store at {vectorstore_path} not found. Make sure it's built and saved correctly.")

    db = FAISS.load_local(
        vectorstore_path,
        embeddings
    )
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,
    }
    )
    return retriever