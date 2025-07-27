from langchain.embeddings.base import Embeddings
from nomic import embed
import os
from dotenv import load_dotenv

load_dotenv()

class NomicEmbeddings(Embeddings):
    def __init__(self):
        super(NomicEmbeddings, self).__init__()
        api_key = os.getenv("EMBEDDING_API_KEY")
        if not api_key:
            raise ValueError("NOMIC_API_KEY not found in environment variables.")
        os.environ["NOMIC_API_KEY"] = api_key  

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = embed.text(
            texts=texts,
            model="nomic-embed-text-v1.5",
            task_type="search_document"
        )
        return result["embeddings"]

    def embed_query(self, text: str) -> list[float]:
        result = embed.text(
            texts=[text],
            model="nomic-embed-text-v1.5",
            task_type="search_query"
        )
        return result["embeddings"][0]
