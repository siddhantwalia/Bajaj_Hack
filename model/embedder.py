from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List
import os
from nomic import embed,login
# from nomic.atlas import AtlasProject
import os
from dotenv import load_dotenv


load_dotenv()

# class NomicEmbeddings(Embeddings):
#     def __init__(self):
#         super(NomicEmbeddings, self).__init__()
#         api_key = os.getenv("NOMIC_TOKEN")
#         if not api_key:
#             raise ValueError("NOMIC_API_KEY not found in environment variables.")
#         os.environ["NOMIC_TOKEN"] =api_key
#         login(api_key) 

#     def embed_documents(self, texts: list[str]) -> list[list[float]]:
#         result = embed.text(
#             texts=texts,
#             model="nomic-embed-text-v1.5",
#             task_type="search_document"
#         )
#         return result["embeddings"]

#     def embed_query(self, text: str) -> list[float]:
#         result = embed.text(
#             texts=[text],
#             model="nomic-embed-text-v1.5",
#             task_type="search_query"
#         )
#         return result["embeddings"][0]


class HuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        super().__init__()
        os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache" 
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()
