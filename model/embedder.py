import os
from langchain.embeddings.base import Embeddings
# from langchain_community.embeddings import CohereEmbeddings
# from langchain_cohere import CohereEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer
from typing import List
from nomic import embed,login
# from nomic.atlas import AtlasProject
from dotenv import load_dotenv
# from more_itertools import chunked
from openai import OpenAI

load_dotenv()

class NomicEmbeddings(Embeddings):
    def __init__(self):
        super(NomicEmbeddings, self).__init__()
        api_key = os.getenv("NOMIC_TOKEN")
        if not api_key:
            raise ValueError("NOMIC_API_KEY not found in environment variables.")
        os.environ["NOMIC_TOKEN"] =api_key
        login(api_key) 


    # def embed_documents(self, texts: list[str]) -> list[list[float]]:
    #     all_embeddings = []
    #     # for batch in chunked(texts,512):  # Adjust based on limits
    #         result = embed.text(
    #             texts=,
    #             model="nomic-embed-text-v1.5",
    #             task_type="search_document"
    #         )
    #         # all_embeddings.extend(result["embeddings"])
    #     return all_embeddings

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


# class CustomCohereEmbeddings(Embeddings):
#     def __init__(self):
#         super().__init__()
#         api_key = os.getenv("CO_API_KEY")
#         if not api_key:
#             raise ValueError("CO_API_KEY not found in environment variables.")

#         self.model = CohereEmbeddings(
#             cohere_api_key=api_key,
#             model="embed-english-v3.0",
#             user_agent='langchain'
#         )

#     def embed_documents(self, texts: list[str]) -> list[list[float]]:
#         return self.model.embed_documents(texts)

#     def embed_query(self, text: str) -> list[float]:
#         return self.model.embed_query(text)

class HuggingFaceEmbed():
    def __init__(self):
        super().__init__()
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache" 
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings(text, convert_to_numpy=True).tolist()



class OpenAITextEmbedding3Small(Embeddings):
    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = 1536):
        super(OpenAITextEmbedding3Small, self).__init__()
        self.model = model
        self.dimensions = dimensions
        self.api_key = os.getenv("OPEN_AI_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        self.client = OpenAI(api_key=self.api_key)

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        response = self.client.embeddings.create(
            input=text,
            model=self.model,
            dimensions=self.dimensions  # optional, defaults to full dimensionality (1536 for this model)
        )
        return response.data[0].embedding