import re
import asyncio
import logging
import httpx
import time
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.schema import Document  # ✅ Correct Document import

from model import Prompt, llm, NomicEmbeddings, rewrite_llm
from utils import parse_document_from_url, split_documents

# ---------------------- Setup ---------------------- #
load_dotenv()
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

faiss_cache = {}  # Cache embeddings per doc

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

# ---------------------- Helpers ---------------------- #
def clean_output(answer):
    """Remove <think> tags and excessive spacing."""
    if hasattr(answer, "content"):
        content = answer.content
    else:
        content = str(answer)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return re.sub(r"\n{3,}", "\n\n", content)

async def rewrite_question(original_question: str, first_doc_chunk: str = "") -> str:
    """Rewrite question for better retrieval."""
    prompt_template_str = """
    You are an expert query rewriter for a document retrieval system.
    Rewrite the original question into a concise, keyword-rich query with synonyms and related terms.
    Avoid unrelated details. Do not change the meaning.

    First Chunk of Document: {first_doc_chunk}

    Original Question: {original_question}

    Rewritten Query:
    """
    prompt_text = PromptTemplate(
        input_variables=["original_question", "first_doc_chunk"],
        template=prompt_template_str
    )
    try:
        formatted_prompt = await prompt_text.ainvoke({
            "original_question": original_question,
            "first_doc_chunk": first_doc_chunk
        })
        rewritten = await rewrite_llm.ainvoke(formatted_prompt)
        return rewritten.content.strip() if hasattr(rewritten, "content") else str(rewritten).strip()
    except Exception as e:
        logger.error(f"Rewrite error: {e}")
        return original_question

async def fetch_url(url: str, auth_token: str = None) -> str:
    """Fetch URL content (with optional Authorization)."""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            headers = {"Authorization": auth_token} if auth_token else {}
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.text.strip()
        except Exception as e:
            return f"<HTTP ERROR fetching {url}: {e}>"

async def enrich_document_with_urls(text_chunks: List[str], auth_token: str = None) -> List[str]:
    """Find and fetch all URLs, append responses into text."""
    url_pattern = r"https?://[^\s)>\]]+"
    urls = set()
    for chunk in text_chunks:
        urls.update(re.findall(url_pattern, chunk))

    if not urls:
        return text_chunks

    logger.info(f"Found {len(urls)} URLs in document. Fetching...")
    fetch_results = await asyncio.gather(*[fetch_url(url, auth_token) for url in urls])

    # Append fetched content after each URL occurrence
    enriched_chunks = []
    for chunk in text_chunks:
        for url, content in zip(urls, fetch_results):
            if url in chunk:
                chunk += f"\n\n[Fetched from {url}]:\n{content}"
        enriched_chunks.append(chunk)

    return enriched_chunks

# ---------------------- API ---------------------- #
@app.get("/")
async def home():
    return {"home": "This is our unified API endpoint"}

@app.post("/hackrx/run")
async def run_query(req: QueryRequest, Authorization: str = Header(default=None)):
    start = time.time()

    if not Authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    doc_key = req.documents
    if doc_key in faiss_cache:
        db, texts = faiss_cache[doc_key]
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.8})
        logger.info("Using cached FAISS retriever")
    else:
        # 1. Parse document
        try:
            parsed_docs = await parse_document_from_url(req.documents)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error parsing document: {e}")

        chunks = split_documents(parsed_docs)
        text_list = [chunk.page_content for chunk in chunks]

        # 2. Enrich doc with fetched URLs
        enriched_text_list = await enrich_document_with_urls(text_list, Authorization)

        # 3. Build embeddings on enriched text
        try:
            embedding_model = NomicEmbeddings()
            enriched_chunks = [Document(page_content=t) for t in enriched_text_list]
            db = FAISS.from_documents(enriched_chunks, embedding_model)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding/Vector store error: {e}")

        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.8})
        faiss_cache[doc_key] = (db, enriched_text_list)

    # 4. Process each question
    async def process_question(question: str):
        try:
            first_chunk = faiss_cache[doc_key][1][0] if faiss_cache[doc_key][1] else ""
            rewritten_question = await rewrite_question(question, first_chunk)
            context_docs = await asyncio.to_thread(retriever.invoke, rewritten_question)
            context = "\n".join([doc.page_content for doc in context_docs]) if context_docs else ""
            inputs = {"context": context, "question": question}
            answer = await (Prompt | llm).ainvoke(inputs)
            return clean_output(answer)
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            return f"Error: {e}"

    answers = await asyncio.gather(*[process_question(q) for q in req.questions])
    logger.info(f"Total time: {time.time() - start:.2f}s")
    return {"answers": answers}