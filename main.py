import re
import asyncio
import logging
import httpx
import time
from collections import OrderedDict
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from model import Prompt, llm, NomicEmbeddings, rewrite_llm
from utils import parse_document_from_url, split_documents

# ---------------------- CONFIG / TUNABLES ---------------------- #
load_dotenv()
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Embedding batching & concurrency (tune based on your API limits)
BATCH_SIZE = 128                 # number of chunks per embedding request
MAX_CONCURRENT_EMBED_CALLS = 3   # concurrency of simultaneous embedding API calls
EMBED_ROG_RETRY_MAX = 3          # retry attempts for 429s or transient failures
EMBED_RETRY_BACKOFF_BASE = 0.6   # exponential backoff base (seconds)

# HTTP fetching limits
HTTP_MAX_CONNECTIONS = 20
HTTP_TIMEOUT = 10  # seconds
HTTP_RETRY_MAX = 3
HTTP_RETRY_BACKOFF_BASE = 0.6

# Cache limits to prevent memory leaks
MAX_CACHE_SIZE = 100

# cache structures
faiss_cache: Dict[str, FAISS] = OrderedDict()  # doc_key -> faiss_db (simplified as requested)
embedding_cache: Dict[int, List[float]] = OrderedDict()  # hash(text) -> embedding (to avoid re-embedding duplicates)

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
    """Rewrite question for better retrieval (kept optional)."""
    prompt_template_str = """
You are an expert document analyst with strong reasoning skills.

Your task is to rewrite the user question into a precise, keyword-rich query to retrieve correct context from the document.

Use intelligence to smartly identify and emphasize key concepts such as "favorite city", "landmarks", "flight numbers", and any city names.

Remove irrelevant details and keep only terms that will help retrieve the exact info about the user's favorite city and related flight details.

Document Chunk:
{first_doc_chunk}

Original Question:
{original_question}

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
        logger.info(f"Rewritten question: {rewritten.content.strip()}")
        return rewritten.content.strip() if hasattr(rewritten, "content") else str(rewritten).strip()
    except Exception as e:
        logger.warning(f"Rewrite error, using original question: {e}")
        return original_question

# ----- Robust HTTP fetch for URLs (concurrent, limited, with retries) -----
async def fetch_url(client: httpx.AsyncClient, url: str, auth_token: str = None) -> str:
    """Fetch URL content (with optional Authorization) using provided client, with retries."""
    headers = {"Authorization": auth_token} if auth_token else {}
    last_exc = None
    for attempt in range(HTTP_RETRY_MAX):
        try:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.text.strip()
        except httpx.HTTPStatusError as e:
            last_exc = e
            if e.response.status_code == 429:
                backoff = HTTP_RETRY_BACKOFF_BASE * (2 ** attempt) + (attempt * 0.1)
                logger.warning(f"Rate-limited fetching {url} (attempt {attempt+1}/{HTTP_RETRY_MAX}). Backing off {backoff:.2f}s.")
                await asyncio.sleep(backoff)
                continue
            raise
        except Exception as e:
            last_exc = e
            logger.debug(f"Fetch URL error for {url}: {e}")
            backoff = HTTP_RETRY_BACKOFF_BASE * (2 ** attempt) + (attempt * 0.1)
            await asyncio.sleep(backoff)
    return f"<ERROR fetching {url}: {last_exc}>"

async def enrich_document_with_urls_fast(text_chunks: List[str], auth_token: str = None, max_conn: int = HTTP_MAX_CONNECTIONS) -> tuple[List[str], list[str]]:
    """Find all unique URLs, fetch them concurrently, append their content once per URL. Returns enriched chunks and found_urls."""
    url_pattern = r"https?://[^\s)>\]]+"
    found_urls = list(dict.fromkeys(re.findall(url_pattern, " ".join(text_chunks))))  # Unique in order

    if not found_urls:
        return text_chunks, found_urls

    logger.info(f"Found {len(found_urls)} unique URLs in document. Fetching concurrently...")
    limits = httpx.Limits(max_connections=max_conn)
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, limits=limits) as client:
        tasks = [fetch_url(client, u, auth_token) for u in found_urls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Build map url -> content (string)
    url_content_map = {}
    for u, r in zip(found_urls, responses):
        if isinstance(r, Exception):
            url_content_map[u] = f"<ERROR fetching {u}: {r}>"
        else:
            url_content_map[u] = r

    # Append fetched content only once per URL, distributing to first chunk where it appears
    enriched_chunks = text_chunks[:]
    appended_urls = set()
    for i, chunk in enumerate(enriched_chunks):
        for u in found_urls:
            if u in chunk and u not in appended_urls:
                enriched_chunks[i] += f"\n\n[Fetched from {u}]:\n{url_content_map[u]}"
                appended_urls.add(u)
    return enriched_chunks, found_urls

# ----- Embedding helpers: dedupe + concurrency + retry -----
def _hash_text_to_int(text: str) -> int:
    """Stable int hash for mapping into embedding_cache."""
    import hashlib
    return int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest()[:8], "big")

async def _embed_with_retries(embedding_model, texts: List[str], retries=EMBED_ROG_RETRY_MAX, backoff_base=EMBED_RETRY_BACKOFF_BASE):
    """Call embedding_model.embed_documents in a thread with retries on transient errors."""
    last_exc = None
    for attempt in range(retries):
        try:
            embeds = await asyncio.to_thread(embedding_model.embed_documents, texts)
            return embeds
        except Exception as e:
            last_exc = e
            s = str(e).lower()
            if "429" in s or "rate" in s or "too many" in s or "try again" in s:
                backoff = backoff_base * (2 ** attempt) + (attempt * 0.1)
                logger.warning(f"Embedding call rate-limited or transient error (attempt {attempt+1}/{retries}). Backing off {backoff:.2f}s. err={e}")
                await asyncio.sleep(backoff)
                continue
            else:
                raise
    logger.error(f"Embedding failed after {retries} attempts: {last_exc}")
    raise last_exc

async def build_faiss_concurrent(docs: List[Document], embedding_model, batch_size: int = BATCH_SIZE, max_concurrent: int = MAX_CONCURRENT_EMBED_CALLS) -> FAISS:
    """
    Embed documents in parallel batches with concurrency control and deduplication.
    Returns a langchain FAISS vectorstore built from (text, embedding) pairs.
    """
    # 1. Deduplicate texts while preserving order
    unique_texts = []
    text_to_original_indices = {}
    for i, d in enumerate(docs):
        txt = d.page_content
        if txt not in text_to_original_indices:
            text_to_original_indices[txt] = []
            unique_texts.append(txt)
        text_to_original_indices[txt].append(i)

    logger.info(f"Embedding {len(unique_texts)} unique chunks (from {len(docs)} total chunks). Batch size={batch_size}, concurrency={max_concurrent}")

    # 2. Check embedding_cache
    texts_to_embed = []
    embed_positions = []
    all_embeddings = [None] * len(unique_texts)
    for idx, txt in enumerate(unique_texts):
        key = _hash_text_to_int(txt)
        if key in embedding_cache:
            all_embeddings[idx] = embedding_cache[key]
        else:
            embed_positions.append(idx)
            texts_to_embed.append(txt)

    if texts_to_embed:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def embed_batches_worker():
            tasks = []
            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i:i + batch_size]

                async def _embed_batch(b):
                    async with semaphore:
                        return await _embed_with_retries(embedding_model, b)

                tasks.append(asyncio.create_task(_embed_batch(batch)))

            batch_results = await asyncio.gather(*tasks)
            flat_embeds = [emb for br in batch_results for emb in br]

            for pos_idx, emb in zip(embed_positions, flat_embeds):
                all_embeddings[pos_idx] = emb
                key = _hash_text_to_int(unique_texts[pos_idx])
                embedding_cache[key] = emb
                if len(embedding_cache) > MAX_CACHE_SIZE:
                    embedding_cache.pop(next(iter(embedding_cache)))  # Evict oldest

        await embed_batches_worker()

    # Sanity check
    missing = [i for i, e in enumerate(all_embeddings) if e is None]
    if missing:
        raise RuntimeError(f"Missing embeddings for indices {missing}")

    # 3. Build FAISS
    pairs = list(zip(unique_texts, all_embeddings))
    try:
        vs = FAISS.from_embeddings(pairs, embedding_model)
    except Exception as e:
        logger.warning(f"FAISS.from_embeddings failed: {e}. Using manual add_embeddings fallback.")
        vs = FAISS(embedding_model)
        vs.add_embeddings(pairs)  # Use existing embeddings

    return vs

# ---------------------- API ---------------------- #
@app.get("/")
async def home():
    return {"home": "This is our unified API endpoint"}

@app.post("/hackrx/run")
async def run_query(req: QueryRequest, Authorization: str = Header(default=None)):
    start = time.time()

    if not Authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    # Hash doc_key with auth for security (prevent cross-user cache pollution)
    import hashlib
    doc_key = hashlib.sha256((req.documents + Authorization).encode()).hexdigest()

    if doc_key in faiss_cache:
        db = faiss_cache[doc_key]
        logger.info("Using cached FAISS retriever")
    else:
        # 1. Parse document
        try:
            parsed_docs = await parse_document_from_url(req.documents)
        except Exception as e:
            logger.error(f"Error parsing document: {e}")
            raise HTTPException(status_code=400, detail=f"Error parsing document: {e}")

        # 2. Split into chunks
        try:
            chunks = split_documents(parsed_docs)
        except Exception as e:
            logger.error(f"Error splitting document: {e}")
            raise HTTPException(status_code=500, detail=f"Error splitting document: {e}")

        text_list = [c.page_content for c in chunks]

        # 3. Enrich URLs concurrently (now returns found_urls for conditional check)
        enriched_text_list, found_urls = await enrich_document_with_urls_fast(text_list, Authorization, max_conn=HTTP_MAX_CONNECTIONS)

        # 4. Build embeddings + FAISS
        try:
            embedding_model = NomicEmbeddings()
            enriched_chunks = [Document(page_content=t) for t in enriched_text_list]
            db = await build_faiss_concurrent(enriched_chunks, embedding_model, batch_size=BATCH_SIZE, max_concurrent=MAX_CONCURRENT_EMBED_CALLS)
        except Exception as e:
            logger.exception("Embedding/Vector store error")
            raise HTTPException(status_code=500, detail=f"Embedding/Vector store error: {e}")

        # 5. Conditional caching: only cache if no HTTP/HTTPS URLs found in the document content
        if len(found_urls) == 0:
            faiss_cache[doc_key] = db
            if len(faiss_cache) > MAX_CACHE_SIZE:
                faiss_cache.pop(next(iter(faiss_cache)))  # Evict oldest
            logger.info("FAISS index built and cached (no URLs in document content)")
        else:
            logger.info("Document content contains HTTP/HTTPS URLs, skipping FAISS cache storage")

    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.3})

    # Process questions in parallel
    async def process_q(question: str):
        try:
            # Optional rewrite (commented to optimize speed; uncomment if needed)
            # first_chunk = ""  # No longer caching enriched list, so fallback to empty
            # rewritten_question = await rewrite_question(question, first_chunk)
            rewritten_question = question

            context_docs = await asyncio.to_thread(retriever.invoke, rewritten_question)
            context = "\n".join([doc.page_content for doc in context_docs]) if context_docs else ""
            context = context[:8000]  # Truncate to avoid token limits

            inputs = {"context": context, "question": question}
            answer = await (Prompt | llm).ainvoke(inputs)
            return clean_output(answer)
        except Exception as e:
            logger.exception(f"Error processing question '{question}': {e}")
            return f"Error: {e}"

    answers = await asyncio.gather(*[process_q(q) for q in req.questions])

    elapsed = time.time() - start
    logger.info(f"Total run_query time: {elapsed:.2f}s")
    return {"answers": answers}
