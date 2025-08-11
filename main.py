import re
import asyncio
import logging
import httpx
import time
import math
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.schema import Document  # ✅ Correct Document import

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
EMBED_RETRY_MAX = 3              # retry attempts for 429s or transient failures
EMBED_RETRY_BACKOFF_BASE = 0.6   # exponential backoff base (seconds)

# HTTP fetching limits
HTTP_MAX_CONNECTIONS = 20
HTTP_TIMEOUT = 10  # seconds

# cache structures
faiss_cache: Dict[str, Tuple[FAISS, List[str]]] = {}   # doc_key -> (faiss_db, enriched_text_list)
embedding_cache: Dict[int, List[float]] = {}          # hash(text) -> embedding (to avoid re-embedding duplicates)

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
        logger.warning(f"Rewrite error, using original question: {e}")
        return original_question

# ----- Robust HTTP fetch for URLs (concurrent, limited) -----
async def fetch_url(client: httpx.AsyncClient, url: str, auth_token: str = None) -> str:
    """Fetch URL content (with optional Authorization) using provided client."""
    headers = {"Authorization": auth_token} if auth_token else {}
    try:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.text.strip()
    except Exception as e:
        logger.debug(f"Fetch URL error for {url}: {e}")
        return f"<HTTP ERROR fetching {url}: {e}>"

async def enrich_document_with_urls_fast(text_chunks: List[str], auth_token: str = None, max_conn: int = HTTP_MAX_CONNECTIONS) -> List[str]:
    """Find all unique URLs, fetch them concurrently, append their content in one pass."""
    url_pattern = r"https?://[^\s)>\]]+"
    # preserve insertion order for deterministic mapping
    found_urls = []
    seen = set()
    for chunk in text_chunks:
        for u in re.findall(url_pattern, chunk):
            if u not in seen:
                seen.add(u)
                found_urls.append(u)

    if not found_urls:
        return text_chunks

    logger.info(f"Found {len(found_urls)} URLs in document. Fetching concurrently...")
    limits = httpx.Limits(max_connections=max_conn)
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, limits=limits) as client:
        tasks = [fetch_url(client, u, auth_token) for u in found_urls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    # build map url -> content (string)
    url_content_map = {}
    for u, r in zip(found_urls, responses):
        if isinstance(r, Exception):
            url_content_map[u] = f"<ERROR fetching {u}: {r}>"
        else:
            url_content_map[u] = r

    # append fetched content only once per chunk where URL appears
    enriched_chunks = []
    for chunk in text_chunks:
        additions = []
        for u in found_urls:
            if u in chunk:
                additions.append(f"\n\n[Fetched from {u}]:\n{url_content_map[u]}")
        enriched_chunks.append(chunk + "".join(additions))
    return enriched_chunks

# ----- Embedding helpers: dedupe + concurrency + retry -----
def _hash_text_to_int(text: str) -> int:
    """Stable int hash for mapping into embedding_cache."""
    # use Python built-in hash may vary between runs, so use a stable hash:
    return int.from_bytes(__import__("hashlib").sha1(text.encode("utf-8")).digest()[:8], "big")

async def _embed_with_retries(embedding_model, texts: List[str], retries=EMBED_RETRY_MAX, backoff_base=EMBED_RETRY_BACKOFF_BASE):
    """Call embedding_model.embed_documents in a thread with retries on transient errors (e.g., 429)."""
    last_exc = None
    for attempt in range(retries):
        try:
            # run blocking embed call in a thread so it doesn't block event loop
            embeds = await asyncio.to_thread(embedding_model.embed_documents, texts)
            return embeds
        except Exception as e:
            last_exc = e
            # If it's clearly a rate-limit or transient network error, sleep then retry
            # We check for 429 or 'rate' keywords in exception string - adjust per provider
            s = str(e).lower()
            if "429" in s or "rate" in s or "too many" in s or "try again" in s:
                backoff = backoff_base * (2 ** attempt) + (attempt * 0.1)
                logger.warning(f"Embedding call rate-limited or transient error (attempt {attempt+1}/{retries}). Backing off {backoff:.2f}s. err={e}")
                await asyncio.sleep(backoff)
                continue
            else:
                # non-retryable - raise immediately
                raise
    # if we get here, all retries failed
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

    # 2. For unique_texts, check embedding_cache to avoid repeating work
    texts_to_embed = []
    embed_positions = []  # positions in unique_texts that actually need embedding
    for idx, txt in enumerate(unique_texts):
        key = _hash_text_to_int(txt)
        if key in embedding_cache:
            continue
        embed_positions.append(idx)
        texts_to_embed.append(txt)

    # 3. If nothing to embed, gather cached embeddings and build index
    all_embeddings = [None] * len(unique_texts)
    for idx, txt in enumerate(unique_texts):
        key = _hash_text_to_int(txt)
        if key in embedding_cache:
            all_embeddings[idx] = embedding_cache[key]

    if texts_to_embed:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def embed_batches_worker():
            tasks = []
            # create batches of texts_to_embed
            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i:i + batch_size]

                async def _embed_batch(b):
                    async with semaphore:
                        return await _embed_with_retries(embedding_model, b)

                tasks.append(asyncio.create_task(_embed_batch(batch)))

            # run all batches concurrently (bounded by semaphore)
            batch_results = await asyncio.gather(*tasks)

            # flatten batches back into embeddings list and fill into all_embeddings at correct positions
            flat_embeds = []
            for br in batch_results:
                flat_embeds.extend(br)

            # map flat_embeds back to embed_positions order
            for pos_idx, emb in zip(embed_positions, flat_embeds):
                # store embedding and cache
                all_embeddings[pos_idx] = emb
                key = _hash_text_to_int(unique_texts[pos_idx])
                embedding_cache[key] = emb

        await embed_batches_worker()

    # sanity check: all_embeddings should be fully populated
    missing = [i for i, e in enumerate(all_embeddings) if e is None]
    if missing:
        raise RuntimeError(f"Missing embeddings for indices {missing}")

    # 4. Build FAISS via langchain helper - pair texts with embeddings
    pairs = list(zip(unique_texts, all_embeddings))
    try:
        vs = FAISS.from_embeddings(pairs, embedding_model)
    except Exception as e:
        # fallback: attempt to use from_documents (slower if embed function is used)
        logger.warning(f"FAISS.from_embeddings failed: {e}. Trying from_documents fallback.")
        docs_for_store = [Document(page_content=t) for t in unique_texts]
        vs = FAISS.from_documents(docs_for_store, embedding_model)

    # 5. If original docs contained duplicates, we still built index on unique_texts.
    # For retriever working, this is fine because the content exists in index.
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

    doc_key = req.documents  # keep as user provides; consider hashing content for better caching
    if doc_key in faiss_cache:
        db, texts = faiss_cache[doc_key]
        logger.info("Using cached FAISS retriever")
    else:
        # 1. Parse document (might be network) - keep original util
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

        # 3. Enrich URLs concurrently
        enriched_text_list = await enrich_document_with_urls_fast(text_list, Authorization, max_conn=HTTP_MAX_CONNECTIONS)

        # 4. Build embeddings + FAISS concurrently with dedupe and controlled concurrency
        try:
            embedding_model = NomicEmbeddings()
            enriched_chunks = [Document(page_content=t) for t in enriched_text_list]
            db = await build_faiss_concurrent(enriched_chunks, embedding_model, batch_size=BATCH_SIZE, max_concurrent=MAX_CONCURRENT_EMBED_CALLS)
        except Exception as e:
            logger.exception("Embedding/Vector store error")
            raise HTTPException(status_code=500, detail=f"Embedding/Vector store error: {e}")

        # 5. Cache the results
        faiss_cache[doc_key] = (db, enriched_text_list)
        logger.info("FAISS index built and cached")

    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.8})

    # Process questions in parallel (each question executes retrieval + single LLM call)
    async def process_q(question: str):
        try:
            # optional: use the first chunk to help rewrite (keeps rewrite step, but you can skip it to save time)
            first_chunk = faiss_cache[doc_key][1][0] if faiss_cache.get(doc_key) and faiss_cache[doc_key][1] else ""
            # you may choose to comment out rewrite_question to save ~1-2s per query
            # rewritten_question = await rewrite_question(question, first_chunk)
            rewritten_question = question

            # retrieval runs in a thread if retriever.invoke is blocking
            context_docs = await asyncio.to_thread(retriever.invoke, rewritten_question)
            context = "\n".join([doc.page_content for doc in context_docs]) if context_docs else ""

            # limit context length if your LLM has token limits (optional)
            # answer with your existing (Prompt | llm).ainvoke pattern
            inputs = {"context": context, "question": question}
            answer = await (Prompt | llm).ainvoke(inputs)
            return clean_output(answer)
        except Exception as e:
            logger.exception(f"Error processing question '{question}': {e}")
            return f"Error: {e}"

    # schedule all question tasks concurrently and wait
    answers = await asyncio.gather(*[process_q(q) for q in req.questions])

    elapsed = time.time() - start
    logger.info(f"Total run_query time: {elapsed:.2f}s")
    return {"answers": answers, "took_seconds": round(elapsed, 2)}
