import re
import asyncio
import logging
import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from model import llm, NomicEmbeddings, rewrite_llm
from utils import parse_document_from_url, split_documents

load_dotenv()
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

faiss_cache = {}

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

def has_non_ascii(s: str) -> bool:
    return any(ord(c) > 127 for c in s)

async def ask_gpt_for_plan(question: str, context_text: str) -> str:
    prompt = f"""
You are an AI assistant that reads provided context/document and makes a precise,
numbered plan to answer the user's question. RETURN ONLY A PLAN (numbered steps).
Do NOT give the final answer in this response.

Rules:
- Respond in the **ENGLISH** only.
- When you need a value from the document, write:
  LOOKUP: <what to find and how to present it>
- When you want to extract with regex, write:
  EXTRACT_REGEX: <regex>
- When you want an HTTP call, write:
  GET https://example.com/path
- Keep each step deterministic.
- Be concise and clear in your responses

Context:
{context_text}

Question:
{question}

Plan:
"""
    result = await llm.ainvoke(prompt)
    return result.content if hasattr(result, "content") else str(result)

# ===== Execute Plan =====
async def execute_plan(plan: str, context_text: str, auth_token: str, question: str) -> str:
    executed_plan = plan

    async with httpx.AsyncClient() as client:
        for line in plan.splitlines():
            line_stripped = line.strip()

            # GET request
            m_get = re.search(r"GET\s+(https?://\S+)", line_stripped)
            if m_get:
                url = re.sub(r"[\`\'\"\,\.\)\;]+$", "", m_get.group(1).strip())
                try:
                    resp = await client.get(url, headers={"Authorization": auth_token})
                    resp.raise_for_status()
                    value = resp.text.strip().replace('"', '')
                except Exception as e:
                    value = f"<HTTP ERROR: {type(e).__name__}: {e}>"
                executed_plan = executed_plan.replace(line, f"{line} → {value}")
                continue

            # LOOKUP
            if line_stripped.upper().startswith("LOOKUP:"):
                instruction = line_stripped[len("LOOKUP:"):].strip()
                lookup_result = perform_lookup(instruction, context_text)
                executed_plan = executed_plan.replace(line, f"{line} → {lookup_result}")
                continue

            # EXTRACT_REGEX
            if line_stripped.upper().startswith("EXTRACT_REGEX:"):
                regex_text = line_stripped[len("EXTRACT_REGEX:"):].strip()
                try:
                    matches = re.findall(regex_text, context_text)
                    if not matches:
                        regex_result = "<NO MATCH>"
                    else:
                        formatted = [" | ".join(m) if isinstance(m, tuple) else str(m) for m in matches]
                        regex_result = "\n".join(formatted[:20])
                except re.error as re_err:
                    regex_result = f"<BAD REGEX: {re_err}>"
                executed_plan = executed_plan.replace(line, f"{line} → {regex_result}")
                continue

    # Final answer
    final_prompt = f"""
    You are a helpful assistant.

    First, detect the language of the question **from the question text only** — do NOT infer it from the document.
    Then, answer in that same language.

    Keep the answer concise (max 2–3 sentences) and include the details from the executed steps and context.
    No need to tell the language we are using in the final answer
    Question:
    {question}

    Context:
    {context_text}

    Executed Steps with Results:
    {executed_plan}

    Final Answer:
    """

    result = await llm.ainvoke(final_prompt)
    return result.content if hasattr(result, "content") else str(result)


def perform_lookup(instruction: str, document_text: str) -> str:
    doc = document_text
    quoted = re.findall(r'"([^"]+)"', instruction)
    if quoted:
        results = []
        for q in quoted:
            results.extend([line.strip() for line in doc.splitlines() if q.lower() in line.lower()])
        if results:
            return "\n".join(results[:20])

    keywords = re.findall(r"\b[a-zA-Z0-9%]{2,}\b", instruction)
    keywords = [k for k in keywords if k.lower() not in ("find", "table", "row", "list", "products", "where", "which")]
    results = []
    if keywords:
        for line in doc.splitlines():
            if all(kw.lower() in line.lower() for kw in keywords[:3]):
                results.append(line.strip())
        if results:
            return "\n".join(results[:30])

    return "<LOOKUP FAILED>"

async def process_question_rag_agent(question: str, retriever, texts, full_doc_text: str, Authorization: str):
    logger.info(f"Processing question: {question}")

    rewritten = await rewrite_llm.ainvoke(question)
    query = rewritten.content if hasattr(rewritten, "content") else str(rewritten)

    retrieved_docs = await asyncio.to_thread(retriever.invoke, query)
    retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else ""
    
    context_for_plan = (retrieved_text + "\n" + full_doc_text).strip()

    plan = await ask_gpt_for_plan(question, context_for_plan)
    return await execute_plan(plan, context_for_plan, Authorization, question)


@app.post("/hackrx/run")
async def run_query(req: QueryRequest, Authorization: str = Header(default=None)):
    if not Authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    # Cache FAISS retriever 
    logger.info(req.documents)
    logger.info(req.questions)
    if req.documents in faiss_cache:
        db, texts, full_doc_text = faiss_cache[req.documents]
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 8, "lambda_mult": 0.8})
        logger.info("Using cached retriever")
    else:
        parsed_doc = await parse_document_from_url(req.documents)
        chunks = split_documents(parsed_doc)
        texts = [chunk.page_content for chunk in chunks]
        full_doc_text = "\n".join(texts)
        embedding_model = NomicEmbeddings()
        db = FAISS.from_documents(chunks, embedding_model)
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 8, "lambda_mult": 0.8})
        faiss_cache[req.documents] = (db, texts, full_doc_text)

    answers = await asyncio.gather(*[
        process_question_rag_agent(q, retriever, texts, full_doc_text, Authorization)
        for q in req.questions
    ])
    return {"answers": answers}
