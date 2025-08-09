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

# ===== Helper: detect non-English (non-ASCII) =====
def has_non_ascii(s: str) -> bool:
    return any(ord(c) > 127 for c in s)

# ===== Ask GPT for Plan =====
async def ask_gpt_for_plan(question: str, context_text: str) -> str:
    prompt = f"""
You are an AI assistant that reads provided context/document and makes a precise,
numbered plan to answer the user's question. RETURN ONLY A PLAN (numbered steps).
Do NOT give the final answer in this response.

Rules:
- Respond in the same language as the Question.
- When you need a value from the document, write:
  LOOKUP: <what to find and how to present it>
- When you want to extract with regex, write:
  EXTRACT_REGEX: <regex>
- When you want an HTTP call, write:
  GET https://example.com/path
- Keep each step deterministic.

Context:
{context_text}

Question:
{question}

Plan:
"""
    result = await llm.ainvoke(prompt)
    return result.content if hasattr(result, "content") else str(result)

# ===== Execute Plan =====
async def execute_plan(plan: str, context_text: str, auth_token: str,question:str) -> str:
    executed_plan = plan

    async with httpx.AsyncClient() as client:
        for line in plan.splitlines():
            line_stripped = line.strip()

            # GET request
            m_get = re.search(r"GET\s+(https?://\S+)", line_stripped)
            if m_get:
                url = m_get.group(1).strip()
                url = re.sub(r"[\`\'\"\,\.\)\;]+$", "", url)
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
                    pattern = re.compile(regex_text)
                    matches = pattern.findall(context_text)
                    if not matches:
                        regex_result = "<NO MATCH>"
                    else:
                        formatted = []
                        for m in matches:
                            if isinstance(m, tuple):
                                formatted.append(" | ".join(m))
                            else:
                                formatted.append(str(m))
                        regex_result = "\n".join(formatted[:20])
                except re.error as re_err:
                    regex_result = f"<BAD REGEX: {re_err}>"
                executed_plan = executed_plan.replace(line, f"{line} → {regex_result}")
                continue

    # Ask GPT for final answer
# Ask GPT for final answer
    final_prompt = f"""
    You are a helpful assistant.
    Always respond in the SAME LANGUAGE as the original question.
    Keep the answer concise — no more than 2–3 sentences.
    Ensure you include all key details from the executed steps and context.
    question:
    {question}
    
    Context:
    {context_text}

    Executed Steps with Results:
    {executed_plan}

    Final Answer:
    """
    result = await llm.ainvoke(final_prompt)
    return result.content if hasattr(result, "content") else str(result)


# ===== Lookup Helper =====
def perform_lookup(instruction: str, document_text: str) -> str:
    doc = document_text
    quoted = re.findall(r'"([^"]+)"', instruction)
    if quoted:
        results = []
        for q in quoted:
            for line in doc.splitlines():
                if q.lower() in line.lower():
                    results.append(line.strip())
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

# ===== Main processing =====
async def process_question_rag_agent(question: str, retriever, texts, full_doc_text: str, Authorization: str):
    logger.info(f"Processing question: {question}")

    # Retrieve chunks
    rewritten = await rewrite_llm.ainvoke(question)
    query = rewritten.content if hasattr(rewritten, "content") else str(rewritten)
    context_docs = await asyncio.to_thread(retriever.invoke, query)
    context_text = "\n".join([doc.page_content for doc in context_docs]) if context_docs else ""

    # If non-English, use full doc for better context
    if has_non_ascii(question):
        context_for_plan = full_doc_text
    else:
        context_for_plan = context_text or full_doc_text

    # Plan → Execute → Answer
    plan = await ask_gpt_for_plan(question, context_for_plan)
    return await execute_plan(plan, context_for_plan, Authorization,question)

# ===== API Route =====
@app.post("/hackrx/run")
async def run_query(req: QueryRequest, Authorization: str = Header(default=None)):
    if not Authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    # Cache FAISS retriever
    if req.documents in faiss_cache:
        db, texts, full_doc_text = faiss_cache[req.documents]
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 8, "lambda_mult": 0.8})
        logger.info("Using cached retriever")
    else:
        parsed_doc = await parse_document_from_url(req.documents)
        chunks = split_documents(parsed_doc)
        texts = [chunk.page_content for chunk in chunks]
        full_doc_text = "\n".join([chunk.page_content for chunk in chunks])
        embedding_model = NomicEmbeddings()
        db = FAISS.from_documents(chunks, embedding_model)
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 8, "lambda_mult": 0.8})
        faiss_cache[req.documents] = (db, texts, full_doc_text)

    answers = await asyncio.gather(*[
        process_question_rag_agent(q, retriever, texts, full_doc_text, Authorization)
        for q in req.questions
    ])
    return {"answers": answers}
