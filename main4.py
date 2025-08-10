import asyncio
import logging
import re
from typing import List, Dict
import httpx

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from model import llm, NomicEmbeddings, rewrite_llm
from utils import parse_document_from_url, split_documents

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()


faiss_cache = {}
facts_cache = {}


class HackRxRunRequest(BaseModel):
    documents: str
    questions: List[str]


async def extract_core_facts(retriever) -> Dict[str, str]:
    """Extract important info from doc via retrieval."""
    core_queries = [
        "Summarize the main topic and purpose of the document.",
        "List all important dates, deadlines, and time periods.",
        "List all key entities, names, and organizations mentioned.",
        "List all numeric values and their meanings.",
        "Summarize any important procedures, steps, or instructions."
    ]

    facts = {}
    for cq in core_queries:
        retrieved_docs = await asyncio.to_thread(retriever.invoke, cq)
        retrieved_text = "\n".join([d.page_content for d in retrieved_docs]) if retrieved_docs else ""

        prompt = f"""
        Extract the most important information for the following request:
        Request: {cq}
        Context:
        {retrieved_text}
        Answer concisely, no filler text:
        """
        result = await llm.ainvoke(prompt)
        facts[cq] = result.content if hasattr(result, "content") else str(result)

    return facts


async def ask_gpt_for_plan(question: str, context_text: str) -> str:
    """Generate numbered plan without giving final answer."""
    prompt = f"""
You are an AI assistant that reads provided context and makes a precise,
numbered plan to answer the user's question. RETURN ONLY A PLAN (numbered steps).
Do NOT give the final answer in this response.

Context:
{context_text}

Question:
{question}

Plan:
"""
    result = await llm.ainvoke(prompt)
    return result.content if hasattr(result, "content") else str(result)


async def execute_plan(plan: str, context_text: str, auth_token: str, question: str) -> str:
    """Execute plan and produce final answer."""
    executed_plan = plan
    async with httpx.AsyncClient() as client:
        for line in plan.splitlines():
            line_stripped = line.strip()

            # GET request execution
            m_get = re.search(r"GET\s+(https?://\S+)", line_stripped)
            if m_get:
                url = re.sub(r"[`'\",.;)]+$", "", m_get.group(1).strip())
                try:
                    resp = await client.get(url, headers={"Authorization": auth_token})
                    resp.raise_for_status()
                    value = resp.text.strip().replace('"', '')
                except Exception as e:
                    value = f"<HTTP ERROR: {type(e).__name__}: {e}>"
                executed_plan = executed_plan.replace(line, f"{line} → {value}")
                continue

    final_prompt = f"""
Question:
**We just need consise answers**

{question}

Context:
{context_text}

Executed Steps:
{executed_plan}

Final Answer:
"""
    result = await llm.ainvoke(final_prompt)
    return result.content if hasattr(result, "content") else str(result)
@app.post("/hackrx/run")
async def hackrx_run(req: HackRxRunRequest, Authorization: str = Header(default=None)):
    if not Authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    doc_url = req.documents

    # If new doc, process & store compact facts
    if doc_url not in facts_cache:
        parsed_doc = await parse_document_from_url(doc_url)
        chunks = split_documents(parsed_doc)

        # Store FAISS for fallback retrieval
        embedding_model = NomicEmbeddings()
        db = FAISS.from_documents(chunks, embedding_model)
        faiss_cache[doc_url] = db

        # Create one compact knowledge base
        prompt = f"""
        You are a knowledge extractor. Read the following text and
        compress all important facts, numbers, names, and events
        into a concise but complete format. Use bullet points.
        ---
        { "\n".join([c.page_content for c in chunks]) }
        """
        result = await llm.ainvoke(prompt)
        facts_cache[doc_url] = result.content.strip()

    compact_facts = facts_cache[doc_url]

    async def fast_answer(q: str):
        prompt = f"""
        Using the following pre-extracted knowledge, answer the question concisely.

        Knowledge:
        {compact_facts}

        Question:
        {q}

        Answer:
        """
        result = await llm.ainvoke(prompt)
        return result.content.strip()

    answers = await asyncio.gather(*(fast_answer(q) for q in req.questions))
    return {"answers": answers}
