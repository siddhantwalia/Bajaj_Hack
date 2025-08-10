import os
import re
import asyncio
import time
import logging
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import httpx
# Import your models, prompts, and utils from both workflows
from model import Prompt, llm, NomicEmbeddings, rewrite_llm
from utils import parse_document_from_url, split_documents

# If you have plan-execution helpers from the first file
# from plan_model import ask_gpt_for_plan, execute_plan  # <- adjust import to match your file

load_dotenv()
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

faiss_cache = {}

# Page threshold for switching pipelines
PAGE_THRESHOLD = 10  # adjust to your needs


class QueryRequest(BaseModel):
    documents: str
    questions: List[str]


# ---------------------- Common Utility ---------------------- #
def clean_output(answer):
    """Cleans unwanted tags and excessive spacing from LLM output."""
    if hasattr(answer, "content"):
        content = answer.content
    else:
        content = str(answer)
    content = content.strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return re.sub(r"\n{3,}", "\n\n", content)


async def rewrite_question(original_question: str, first_doc_chunk: str = "") -> str:
    """Use LLM to rewrite question for better retrieval."""
    prompt_template_str = """
    You are an expert query rewriter for a document retrieval system using OpenAI embeddings and MMR search.
    Rewrite the original question into a concise, keyword-rich query with synonyms and related terms.
    Avoid unrelated details. Do not change the meaning.

    First Chunk of Document (for context, if available): {first_doc_chunk}

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
import re
import httpx
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

async def execute_plan(plan: str, context_text: str, auth_token: str, question: str) -> str:
    executed_plan = plan

    # Clean the Authorization token (remove BOM, spaces, non-printables)
    if auth_token:
        auth_token = auth_token.strip().encode("utf-8", errors="ignore").decode("ascii", errors="ignore")

    async with httpx.AsyncClient() as client:
        for line in plan.splitlines():
            line_stripped = line.strip()

            # --- GET request handling ---
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

            # --- LOOKUP handling ---
            if line_stripped.upper().startswith("LOOKUP:"):
                instruction = line_stripped[len("LOOKUP:"):].strip()
                lookup_result = perform_lookup(instruction, context_text)
                executed_plan = executed_plan.replace(line, f"{line} → {lookup_result}")
                continue

            # --- EXTRACT_REGEX handling ---
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

    return executed_plan


    # Final answer
    final_prompt = f"""
    You are a helpful assistant.

    Answer the question in clear, natural English, using a maximum of 2–3 sentences.  
    Base your answer strictly on the provided context and executed steps — do not add any information from outside them.  
    Ensure the tone is precise, human-like, and directly addresses the question without extra fluff.

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

# ---------------------- Direct RAG Pipeline ---------------------- #
async def direct_rag_pipeline(parsed_docs, req, Authorization):
    doc_key = req.documents

    if doc_key in faiss_cache:
        db, texts = faiss_cache[doc_key]
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.8})
        logger.info("Using cached FAISS retriever")
    else:
        try:
            chunks = split_documents(parsed_docs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Splitting failed: {e}")

        if not chunks:
            raise HTTPException(status_code=400, detail="No text chunks found in document.")

        texts = [chunk.page_content for chunk in chunks]

        try:
            embedding_model = NomicEmbeddings()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding init failed: {e}")

        try:
            db = FAISS.from_documents(chunks, embedding_model)
            retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.8})
            faiss_cache[doc_key] = (db, texts)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Vector store error: {e}")

    async def process_question(question: str):
        try:
            logger.info(f"Processing question: {question}")

            if re.fullmatch(r"\s*\d+\s*[\+\-\*/]\s*\d+\s*", question):
                rewritten_question = question
            else:
                rewritten_question = await rewrite_question(question, texts[0] if texts else "")

            context_docs = await asyncio.to_thread(retriever.invoke, rewritten_question)
            context = "\n".join([doc.page_content for doc in context_docs]) if context_docs else ""

            if not context.strip():
                logger.warning(f"No context found for '{question}', falling back to direct LLM answer")
                inputs = {"context": "", "question": question}
                answer = await (Prompt | llm).ainvoke(inputs)
                return clean_output(answer)

            inputs = {"context": context, "question": question}
            answer = await (Prompt | llm).ainvoke(inputs)
            return clean_output(answer)

        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            return f"Error: {e}"

    return {"answers": await asyncio.gather(*[process_question(q) for q in req.questions])}


# ---------------------- Plan Execution Pipeline ---------------------- #
async def plan_execution_pipeline(parsed_docs, req, Authorization):
    # Reuse your first workflow logic
    chunks = split_documents(parsed_docs)
    full_doc_text = "\n".join(chunk.page_content for chunk in chunks)

    answers = []
    for question in req.questions:
        logger.info(f"Generating plan for question: {question}")
        plan = await ask_gpt_for_plan(question, full_doc_text)
        logger.info(f"Executing plan: {plan}")
        answer = await execute_plan(plan, chunks, full_doc_text, question)
        answers.append(answer)

    return {"answers": answers}


# ---------------------- API Routes ---------------------- #
@app.get("/")
async def home():
    return {"home": "This is our API endpoint"}


@app.post("/hackrx/run")
async def run_query(req: QueryRequest, Authorization: str = Header(default=None)):
    start = time.time()

    if not Authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    # Parse the document (returns list of Document objects)
    try:
        parsed_docs = await parse_document_from_url(req.documents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing document: {e}")

    # Determine page count
    try:
        page_count = len(parsed_docs)
    except Exception:
        page_count = 1  # fallback

    logger.info(f"Document has {page_count} pages. Threshold is {PAGE_THRESHOLD}.")

    # Choose pipeline based on page count
    if page_count <= PAGE_THRESHOLD:
        logger.info("Using Plan Execution Pipeline")
        result = await plan_execution_pipeline(parsed_docs, req, Authorization)
    else:
        logger.info("Using Direct RAG Pipeline")
        result = await direct_rag_pipeline(parsed_docs, req, Authorization)

    logger.info(f"Total time: {time.time() - start:.2f}s")
    return result
