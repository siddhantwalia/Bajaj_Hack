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

from model import Prompt, llm, NomicEmbeddings, rewrite_llm
from utils import parse_document_from_url, split_documents

load_dotenv()
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

faiss_cache = {}

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]


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

def clean_output(answer):
    """Cleans unwanted tags and excessive spacing from LLM output."""
    if hasattr(answer, "content"):
        content = answer.content
    else:
        content = str(answer)
    content = content.strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return re.sub(r"\n{3,}", "\n\n", content)

@app.get("/")
async def home():
    return {"home": "This is our API endpoint"}

@app.post("/hackrx/run")
async def run_query(req: QueryRequest, Authorization: str = Header(default=None)):
    start = time.time()

    if not Authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if req.documents in faiss_cache:
        db, texts = faiss_cache[req.documents]
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.8})
        logger.info("Using cached FAISS retriever")
    else:
        try:
            parsed_doc = await parse_document_from_url(req.documents)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error parsing document: {e}")

        try:
            chunks = split_documents(parsed_doc)
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
            faiss_cache[req.documents] = (db, texts)
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
            logger.info(f"Context {context} for question {question}")
            inputs = {"context": context, "question": question}
            answer = await (Prompt | llm).ainvoke(inputs)
            return clean_output(answer)

        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            return f"Error: {e}"

    answers = await asyncio.gather(*[process_question(q) for q in req.questions])
    logger.info(f"Total time: {time.time() - start:.2f}s")
    return {"answers": answers}
