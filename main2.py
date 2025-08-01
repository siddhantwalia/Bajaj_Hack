import os
import re
import asyncio
import time
from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
from typing import List
from model import Prompt, llm,NomicEmbeddings,HuggingFaceEmbed,rewrite_llm
from utils import parse_document_from_url, split_documents
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import logging

load_dotenv()
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    documents : str
    questions : List[str]
    

@app.get('/')
async def home():
    return {"home":"This is our api endpoint"}

@app.post("/hackrx/run")
async def run_query(
    req: QueryRequest,
    Authorization: str = Header(default=None, alias="Authorization")

):
    start = time.time()
    # print(f"Received auth: {Authorization}")
    if not Authorization:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

# Parse document
    try:
        parse_doc = await parse_document_from_url(req.documents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing document: {str(e)}")

    # Split document into chunks
    try:
        chunks = split_documents(parse_doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Splitting failed: {str(e)}")

    texts = [chunk.page_content for chunk in chunks]

    # Embedding
    try:
        embedding_model = NomicEmbeddings()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    # Create vector store
    try:
        db = FAISS.from_documents(chunks, embedding_model)
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10, "lambda_mult": 0.5})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector DB error: {str(e)}")

    # Query rewriting layer using GPT-4.1 Mini
    async def rewrite_question(original_question: str) -> str:
        prompt_text = (
            "Rewrite the following question so that it is formal, clear, and matches the style used in policy documents. "
            "Ensure full entities and unambiguous phrasing:\n\n"
            f"Question: {original_question}"
        )
        try:
            rewritten = await rewrite_llm.ainvoke(prompt_text)
            return rewritten.content.strip() if hasattr(rewritten, "content") else str(rewritten).strip()
        except Exception as e:
            print(f"Rewrite error: {e}")
            return original_question  # fallback

    # Final answer generator
    async def get_answer(question: str):
        logger.info(f"Retrieving context for question: {question}")
        rewritten_question = await rewrite_question(question)
        context_docs = retriever.invoke(rewritten_question)
        context = "\n".join([doc.page_content for doc in context_docs])
        # logger.info(f"Context retrieved for question: {context}")
        inputs = {"context": context, "question": question}
        try:
            answer = await (Prompt | llm).ainvoke(inputs)
            return clean_output(answer)
        except Exception as e:
            return f"Error: {str(e)}"

    # Clean up LLM output
    def clean_output(answer):
        if hasattr(answer, "content"):
            content = answer.content
        else:
            content = str(answer)
        content = content.strip()
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content

    # Run LLM calls concurrently for all questions
    answers = await asyncio.gather(*[get_answer(q) for q in req.questions])
    logger.info(f"Total time: {time.time() - start:.2f}s")
    return {"answers": answers}