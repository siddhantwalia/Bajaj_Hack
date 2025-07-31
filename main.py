import os
import re
import asyncio
import time
import logging
from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
from typing import List
from model import Prompt, llm, NomicEmbeddings, HuggingFaceEmbed
from utils import parse_document_from_url, split_documents
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

@app.get('/')
async def home():
    return {"home": "This is our api endpoint"}

@app.post("/hackrx/run")
async def run_query(
    req: QueryRequest,
    Authorization: str = Header(default=None, alias="Authorization")
):
    start = time.time()
    if not Authorization:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    p_time = time.time()
    try:
        parse_doc = await parse_document_from_url(req.documents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing document: {str(e)}")
    logger.info(f"Parsing time: {time.time() - p_time}")
    
    c_time = time.time()
    try:
        chunks = split_documents(parse_doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Splitting failed: {str(e)}")
    
    texts = [chunk.page_content for chunk in chunks]
    logger.info(f"Chunking time: {time.time() - c_time}")
    
    e_time = time.time()
    try:
        embedding_model = NomicEmbeddings()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")
    logger.info(f"Embedding generation time: {time.time() - e_time}")
    
    s_time = time.time()
    try: 
        db = FAISS.from_documents(chunks, embedding_model)
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10, "lambda_mult": 0.5})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector DB error: {str(e)}")
    logger.info(f"Storing VDB time: {time.time() - s_time}")
    
    async def get_answer(question): 
        try:
            context_docs = retriever.invoke(question)
            context = "\n".join([doc.page_content for doc in context_docs])
            inputs = {"context": context, "question": question}
            answer = await (Prompt | llm).ainvoke(inputs) 
            return clean_output(answer)
        except Exception as e:
            logger.error(f"Error processing question '{question}': {str(e)}")
            return f"Error: {str(e)}"
    
    def clean_output(answer):
        if hasattr(answer, "content"): 
            content = answer.content
        else:
            content = str(answer)
        content = content.strip()
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content
    
    llm_res = time.time()
    answers = await asyncio.gather(*[get_answer(q) for q in req.questions])
    logger.info(f"LLM response time: {time.time() - llm_res}")
    
    logger.info(f"Total time: {time.time() - start}")
    return {"answers": answers}
