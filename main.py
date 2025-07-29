import os
import re
import asyncio
import time
from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
from typing import List
from model import Prompt, llm,NomicEmbeddings
from utils import parse_document_from_url, split_documents
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

class QueryRequest(BaseModel):
    documents : str
    questions : List[str]
    

@app.get('/')
async def home():
    return {"home":"This is our api endpoint"}

@app.post("/hackrx/run")
async def run_query(
    req: QueryRequest,
    Authorization: str = Header(None)
):
    start = time.time()
    # print(f"Received auth: {Authorization}")
    if not Authorization:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    p_time =time.time()
    try:
        parse_doc = await parse_document_from_url(req.documents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing document: {str(e)}")
    print(f"Parsing time: {time.time()-p_time}")
    
    c_time = time.time()
    try:
        chunks = split_documents(parse_doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Splitting failed: {str(e)}")
    
    texts = [chunk.page_content for chunk in chunks]
    print(f"Chunking time:  {time.time()-c_time}")
    
    e_time = time.time()
    try:
        embedding_model = NomicEmbeddings()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")
    print(f"embedding generation time:  {time.time()-e_time}")
    
    s_time = time.time()
    try:
        db = FAISS.from_documents(chunks, embedding_model)
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10,"fetch_k":20})

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Vector DB error: {str(e)}")
    print(f"storing vdb time:  {time.time()-s_time}")
    
    async def get_answer(question): 
        context_docs = retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in context_docs])
        inputs = {"context": context, "question": question}
        try:
            answer = await (Prompt | llm).ainvoke(inputs) 
            return clean_output(answer)
        except Exception as e:
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
    # for question in req.questions:
    #     context_docs = retriever.invoke(question)
    #     context = "\n".join([doc.page_content for doc in context_docs])
    #     inputs = {"context": context, "question": question}
    #     try:
    #         answer = (Prompt | llm).invoke(inputs)
    #         cleaned = answer.content.strip()
    #         cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    #         answers.append(cleaned)
    #     except Exception as e:
    #         answers.append(f"Error generating answer: {str(e)}")
    print(f"llm respnose time: {time.time()-llm_res}")
    
    print(f"Total time: {time.time()-start}")
    return {"answers": answers}