import os
import re
import asyncio
import time
from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from typing import List
from model import Prompt, llm,NomicEmbeddings,HuggingFaceEmbed,rewrite_llm,OpenAITextEmbedding3Small
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
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.5})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector DB error: {str(e)}")

    # Query rewriting layer using GPT-4.1 Mini
    async def rewrite_question(original_question: str, first_doc_chunk: str = "") -> str:
        prompt_template_str = """
        You are an expert query rewriter for a document retrieval system using OPEN AI embeddings text-embedding-3-small and MMR (Maximum Marginal Relevance) search. 
        Your task is to rewrite the original question into an optimized query that maximizes retrieval accuracy. 
        Focus on making it easy for the retriever to match relevant document chunks by incorporating synonyms, related terms, and key phrases that align with the document's content. 
        Do NOT make the rewritten question formal, standalone, or complete sentences—keep it concise, keyword-rich, and query-like (e.g., similar to a search engine query). 
        Avoid adding unrelated details or changing the core meaning. The goal is semantic expansion for better embedding matches and diverse results via MMR.

        Key Guidelines:
        - Expand with 2-4 synonyms or related terms for main concepts (e.g., if the question is about 'freedom of speech', include 'free expression, first amendment rights, speech liberties').
        - Include specific entities or phrases from the original if they are precise (e.g., article numbers, names).
        - If provided, use the first chunk of the document to infer the topic, terminology, and style—incorporate matching keywords from it to boost relevance.
        - Keep the rewritten query under 100 words; prioritize diversity for MMR by avoiding repetition.
        - Do NOT use formal language, questions marks, or full sentences unless it helps retrieval—aim for a list-like or phrasal structure if beneficial.
        - If the document chunk is empty, rely solely on the original question.

        Examples:
        - Original: 'What is Article 21 about?'
        Rewritten: 'Article 21 significance, right to life liberty, personal freedom protections, constitutional importance India'
        - Original: 'Prohibitions on child labor'
        Rewritten: 'Child labor bans, age limits factories mines, hazardous work restrictions, Article 24 regulations'

        First Chunk of Document (for context, if available): {first_doc_chunk}

        Original Question: {original_question}

        Rewritten Query:
        """
        prompt_text = PromptTemplate(
            input_variables=['original_question', 'first_doc_chunk'],
            template=prompt_template_str
        )
        try:
            input_dict = {'original_question': original_question, 'first_doc_chunk': first_doc_chunk}
            formatted_prompt = await prompt_text.ainvoke(input_dict)  # Await to get the formatted prompt
            rewritten = await rewrite_llm.ainvoke(formatted_prompt)  # Pass the formatted prompt to LLM
            return rewritten.content.strip() if hasattr(rewritten, "content") else str(rewritten).strip()
        except Exception as e:
            print(f"Rewrite error: {e}")
            return original_question  # fallback


    # Final answer generator
    async def get_answer(question: str):
        logger.info(f"Retrieving context for question: {question}")
        rewritten_question = await rewrite_question(question,texts[0])
        logger.info(f"question reframed {rewritten_question}")
        # logger.info("/////////////////////////////////////////////////////////////")
        context_docs = retriever.invoke(rewritten_question)
        context = "\n".join([doc.page_content for doc in context_docs])
        logger.info(f"Context retrieved for question: {context}")
        # with open("context.txt", "w",encoding="utf-8") as f:
        #     f.write(context)
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