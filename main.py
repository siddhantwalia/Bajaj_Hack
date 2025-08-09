import asyncio
import logging
import httpx
import re
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from utils import parse_document_from_url
from model import llm  # Your GPT model wrapper

load_dotenv()
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]


async def ask_gpt_for_plan(question: str, document_text: str) -> str:
    """Ask GPT for a detailed step-by-step plan to get the answer."""
    prompt = f"""
You are an AI assistant that reads documents and figures out exactly how to answer a question.

Document:
{document_text}

Question:
{question}

Instructions:
1. Read the document carefully.
2. Make a clear, numbered step-by-step plan to get the final answer.
3. Include any API calls explicitly (e.g., GET https://...).
4. If a value must be looked up from the document, describe exactly how.
5. Do NOT give the final answer yet, only the plan.

Plan:
"""
    result = await llm.ainvoke(prompt)
    return result.content if hasattr(result, "content") else str(result)


async def execute_plan(plan: str, document_text: str, auth_token: str) -> str:
    """Very simple executor: runs API calls mentioned in plan and substitutes results."""
    # Example: detect and run GET requests
    async with httpx.AsyncClient() as client:
        for line in plan.splitlines():
            match = re.search(r"GET\s+(https?://\S+)", line)

            if match:
                url = match.group(1).strip()
                url = re.sub(r"[\`\'\"\,\.\)]*$", "", url)  # strip unwanted chars
                resp = await client.get(url, headers={"Authorization": auth_token})
                resp.raise_for_status()
                value = resp.text.strip().replace('"', '')
                plan = plan.replace(line, f"{line} → {value}")

        match = re.search(r"GET\s+(https?://\S+)", line)
    
    # Ask GPT to now produce final answer based on executed steps
    final_prompt = f"""
Document:
{document_text}

Executed Steps with Results:
{plan}

Now, using the results above, give the final answer to the question clearly.
"""
    result = await llm.ainvoke(final_prompt)
    return result.content if hasattr(result, "content") else str(result)


@app.post("/run-agent")
async def run_agent(req: QueryRequest, Authorization: str = Header(default=None)):
    if not Authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    parsed_doc = await parse_document_from_url(req.documents)
    document_text = "\n".join([sec.page_content for sec in parsed_doc])

    answers = []
    for question in req.questions:
        logger.info(f"Processing: {question}")
        plan = await ask_gpt_for_plan(question, document_text)
        answer = await execute_plan(plan, document_text, Authorization)
        answers.append(answer)

    return {"answers": answers}
