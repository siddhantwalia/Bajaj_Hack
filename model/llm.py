from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

LLM_API = os.getenv("LLM_API_KEY")
OPEN_AI_KEY =os.getenv("OPEN_AI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_APO_KEY")

OPEN_ROUTER_Model = "deepseek/deepseek-r1-0528:free"
OPEN_AI_Model = 'gpt-4.1-mini-2025-04-14'
GROQ_Model = "gemma2-9b-it"

llm = ChatGroq(
    model=GROQ_Model,
    temperature=0.2,
    # reasoning_format="parsed",
    api_key=GROQ_API_KEY
)

# llm = ChatOpenAI(
#     model=OPEN_AI_Model,
#     api_key=OPEN_AI_KEY 
# )
# llm = ChatOpenAI( 
#     model = OPEN_ROUTER_Model,
#     api_key=LLM_API,
#     base_url = "https://openrouter.ai/api/v1"
# )