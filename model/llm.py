# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
import os

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# os.environ['GROQ_API_KEY'] = GROQ_API_KEY

GROQ_Model = "gemma2-9b-it"
Cohere_Model = "command-r"

llm  = ChatCohere(
    model=Cohere_Model,
    cohere_api_key=COHERE_API_KEY
    )

# llm = ChatGroq(
#     model=GROQ_Model,
#     temperature=0.2,
#     # reasoning_format="parsed",
#     api_key=GROQ_API_KEY
# )

# llm = ChatOpenAI(
#     model=OPEN_AI_Model,
#     api_key=OPEN_AI_KEY 
# )
# llm = ChatOpenAI( 
#     model = OPEN_ROUTER_Model,
#     api_key=LLM_API,
#     base_url = "https://openrouter.ai/api/v1"
# )