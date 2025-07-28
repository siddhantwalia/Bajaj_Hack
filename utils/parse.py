import nest_asyncio
import os
import requests
import tempfile
from llama_parse import LlamaParse
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument
from langchain_community.document_loaders import PyMuPDFLoader


load_dotenv()
nest_asyncio.apply()

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

async def parse_pdf_from_url(url: str):

    response = requests.get(url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    # parser = LlamaParse(
    #     result_type="markdown",   
    #     output_tables_as_HTML=True,
    #     api_key=LLAMA_CLOUD_API_KEY
    # )
    parser = PyMuPDFLoader(tmp_path)

    # documents = parser.load_data(tmp_path)
    documents = parser.load()
    return documents

def split_documents(parsed_docs, chunk_size=600, chunk_overlap=100):
    all_chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    try:
        # doc_obj = [LCDocument(
        #     page_content=doc.text_resource.text,
        #     metadata=doc.metadata,
        #     id=doc.id_
        # ) for doc in parsed_docs]
        # # doc_obj = 
        
        chunks = splitter.split_documents(parsed_docs)
        all_chunks.extend(chunks)
    except Exception as e:
        print(f"Error processing document chunk: {e}")

    return all_chunks
