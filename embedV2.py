import os
import glob
from multiprocessing import Pool
from tqdm import tqdm
from typing import List

# Import necessary components for document processing and embedding
from dotenv import load_dotenv
#from excel_loader import ExcelLoader
#from langchain.document_loaders import *
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    PyPDFLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Load environment variables
if not load_dotenv():
    print("Failed to load .env file. Ensure it exists and is not empty.")
    exit(1)

# Configuration from environment variables
CONFIG = {
    "persist_directory": os.getenv("PERSIST_DIRECTORY"),
    "source_directory": os.getenv("SOURCE_DIRECTORY", "source_documents"),
    "embeddings_model_name": os.getenv("EMBEDDINGS_MODEL_NAME"),
    "chunk_size": 500,
    "chunk_overlap": 50,
}

# Document loader mapping
LOADER_MAPPING = {
    ".csv": CSVLoader,
    ".doc": UnstructuredWordDocumentLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".enex": EverNoteLoader,
    ".epub": UnstructuredEPubLoader,
    ".html": UnstructuredHTMLLoader,
    ".md": UnstructuredMarkdownLoader,
    ".odt": UnstructuredODTLoader,
    ".pdf": PyPDFLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".txt": TextLoader,
    #".xls": ExcelLoader,  # For older Excel files
    #".xlsx": ExcelLoader,  # For newer Excel files
}

def load_document(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in LOADER_MAPPING:
        loader = LOADER_MAPPING[ext](file_path)
        return loader.load()
    raise ValueError(f"Unsupported file extension '{ext}'")

def load_all_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    document_files = [
        file for ext in LOADER_MAPPING.keys()
        for file in glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        if file not in ignored_files
    ]
    
    with Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.imap(load_document, document_files), total=len(document_files), desc='Loading Documents'))
    return [doc for result in results for doc in result]

def process_and_embed_documents():
    print(f"Loading documents from {CONFIG['source_directory']}")
    documents = load_all_documents(CONFIG['source_directory'])
    
    if not documents:
        print("No documents to process.")
        return
    
    print(f"Processing {len(documents)} documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CONFIG['chunk_size'], chunk_overlap=CONFIG['chunk_overlap'])
    texts = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name=CONFIG['embeddings_model_name'])
    chroma_client = None 

    # Check if the vectorstore exists and append or create new if it does not exists already 
    vectorstore_path = CONFIG['persist_directory']
    if os.path.exists(vectorstore_path):
        print("Updating existing vectorstore.")
        db = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings, client=chroma_client)
    else:
        print("Creating new vectorstore.")
        db = Chroma.from_documents(texts, embeddings, persist_directory=vectorstore_path, client=chroma_client)
    
    db.persist()
    print("Document embedding complete.")

if __name__ == "__main__":
    process_and_embed_documents()
