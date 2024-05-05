import streamlit as st
import os
import glob
from multiprocessing import Pool
from tqdm import tqdm
from typing import List
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Load environment variables
if not load_dotenv():
    st.error("Failed to load .env file. Ensure it exists and is not empty.")
    st.stop()

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
    ".pdf": PyPDFLoader
}

def check_if_file_exists(file_name: str) -> bool:
    # Check if a file with the same name already exists in the source directory.
    return os.path.exists(os.path.join(CONFIG['source_directory'], file_name))

def load_document(file_path: str) -> List[Document]:
    #Loads document based on the file extension using the appropriate loader.
    ext = os.path.splitext(file_path)[1].lower()
    if ext in LOADER_MAPPING:
        loader = LOADER_MAPPING[ext](file_path)
        return loader.load()
    raise ValueError(f"Unsupported file extension '{ext}'")

def load_all_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    #Loads all documents from the source directory, ignoring files listed in ignored_files.
    document_files = [
        file for ext in LOADER_MAPPING.keys()
        for file in glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        if file not in ignored_files
    ]
    
    with Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.imap(load_document, document_files), total=len(document_files), desc='Loading Documents'))
    return [doc for result in results for doc in result]

def process_and_embed_documents():
    #Processes and embeds documents from the source directory.
    print(f"Loading documents from {CONFIG['source_directory']}")
    documents = load_all_documents(CONFIG['source_directory'])
    
    if not documents:
        print("No documents to process.")
        return
    
    print(f"Processing {len(documents)} documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CONFIG['chunk_size'], chunk_overlap=CONFIG['chunk_overlap'])
    texts = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name=CONFIG['embeddings_model_name'])
    vectorstore_path = CONFIG['persist_directory']
    if os.path.exists(vectorstore_path):
        print("Updating existing vectorstore.")
        db = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
    else:
        print("Creating new vectorstore.")
        db = Chroma.from_documents(texts, embeddings, persist_directory=vectorstore_path)
    
    db.persist()
    print("Document embedding complete.")

def run_streamlit_upload():
    # aploading, checking, and processing documents.
    st.title('Document Upload for Embedding')

    uploaded_file = st.file_uploader("Choose a document file", type=[ext.lstrip('.') for ext in LOADER_MAPPING.keys()])
    if uploaded_file is not None:
        if check_if_file_exists(uploaded_file.name):
            st.error(f"A document with the name '{uploaded_file.name}' already exists in the source folder.")
        else:
            # save the uploaded file to the source directory
            file_path = os.path.join(CONFIG['source_directory'], uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
                st.success(f"File '{uploaded_file.name}' uploaded successfully.")
            
            
            process_and_embed_documents() 
            st.write(f"Document embedding complete.")

if __name__ == "__main__":
    run_streamlit_upload()
