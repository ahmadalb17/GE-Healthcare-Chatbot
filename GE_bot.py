from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import chromadb
import os
import argparse
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

_threshold = 0.2


# Load environment variables
if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

# Environment variable configurations
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

# Import config loader 
from config_loader import CHROMA_SETTINGS

# Initialize global variables
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS, path=persist_directory)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})


# Cosine similarity function to compute hte cosine similarity between 2 embeddings
def compute_similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)

    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)

    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    return similarity

# Function to process a single query
def process_query(query, mute_stream=True, hide_source=True):
    # Prepare the LLM based on the model type
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=[], verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=[], verbose=False)
        case _default:
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=not hide_source)

    # Get the answer from the chain
    res = qa(query)
    answer, docs = res['result'], [] if hide_source else res['source_documents']

    # Generate the embedding for the question 
    question_embedding = embeddings.embed_query(query)

    
    # A list to store documents with their similarity scores
    docs_with_similarity = []
    
    for document in docs:
        doc_embedding = embeddings.embed_query(document.page_content)
        similarity = compute_similarity(question_embedding, doc_embedding)

        # Apply threshold 
        if similarity >= _threshold:
            # Append document information along with similarity score
            docs_with_similarity.append({
                "source": document.metadata["source"],
                "content": document.page_content,
                "similarity_score": similarity
            })
        
    # Return only the chunks with value >= _threshold 
    return answer, docs_with_similarity


# Main function
def main():
    # Parse the command line arguments
    args = parse_arguments()

    # Activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    # Prepare the LLM based on the model type
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case _default:
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=not args.hide_source)

    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        answer, docs_with_similarity = process_query(query, args.mute_stream, args.hide_source)

        # Check for chunks with value >= threshold 
        if docs_with_similarity:
            print("\n\n> Question:")
            print(query)
            print("\n> Answer:")
            print(answer)

            # Print the relevant sources used for the answer
            for document in docs_with_similarity:
                print("\n> " + document["source"] + ":")
                print(document["content"])
                print(f"Similarity Score: {document['similarity_score']:.4f}")
        else: 
            print("I don't have enough information in my source documents to answer your question \nplease ask the question again differently or add new information in the add document page")

# Argument parser function
def parse_arguments():
    parser = argparse.ArgumentParser(description='GE-bot: Ask questions to your documents.')
    parser.add_argument("--hide-source", "-S", action='store_true', help='Use this flag to disable printing of source documents used for answers.')
    parser.add_argument("--mute-stream", "-M", action='store_true', help='Use this flag to disable the streaming StdOut callback for LLMs.')
    return parser.parse_args()


if __name__ == "__main__":
    main()
