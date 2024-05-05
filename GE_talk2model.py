from dotenv import load_dotenv
from langchain.llms import GPT4All, LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
import argparse

# Load environment variables
if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

# Environment variable configurations
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))

# Function to process a single query
def process_query(query, callbacks=[]):
    # Prepare the LLM based on the model type
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case _default:
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

    # Get the answer from the model directly
    answer = llm(query)

    # Return the result
    return answer

# Main function
def main():
    # Parse the command line arguments
    args = parse_arguments()

    # Activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        answer = process_query(query, callbacks)
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

# Argument parser function
def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your model directly without document retrieval.')
    parser.add_argument("--mute-stream", "-M", action='store_true', help='Use this flag to disable the streaming StdOut callback for LLMs.')
    return parser.parse_args()

# Entry point
if __name__ == "__main__":
    main()
