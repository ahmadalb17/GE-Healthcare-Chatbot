import streamlit as st
from dotenv import load_dotenv
from langchain.llms import GPT4All, LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os

# Load environment variables
if not load_dotenv():
    st.error("Could not load .env file or it is empty. Please check if it exists and is readable.")
    st.stop()

# Environment variable configurations
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))

# Function to process query
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



logo_path = 'logo.png'  
st.image(logo_path, width=400)

st.title('Ask the model')
st.write('Ask questions to the model directly without document retrieval.\n You will not get answers based on the documents nor any sources.')


# User input
user_query = st.text_input("Ask me something !", "")
submit_button = st.button('Submit')

if user_query and submit_button:
    with st.spinner('Processing...'):
        try:
            answer = process_query(user_query)
            #st.write("### Question:")
            #st.write(user_query)
            st.write("### Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"Error processing the query: {e}")

