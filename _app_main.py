import streamlit as st
from dotenv import load_dotenv
import os
from streamlit_add_document import LOADER_MAPPING, check_if_file_exists, process_and_embed_documents

# Configuration from environment variables
CONFIG = {
    "persist_directory": os.getenv("PERSIST_DIRECTORY"),
    "source_directory": os.getenv("SOURCE_DIRECTORY", "source_documents"),
    "embeddings_model_name": os.getenv("EMBEDDINGS_MODEL_NAME"),
    "chunk_size": 500,
    "chunk_overlap": 50,
}



def back_to_main_button():
    if 'main_button_placeholder' not in st.session_state:
        # Create a placeholder for the back button
        st.session_state.main_button_placeholder = st.empty()

    # Display the button in the placeholder
    with st.session_state.main_button_placeholder.container():
        if st.button('Back to Main Page'):
            st.session_state.current_page = 'main'
            # Clear the placeholder when navigating back to main
            st.session_state.main_button_placeholder.empty()
            st.experimental_rerun()


# Function for Page 1 
def talk2documents():
    from GE_bot_1 import process_query  

    logo_path = 'logo.png'
    st.image(logo_path, width=400)

    if not load_dotenv():
        st.error("Could not load .env file or it is empty. Please check if it exists and is readable.")
        st.stop()

    st.title('Chat with your document')

    query = st.text_input('Enter your query:', '')
    mute_stream = st.checkbox('Mute Stream', value=False)
    hide_source = st.checkbox('Hide Source', value=False)

    if st.button('Submit'):
        try:
            with st.spinner('Processing...'):
                answer, docs = process_query(query, mute_stream, hide_source)
            st.subheader('Answer')
            st.write(answer)

            if not hide_source:
                st.subheader('Sources')
                for doc in docs:
                    st.write(f"**Source**: {doc.metadata['source']}")
                    st.text(doc.page_content)
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Bacl to main 
    if st.button('Back to Main Page'):
        st.session_state.current_page = 'main'
        st.experimental_rerun()




# Function for Page 2 
def talk2model():

    from GE_talk2model import process_query

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
    st.write('Ask questions directly to the model without document retrieval. You will not receive answers based on documents or any sources.')

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
    # Back to main 
    if st.button('Back to Main Page'):
        st.session_state.current_page = 'main'
        st.experimental_rerun()




def add_document_page():
    st.title('Document Upload for Embedding')

    uploaded_file = st.file_uploader("Choose a document file", type=[ext.lstrip('.') for ext in LOADER_MAPPING.keys()])
    if uploaded_file is not None:
        if check_if_file_exists(uploaded_file.name):
            st.error(f"The document with the name '{uploaded_file.name}' already exists in the source folder.")
        else:
            file_path = os.path.join(CONFIG['source_directory'], uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
                st.success(f"File '{uploaded_file.name}' uploaded successfully.")
            
            process_and_embed_documents()
            st.write("Document embedding complete.")

    if st.button('Back to Main Page'):
        st.session_state.current_page = 'main'
        st.experimental_rerun()



# Main Page
def main_page():
    logo_path = 'logo.png'
    st.image(logo_path, width=400)

    st.title('Main Page')

    if st.button('Ask the documents'):
        st.session_state.current_page = 'talk2documents'
        st.experimental_rerun()
    elif st.button('Ask the model'):
        st.session_state.current_page = 'talk2model'
        st.experimental_rerun()
    elif st.button('Add Document'):
        st.session_state.current_page = 'add_document_page'
        st.experimental_rerun()

# Initialize session state for current page if not present
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'main'

# Page navigation
if st.session_state.current_page == 'main':
    main_page()
elif st.session_state.current_page == 'talk2documents':
    talk2documents()
elif st.session_state.current_page == 'talk2model':
    talk2model()
elif st.session_state.current_page == 'add_document_page':
    add_document_page()