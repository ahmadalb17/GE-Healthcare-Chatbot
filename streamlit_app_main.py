import streamlit as st
from dotenv import load_dotenv
import os

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
    from GE_bot import process_query  

    logo_path = 'logo.png'
    st.image(logo_path, width=400)

    if not load_dotenv():
        st.error("Could not load .env file or it is empty. Please check if it exists and is readable.")
        st.stop()

    st.title('Chat with your document')

    query = st.text_input('Enter your question:', '')
    mute_stream = st.checkbox('Mute Stream', value=False)
    hide_source = st.checkbox('Hide Source', value=False)

    # Button to process the query
    if st.button('Submit'):
        with st.spinner('Processing...'):  
            try:
                # Call processing function
                answer, docs_with_similarity = process_query(query, hide_source)

                # Apply threshold
                if docs_with_similarity:
                    # Display the answer
                    st.subheader('Answer')
                    st.write(answer)

                    # If wanted display the sources if not hidden
                    if not hide_source:
                        st.subheader('Sources and Similarity Scores')
                        for doc in docs_with_similarity:
                            st.write(f"**Source**: {doc['source']}")
                            st.write(f"**Similarity Score**: {doc['similarity_score']:.4f}")
                            st.text(doc['content'])
                else:
                    st.subheader('Answer')
                    st.write("I don't have enough information in my source documents to answer your question \nplease ask the question again differently or add new information in the add document page")

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



    # Back to main 
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