import streamlit as st
from GE_bot import process_query, load_dotenv


logo_path = 'logo.png'  
st.image(logo_path, width=400)

# Load environment variables
if not load_dotenv():
    st.error("Could not load .env file or it is empty. Please check if it exists and is readable.")
    st.stop()

# Title 
st.title('Chat with your document')

# Question from user
query = st.text_input('Enter your question:', '')

# Checkbox options for mute stream / hide source
mute_stream = st.checkbox('Mute Stream', value=False)
hide_source = st.checkbox('Hide Source', value=False)

# Button to process the query   
if st.button('Submit'):
    try:
        # Call processing function
        answer, docs_with_similarity  = process_query(query,mute_stream, hide_source)

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
