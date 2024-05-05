import streamlit as st
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF
from transformers import MarianMTModel, MarianTokenizer

# Load environment variables
if not load_dotenv():
    st.error("Could not load .env file or it is empty. Please check if it exists and is readable.")
    st.stop()

# Environment variable configurations
model_name = os.environ.get('MODEL_NAME')
model = MarianMTModel.from_pretrained(model_name, use_auth_token="your_token_here")


def translate_text(text, model, tokenizer):
    translated = model.generate(**tokenizer.prepare_translation_batch([text]))
    translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return ' '.join(translated_text)

def read_pdf(file):
    """
    Extracts text from the first page of a PDF file.
    """
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def main():
    st.title("Document Translator")
    st.write("Upload a document in Norwegian, and it will be translated to English.")

    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf'])
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            document_content = read_pdf(uploaded_file)
        else:
            document_content = uploaded_file.getvalue().decode("utf-8")

        # Download the translation model from Hugging Face
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)

        # Translate the document
        translated_text = translate_text(document_content, model, tokenizer)

        # Display the translated text
        st.text_area("Translated Text", translated_text, height=250)

if __name__ == "__main__":
    main()
