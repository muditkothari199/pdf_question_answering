from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain.docstore.document import Document
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
import pytesseract
from pdf2image import convert_from_path
import re
import os
import tkinter as tk
from tkinter import filedialog

import warnings

# Suppress specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0.")


#remove stops words from text
def remove_stopwords(pdf_text):
    words =word_tokenize(pdf_text)
    stop_words= set(stopwords.words("english"))

    filtered_words= [word for word in words if word not in stop_words]
    filter_text=" ".join(filtered_words)
    
    return filter_text




def get_pdf_text(pdf_docs):
    documents = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            documents += page.extract_text()
    return documents

def get_text_chunk(text):
    text_splitter = RecursiveCharacterTextSplitter(separators='\n',chunk_size=10000, chunk_overlap=7000)
    chunks = text_splitter.split_text(text)
    # print(chunks)
    return chunks



def get_vector_store(text_chunk):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    docs = [Document(page_content=text) for text in text_chunk]
    vector_store = FAISS.from_documents(docs, embedding)
    vector_store.save_local("faiss_index")
    return vector_store


def get_conversational_chain(retriever):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    llm = Ollama(model='llama3.1', temperature=0.03)
    prompt = PromptTemplate(template=prompt_template, input_variables=["Context", "Question"])
    
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return chain

def user_input(user_question):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
    
    # Convert vector store to retriever
    retriever = vector_store.as_retriever()
    
    # Perform similarity search directly on the retriever
    
    # Create the conversational chain using the retriever
    chain = get_conversational_chain(retriever)
    docs = retriever.get_relevant_documents(user_question)
   

    response = chain(
        {
            "input_documents": docs,  # Pass the documents from similarity search
            "query": user_question
        }, return_only_outputs=True)

    # print(response['result'])
    st.write("Reply: ", response['result'])



def main():
    st.header("Question-Answering with PDF")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        st.write(pdf_docs)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                # print(raw_text)
                raw_text= remove_stopwords(raw_text)
                text_chunks = get_text_chunk(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
