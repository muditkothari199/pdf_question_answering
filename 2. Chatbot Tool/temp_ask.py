import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
import warnings

# Download NLTK data if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Suppress specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0.")

# Remove stop words from text
def remove_stopwords(pdf_text):
    words = word_tokenize(pdf_text)
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filter_text = " ".join(filtered_words)
    return filter_text

def get_pdf_text(pdf_docs):
    documents = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            documents += page.extract_text()
    return documents

def get_text_chunk(text):
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n'], chunk_size=100000, chunk_overlap=70000)
    chunks = text_splitter.split_text(text)
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
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return chain

def generate_response(user_question):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
    
    # Convert vector store to retriever
    retriever = vector_store.as_retriever()
    
    # Create the conversational chain using the retriever
    chain = get_conversational_chain(retriever)
    docs = retriever.get_relevant_documents(user_question)
   

    response = chain(
        {
            "input_documents": docs,  # Pass the documents from similarity search
            "query": user_question
        }, return_only_outputs=True)

    # print(response['result'])
    # st.write("Reply: ", response['result'])
    # response = chain.run(user_question)
    return response['result']

def main():
    st.title("PDF Question-Answering Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        st.write(pdf_docs)
        
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    raw_text = remove_stopwords(raw_text)
                    text_chunks = get_text_chunk(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete!")
            else:
                st.warning("Please upload at least one PDF file.")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a Question from the PDF Files"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = generate_response(prompt)
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
