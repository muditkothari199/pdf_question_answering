from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.text_splitter import TextSplitter
from PyPDF2 import PdfReader
from langchain.vectorstores import SimpleVectorStore
from langchain.chains import ConversationalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain import LLMChain

# Initialize the LLaMA model (replace with appropriate initialization code)
llama_model = Ollama("llama-3.1")  # Ensure this matches the actual initialization for LLaMA 3.1



from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

pdf_texts = [extract_text_from_pdf("Auto\Auto 1.pdf")]




# Combine text from all PDFs
combined_text = "\n".join(pdf_texts)

# Initialize text splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(combined_text)

# Initialize VectorStore (for document retrieval, optional)
vector_store = SimpleVectorStore()
vector_store.add_documents(chunks)

# Initialize chain for querying
chain = LLMChain(
    llm=llama_model,
    prompt=PromptTemplate(
        template="Summarize the following text:\n\n{text}",
        input_variables=["text"]
    )
)

# Example query
query = "what is the policy type?"

# Process the query
summary = chain.run(text=combined_text)
print(summary)
