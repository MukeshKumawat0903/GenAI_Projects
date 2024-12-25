import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time

# Load environment variables
dotenv_path = os.path.abspath('../.env')  # Adjust the path as needed
load_dotenv(dotenv_path)

# Set API keys
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the language model and embeddings
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Define the prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

def initialize_vector_database():
    """Initialize the vector database for document retrieval."""
    if "vector_db" not in st.session_state:
        try:
            st.session_state.embeddings = embeddings_model
            st.session_state.loader = PyPDFDirectoryLoader("research_papers")
            st.session_state.documents = st.session_state.loader.load()

            if not st.session_state.documents:
                st.error("No documents found in the 'research_papers' directory.")
                return

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.split_documents = text_splitter.split_documents(st.session_state.documents[:50])

            st.session_state.vector_db = FAISS.from_documents(
                st.session_state.split_documents, st.session_state.embeddings
            )
            st.success("Vector database initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing vector database: {e}")
    else:
        st.info("Vector database is already initialized.")

# Ensure the vector database is initialized
def ensure_vector_database():
    if "vector_db" not in st.session_state:
        initialize_vector_database()

# Streamlit application
st.title("RAG Document Q&A with Groq and Llama3")

# User input for query
user_query = st.text_input("Enter your query about the research papers")

# Button to manually initialize the vector database
if st.button("Initialize Vector Database"):
    initialize_vector_database()

# Automatically initialize the vector database if not already done
ensure_vector_database()

if user_query:
    if "vector_db" in st.session_state:
        try:
            document_chain = create_stuff_documents_chain(llm, prompt_template)
            retriever = st.session_state.vector_db.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start_time = time.process_time()
            response = retrieval_chain.invoke({"input": user_query})
            elapsed_time = time.process_time() - start_time

            st.write(f"### Response ({elapsed_time:.2f} seconds):")
            st.write(response["answer"])

            with st.expander("Relevant Documents"):
                for i, doc in enumerate(response.get("context", [])):
                    st.write(doc.page_content)
                    st.write("------------------------")
        except Exception as e:
            st.error(f"Error processing query: {e}")
    else:
        st.error("Please initialize the vector database first.")

# Check for missing research papers directory
if not os.path.exists("research_papers") or not os.listdir("research_papers"):
    st.error("""The 'research_papers' directory is missing or empty. 
             Please ensure the directory exists and contains valid documents."""
             )
