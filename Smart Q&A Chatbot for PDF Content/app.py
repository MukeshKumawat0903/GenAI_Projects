import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# Load environment variables
def load_env():
    """
    Load environment variables from a .env file.

    Returns:
        str: HuggingFace token retrieved from the .env file.
    """
    dotenv_path = os.path.abspath('../.env')
    load_dotenv(dotenv_path)
    hf_token = os.getenv("HF_TOKEN")
    return hf_token


# Initialize HuggingFace embeddings
def initialize_embeddings(hf_token):
    """
    Initialize HuggingFace embeddings with the provided token.

    Args:
        hf_token (str): HuggingFace token for accessing the API.

    Returns:
        HuggingFaceEmbeddings: Initialized embeddings instance.

    Raises:
        ValueError: If the HuggingFace token is missing.
    """
    if not hf_token:
        raise ValueError("HuggingFace token is missing.")
    os.environ['HF_TOKEN'] = hf_token
    embeddings =  HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


# Upload and process PDF files
def process_uploaded_files(uploaded_files):
    """
    Process uploaded PDF files into a list of documents.

    Args:
        uploaded_files (list): List of uploaded PDF files.

    Returns:
        list: Extracted documents from all uploaded files.
    """
    documents = []
    temp_path = "./temp.pdf"
    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily for processing
        with open(temp_path, "wb") as file:
            file.write(uploaded_file.getvalue())
        # Load the PDF and extract documents
        loader = PyPDFLoader(temp_path)
        documents.extend(loader.load())
    return documents


# Create retriever and RAG chain
def setup_rag_chain(llm, embeddings, documents):
    """
    Set up a Retrieval-Augmented Generation (RAG) chain.

    Args:
        llm (ChatGroq): Language model instance for Q&A.
        embeddings (HuggingFaceEmbeddings): Pre-initialized embeddings.
        documents (list): List of documents to create the retriever.

    Returns:
        RetrievalChain: A retrieval chain for Q&A tasks.
    """
    # Split documents into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)

    # Create a vector store and retriever
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Create contextualize question prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, "
                   "formulate a standalone question without relying on chat history."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Create question-answering prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. Use the context to answer concisely. "
                   "If unsure, respond with 'I don't know.'\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # Combine retriever and QA chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


# Manage chat history
def get_session_history(session_id, store):
    """
    Retrieve or initialize the chat history for the given session.

    Args:
        session_id (str): Unique identifier for the session.
        store (dict): Persistent storage for session histories.

    Returns:
        ChatMessageHistory: The chat history for the session.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Main application
def main():
    """
    Streamlit-based application for a Conversational RAG chatbot
    with PDF uploads and chat history management.
    """
    st.title("Smart Q&A Chatbot for PDF Content")
    st.write("Upload PDFs and chat with their content.")

    # Load environment variables and initialize embeddings
    hf_token = load_env()
    embeddings = initialize_embeddings(hf_token)

    # Input for API key
    api_key = st.text_input("Enter your Groq API key:", type="password")
    if not api_key:
        st.warning("Please enter the Groq API Key.")
        return

    # Initialize the language model
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    # Input for session ID
    session_id = st.text_input("Session ID", value="default_session")

    # Initialize session state for storing histories
    if 'store' not in st.session_state:
        st.session_state.store = {}

    # Handle file uploads
    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        documents = process_uploaded_files(uploaded_files)
        rag_chain = setup_rag_chain(llm, embeddings, documents)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]       

        def run_rag_chain(user_input):
            """
            Execute the RAG chain with the user's input and session history.

            Args:
                user_input (str): User's question or input.

            Returns:
                str: Response from the assistant.
            """
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                },  # constructs a key "abc123" in `store`.
            )    

            return response

        # Handle user input and display responses
        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = run_rag_chain(user_input)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)


if __name__ == "__main__":
    main()
