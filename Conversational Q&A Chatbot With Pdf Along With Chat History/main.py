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
import os
from dotenv import load_dotenv

# Load environment variables
# Adjust the path to your .env file as needed
dotenv_path = os.path.abspath('../.env')
load_dotenv(dotenv_path)
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit app setup
st.title("Conversational RAG with PDF Uploads and Chat History")
st.write("Upload PDF files and interact with their content using conversational AI.")

# Input the Groq API key
api_key = st.text_input("Enter your Groq API key:", type="password")

if api_key:
    # Initialize the Groq language model
    language_model = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    # Input for session ID
    session_id = st.text_input("Session ID", value="default_session")

    # Manage chat history state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = {}

    # File uploader for PDFs
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        # Process uploaded PDFs
        all_documents = []
        for uploaded_file in uploaded_files:
            temp_pdf_path = "./temp.pdf"
            with open(temp_pdf_path, "wb") as temp_file:
                temp_file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf_path)
            docs = loader.load()
            all_documents.extend(docs)

        # Split documents and create embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        document_chunks = text_splitter.split_documents(all_documents)
        vector_store = Chroma.from_documents(documents=document_chunks, embedding=embeddings)
        retriever = vector_store.as_retriever()

        # Contextualization prompt for reformulating user questions
        contextualization_prompt_text = (
            "Given a chat history and the latest user question, formulate a standalone question that can be "
            "understood without the chat history. Do NOT answer the question, just reformulate it if needed."
        )
        contextualization_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualization_prompt_text),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(language_model, retriever, contextualization_prompt)

        # QA prompt for providing concise answers
        qa_system_prompt = (
            "You are an assistant for answering questions. Use the retrieved context to answer the question. "
            "If you don't know the answer, say so. Keep the answer concise, using no more than three sentences.\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        qa_chain = create_stuff_documents_chain(language_model, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        def get_chat_history(session: str) -> BaseChatMessageHistory:
            """Retrieve or initialize chat history for a given session ID."""
            if session not in st.session_state.chat_history:
                st.session_state.chat_history[session] = ChatMessageHistory()
            return st.session_state.chat_history[session]

        conversational_chain = RunnableWithMessageHistory(
            rag_chain,
            get_chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Input for user query
        user_query = st.text_input("Your question:")
        if user_query:
            session_history = get_chat_history(session_id)
            response = conversational_chain.invoke(
                {"input": user_query},
                config={"configurable": {"session_id": session_id}}
            )
            st.write("Response:", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter your Groq API key to proceed.")