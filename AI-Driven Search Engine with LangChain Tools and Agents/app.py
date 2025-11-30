import os
from typing import Dict, Any, List
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables
dotenv_path = os.path.abspath('../.env')
load_dotenv(dotenv_path)

# Initialize ArXiv and Wikipedia API wrappers
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)

# Define tools
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# Wrapper for DuckDuckGo Search to handle rate limits
ddg_wrapper = DuckDuckGoSearchAPIWrapper(
    max_results=5,  # Limit results to reduce noise and improve performance
)

def safe_search(query: str) -> str:
    """
    Safe web search using DuckDuckGo with rate-limit and error handling.
    
    Args:
        query (str): Search query string
        
    Returns:
        str: Search results or error message
    """
    try:
        return ddg_wrapper.run(query)
    except Exception as e:
        err = str(e)
        if "rate" in err.lower() or "ratelimit" in err.lower():
            return (
                "The web search tool hit a rate limit. "
                "Please wait a moment and try again, or ask me to use Wikipedia or arXiv instead."
            )
        return f"Search failed due to an unexpected error: {err}"

search_tool = Tool(
    name="Search",
    func=safe_search,
    description="Useful for searching the internet for current events and general information."
)

def process_uploaded_pdfs(uploaded_files: List, llm: ChatGroq) -> tuple:
    """
    Process uploaded PDF files and create a vector store for searching.
    
    Args:
        uploaded_files: List of uploaded PDF files from Streamlit
        llm: Language model instance
        
    Returns:
        tuple: (vectorstore, pdf_search_tool) or (None, None) if no files
    """
    if not uploaded_files:
        return None, None
    
    all_docs = []
    
    # Process each uploaded PDF
    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load PDF
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        
        # Clean up temp file
        os.remove(temp_path)
        
        all_docs.extend(documents)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(all_docs)
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Create retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    # Create a tool for PDF search
    def search_pdfs(query: str) -> str:
        """Search through uploaded PDF documents."""
        try:
            result = qa_chain.invoke({"query": query})
            answer = result["result"]
            sources = result.get("source_documents", [])
            
            # Add source information
            if sources:
                source_info = "\n\nSources from uploaded PDFs:\n"
                for i, doc in enumerate(sources[:2], 1):  # Show top 2 sources
                    page = doc.metadata.get("page", "unknown")
                    source_info += f"- Page {page}\n"
                answer += source_info
            
            return answer
        except Exception as e:
            return f"Error searching PDFs: {str(e)}"
    
    pdf_tool = Tool(
        name="PDF_Search",
        func=search_pdfs,
        description="Search through uploaded PDF documents for specific information. Use this when the question relates to the uploaded documents."
    )
    
    return vectorstore, pdf_tool

def initialize_llm(api_key: str) -> ChatGroq:
    """
    Initialize the Groq language model.
    
    Args:
        api_key (str): API key for accessing the Groq language model.
        
    Returns:
        ChatGroq: Instance of the initialized language model.
    """
    return ChatGroq(
        groq_api_key=api_key, 
        model_name="llama-3.3-70b-versatile", 
        streaming=True,
        temperature=0.7  # Balance between creativity and consistency
    )

def initialize_agent_with_tools(llm: ChatGroq, pdf_tool=None) -> AgentExecutor:
    """
    Initializes and configures an agent with a set of tools to handle user queries.

    Args:
        llm (ChatGroq): The large language model instance used for reasoning and query processing.
        pdf_tool: Optional PDF search tool to add to the agent's capabilities.

    Returns:
        AgentExecutor: The configured agent executor ready to handle user queries with tools.

    Process:
        1. Defines a list of tools (`search_tool`, `arxiv_tool`, `wiki_tool`, optionally `pdf_tool`) that the agent can use.
        2. Pulls a predefined "react" prompt from the LangChain Hub for generating instructions for the agent.
        3. Creates a ReAct-based agent using the tools and the LLM.
        4. Configures the agent executor with settings for error handling, verbosity, and execution constraints.
    """
    # Define the tools that the agent will have access to
    tools = [search_tool, arxiv_tool, wiki_tool]
    
    # Add PDF tool if available
    if pdf_tool:
        tools.append(pdf_tool)

    # Retrieve the "ReAct" prompt from LangChain Hub to guide the agent's decision-making
    prompt = hub.pull("hwchase17/react")

    # Create the agent using the ReAct framework, which allows reasoning and tool use
    search_agent = create_react_agent(
        llm=llm, tools=tools, prompt=prompt
    )

    # Initialize the AgentExecutor to manage the agent's execution with constraints
    agent_executor = AgentExecutor(
        agent=search_agent,
        tools=tools,
        verbose=True,                # Enable detailed logs for execution
        handle_parsing_errors=True,  # Allow graceful handling of parsing errors
        max_iterations=15,           # Increased from 10 to allow more tool calls
        max_execution_time=300,      # Set the timeout (in seconds) for query processing
        early_stopping_method="generate"  # Generate a response even if not fully complete
    )

    # Return the configured agent executor
    return agent_executor

def main() -> None:
    """
    Main Streamlit application for a chatbot with search capabilities.
    """
    # Streamlit UI setup
    st.title("🔎 AI-Driven Search Engine with LangChain Tools and Agents")
    st.write(
        "Chat with a multi-source search agent using ArXiv, Wikipedia, "
        "DuckDuckGo, and uploaded PDF documents. The agent will intelligently choose which sources to use "
        "and cite them in its answers."
    )

    # Sidebar settings
    st.sidebar.title("Settings")
    api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
    
    # PDF Upload Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📄 Upload PDFs")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF documents to search through",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to add them as a searchable source"
    )
    
    # Add helpful information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Tips")
    st.sidebar.markdown(
        "- The agent can search across **Web**, **Wikipedia**, **arXiv**, and **uploaded PDFs**\n"
        "- It will cite sources used in responses\n"
        "- Follow-up questions use conversation context\n"
        "- If search fails, try rephrasing your question"
    )
    
    # Add clear chat button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web and your documents. How can I help you?"}
        ]
        if "llm" in st.session_state:
            del st.session_state["llm"]
        if "agent_executor" in st.session_state:
            del st.session_state["agent_executor"]
        if "pdf_vectorstore" in st.session_state:
            del st.session_state["pdf_vectorstore"]
        st.rerun()

    # Initialize chat messages in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web and your documents. How can I help you?"}
        ]
    
    # Process PDFs if uploaded and not already processed
    pdf_tool = None
    if uploaded_files and api_key:
        # Check if PDFs have changed
        current_pdf_names = [f.name for f in uploaded_files]
        if "processed_pdf_names" not in st.session_state or st.session_state.get("processed_pdf_names") != current_pdf_names:
            with st.spinner("Processing uploaded PDFs..."):
                if "llm" not in st.session_state:
                    st.session_state["llm"] = initialize_llm(api_key)
                
                vectorstore, pdf_tool = process_uploaded_pdfs(uploaded_files, st.session_state["llm"])
                st.session_state["pdf_vectorstore"] = vectorstore
                st.session_state["pdf_tool"] = pdf_tool
                st.session_state["processed_pdf_names"] = current_pdf_names
                
                # Force agent recreation with new PDF tool
                if "agent_executor" in st.session_state:
                    del st.session_state["agent_executor"]
                
                st.sidebar.success(f"✅ Processed {len(uploaded_files)} PDF(s)")
        else:
            # Use cached PDF tool
            pdf_tool = st.session_state.get("pdf_tool")
    
    # Initialize agent in session state if API key is provided (caching for performance)
    if api_key and "llm" not in st.session_state:
        with st.spinner("Initializing AI agent..."):
            st.session_state["llm"] = initialize_llm(api_key)
    
    # Initialize or recreate agent if needed
    if api_key and ("agent_executor" not in st.session_state or pdf_tool):
        with st.spinner("Initializing AI agent..."):
            st.session_state["agent_executor"] = initialize_agent_with_tools(
                st.session_state["llm"], 
                pdf_tool=pdf_tool
            )

    # Display chat messages
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    # Handle user input
    user_input = st.chat_input(placeholder="What is machine learning?")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Ensure the API key is provided
        if not api_key:
            st.error("Please enter your Groq API Key in the sidebar.")
            return
        
        # Ensure agent is initialized
        if "agent_executor" not in st.session_state:
            st.error("Agent is not initialized. Please check your API key and try again.")
            return
        
        agent_executor = st.session_state["agent_executor"]
        
        # Build conversation context for better follow-up questions
        history_context = ""
        if len(st.session_state["messages"]) > 2:  # More than just initial greeting and current question
            recent_messages = st.session_state["messages"][-6:-1]  # Last 5 messages (excluding current)
            for msg in recent_messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_context += f"{role}: {msg['content']}\n"
            
            full_input = f"""Previous conversation context:
{history_context}
Current question: {user_input}

Please answer the current question, using the conversation context if relevant. Always mention which sources you used (Wikipedia, arXiv, Web search, or PDF documents) in your response."""
        else:
            full_input = f"{user_input}\n\nPlease mention which sources you used (Wikipedia, arXiv, Web search, or PDF documents) in your response."

        # Process the user query with the agent
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                response = agent_executor.invoke(
                    {"input": full_input}, {"callbacks": [st_callback]}
                )
                st.session_state["messages"].append({"role": "assistant", "content": response["output"]})
                st.write(response["output"])
            except Exception as e: 
                error_msg = str(e)
                if "iteration limit" in error_msg or "time limit" in error_msg:
                    fallback_response = (
                        "The agent reached its iteration or time limit while processing your question. "
                        "This usually happens with complex queries. Try:"
                        "\n- Breaking your question into smaller parts"
                        "\n- Being more specific about what you want to know"
                        "\n- Asking a follow-up question"
                    )
                    st.session_state["messages"].append({"role": "assistant", "content": fallback_response}) 
                    st.warning(fallback_response)
                else: 
                    error_response = f"An error occurred: {error_msg}"
                    st.error(error_response)
                    st.session_state["messages"].append({"role": "assistant", "content": error_response})

if __name__ == "__main__":
    main()


