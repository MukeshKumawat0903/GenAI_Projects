import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain import hub

# Load environment variables
dotenv_path = os.path.abspath('../.env')
load_dotenv(dotenv_path)

# Initialize ArXiv and Wikipedia API wrappers
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)

# Define tools
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
search_tool = DuckDuckGoSearchRun(name="Search")

def initialize_llm(api_key: str):
    """
    Initialize the Groq language model.
    
    Args:
        api_key (str): API key for accessing the Groq language model.
        
    Returns:
        ChatGroq: Instance of the initialized language model.
    """
    return ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", streaming=True)

def initialize_agent_with_tools(llm):
    """
    Initializes and configures an agent with a set of tools to handle user queries.

    Args:
        llm (ChatGroq): The large language model instance used for reasoning and query processing.

    Returns:
        AgentExecutor: The configured agent executor ready to handle user queries with tools.

    Process:
        1. Defines a list of tools (`search_tool`, `arxiv_tool`, `wiki_tool`) that the agent can use.
        2. Pulls a predefined "react" prompt from the LangChain Hub for generating instructions for the agent.
        3. Creates a ReAct-based agent using the tools and the LLM.
        4. Configures the agent executor with settings for error handling, verbosity, and execution constraints.
    """
    # Define the tools that the agent will have access to
    tools = [search_tool, arxiv_tool, wiki_tool]

    # prompt_template = """
    # You are an AI assistant equipped with tools to help answer questions.
    # Available tools:
    # {tools}

    # The tools you can use: {tool_names}

    # When solving a query, think step-by-step and decide which tool to use.
    # Keep track of your progress in the scratchpad:
    # {agent_scratchpad}

    # Query: {input}
    # """
    # prompt = PromptTemplate(
    #     input_variables=["tools", "tool_names", "agent_scratchpad", "input"],
    #     template=prompt_template
    # )

    # Retrieve the "ReAct" prompt from LangChain Hub to guide the agent's decision-making
    prompt = hub.pull("hwchase17/react")  # Replace with a valid prompt if needed

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
        max_iterations=10,           # Set the maximum number of iterations per query
        max_execution_time=300       # Set the timeout (in seconds) for query processing
    )

    # Return the configured agent executor
    return agent_executor

def main():
    """
    Main Streamlit application for a chatbot with search capabilities.
    """
    # Streamlit UI setup
    st.title("🔎 AI-Driven Search Engine with LangChain Tools and Agents")
    st.write(
        "Chat with a multi-source search agent using ArXiv, Wikipedia, "
        "and DuckDuckGo tools. For more examples, visit the "
        "[LangChain Streamlit Agent Repository](https://github.com/langchain-ai/streamlit-agent)."
    )

    # Sidebar settings
    st.sidebar.title("Settings")
    api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

    # Initialize chat messages in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
        ]

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

        # Initialize LLM and agent
        llm = initialize_llm(api_key)
        # search_agent = initialize_agent_with_tools(llm)

        # Initialize the AgentExecutor
        agent_executor = initialize_agent_with_tools(llm)

       # Process the user query with the agent
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                response = agent_executor.invoke(
                    {"input": user_input}, {"callbacks": [st_callback]}
                )
                st.session_state["messages"].append({"role": "assistant", "content": response["output"]})
                st.write(response["output"])
            except Exception as e: 
                if "iteration limit" in str(e) or "time limit" in str(e):  
                    st.session_state["messages"].append({"role": "assistant", "content": "The agent reached its iteration or time limit. Here's the best answer I can provide based on the information gathered so far."}) 
                    st.write("The agent reached its iteration or time limit. Here's the best answer I can provide based on the information gathered so far.") 
                else: 
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()


