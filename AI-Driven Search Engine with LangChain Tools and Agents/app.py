import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
# from langchain.agents import initialize_agent, AgentType
from langchain.agents import create_react_agent
# from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.prompts import PromptTemplate


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
    return ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

def initialize_agent_with_tools(llm):
    """
    Initialize the LangChain agent with the specified tools and LLM.
    
    Args:
        llm (ChatGroq): Instance of the language model to use.
        
    Returns:
        LangChainAgent: Configured LangChain agent.
    """
    tools = [search_tool, arxiv_tool, wiki_tool]

    prompt_template = """
    You are an AI assistant equipped with tools to help answer questions.
    Available tools:
    {tools}

    The tools you can use: {tool_names}

    When solving a query, think step-by-step and decide which tool to use.
    Keep track of your progress in the scratchpad:
    {agent_scratchpad}

    Query: {input}
    """
    prompt = PromptTemplate(
        input_variables=["tools", "tool_names", "agent_scratchpad", "input"],
        template=prompt_template
    )

    # search_agent = initialize_agent(
    #     tools=tools,
    #     llm=llm,
    #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     handle_parsing_errors=True
    # )

    search_agent = create_react_agent(
        llm, tools=tools,
        prompt= prompt
    )
    return search_agent

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

    # for msg in st.session_state.messages:
    #     st.chat_message(msg["role"]).write(msg['content'])

    # Handle user input
    user_input = st.chat_input(placeholder="What is machine learning?")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        # st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Ensure the API key is provided
        if not api_key:
            st.error("Please enter your Groq API Key in the sidebar.")
            return

        # Initialize LLM and agent
        llm = initialize_llm(api_key)
        search_agent = initialize_agent_with_tools(llm)

        # Process the user query with the agent
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                # response = search_agent.run(st.session_state["messages"], callbacks=[st_callback])
                # response = search_agent.invoke(st.session_state["messages"], callbacks=[st_callback])
                print("Test1")
                # Send only the latest user query to the agent
                response = search_agent.invoke({"input": user_input}, callbacks=[st_callback])
                print("Test2")

                st.session_state["messages"].append({"role": "assistant", "content": response})
                # st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
