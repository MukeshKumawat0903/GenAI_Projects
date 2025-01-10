import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.abspath('../.env'))

# Fetch API key from environment
api_key = os.getenv("LANGCHAIN_API_KEY")

if not api_key:
    st.error("Missing LANGCHAIN_API_KEY. Please check your .env file.")
    st.stop()

os.environ["LANGCHAIN_API_KEY"] = api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With Ollama"

# Define the prompt template
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to user queries."),
        ("user", "Question: {user_query}")
    ]
)

def generate_model_response(user_query, model_name, temp, max_tokens):
    """
    Generate a response from the selected LLM model based on user input.

    Args:
        user_query (str): The question or input provided by the user.
        model_name (str): The name of the language model to be used (e.g., "mistral").
        temp (float): The temperature parameter controlling the randomness of the model's response.
        max_tokens (int): The maximum number of tokens (words or subwords) for the model's response.

    Returns:
        str: The generated response from the model or an error message in case of failure.
    
    Raises:
        Exception: If there is an issue in the model invocation or response generation.
    """
    try:
        # Initialize the model
        model = OllamaLLM(model=model_name, temperature=temp, max_tokens=max_tokens, format="json")
        output_parser = StrOutputParser()

        # Prepare input data for the model
        input_data = {"user_query": user_query}

        # Build the processing chain
        processing_chain = chat_prompt | model | output_parser

        # Display input data for debugging (can be removed in production)
        st.write(f"Processing query: {input_data}")

        # Invoke the chain and return the model's response
        response = processing_chain.invoke(input_data)
        return response
    except Exception as error:
        # Handle any errors that occur during response generation
        st.error(f"Error: {str(error)}")
        return f"An error occurred while generating the response."

# Streamlit interface
st.title("Enhanced Q&A Chatbot with Ollama")

# Sidebar configurations
selected_model = st.sidebar.selectbox("Select Open Source Model", ["mistral"])
temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 300, 150)

# Input field for user query
st.write("Feel free to ask any question:")
user_query_input = st.text_input("Your Question:")

# Display response if the user has entered a query
if user_query_input:
    model_response = generate_model_response(user_query_input, selected_model, temp, max_tokens)
    st.write("Assistant:", model_response)
else:
    st.write("Please enter a question to start the conversation.")
