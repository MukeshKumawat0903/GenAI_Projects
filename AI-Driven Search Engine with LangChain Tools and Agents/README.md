# AI-Driven Search Engine with LangChain Tools and Agents

## Introduction

This project is an AI-powered search engine that leverages LangChain tools and agents to provide intelligent, multi-source query responses. The application integrates ArXiv, Wikipedia, and DuckDuckGo search capabilities, allowing users to interact with a chatbot that retrieves and summarizes relevant information from these sources.

## Features

- **ArXiv Integration**: Query the ArXiv database for the latest research papers and summaries.
- **Wikipedia Integration**: Retrieve concise and relevant information from Wikipedia articles.
- **DuckDuckGo Search**: Perform general web searches for comprehensive results.
- **Streamlit Interface**: User-friendly chatbot interface powered by Streamlit.
- **Dynamic LLM Integration**: Uses the Groq language model for advanced natural language understanding and response generation.

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.9+
- [Streamlit](https://streamlit.io/)
- An API key for the Groq language model.

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/ai-driven-search-langchain.git
   cd ai-driven-search-langchain
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your environment variables:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the Streamlit app in your browser.
2. Enter your Groq API key in the sidebar.
3. Start chatting with the AI by typing your query in the input box.
4. The chatbot will respond by retrieving relevant information from ArXiv, Wikipedia, or the web.

## Code Overview

- **`initialize_llm(api_key)`**: Initializes the Groq language model with the provided API key.
- **`initialize_agent_with_tools(llm)`**: Configures the LangChain agent with ArXiv, Wikipedia, and DuckDuckGo tools.
- **`main()`**: Runs the Streamlit app, managing user interactions and responses.

## Requirements

- Python 3.9+
- `streamlit`
- `langchain`
- `dotenv`

## Example Queries

- *"What is quantum computing?"*
- *"Find research papers on reinforcement learning."*
- *"Summarize the Wikipedia page on artificial intelligence."*

## Troubleshooting

- Ensure your Groq API key is valid and added to the `.env` file or entered in the Streamlit sidebar.
- If the chatbot does not respond, check for network connectivity or API rate limits.

## Contributing

We welcome contributions! Feel free to submit a pull request or open an issue for suggestions and bug reports.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [LangChain](https://langchain.com/) for providing powerful tools and integrations.
- [Streamlit](https://streamlit.io/) for the intuitive app interface.
- [DuckDuckGo](https://duckduckgo.com/) and other data sources for search capabilities.