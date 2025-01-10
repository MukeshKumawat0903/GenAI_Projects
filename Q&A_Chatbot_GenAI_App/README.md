# Q&A Chatbot with Ollama

An interactive **Q&A Chatbot** built using **Ollama** and **LangChain** frameworks. This chatbot can answer user queries by leveraging a language model (LLM) and a pre-defined prompt template. It's designed to be simple, efficient, and easy to use.

## Table of Contents

- [Q\&A Chatbot with Ollama](#qa-chatbot-with-ollama)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
  - [Usage](#usage)
  - [Development](#development)
    - [Running Tests](#running-tests)

## Features

- **Interactive Interface**: Built using **Streamlit**.
- **Customizable Parameters**: Control model selection, temperature, and max tokens.
- **Environment Configuration**: Load sensitive API keys securely.
- **Multiple Model Support**: Choose models (e.g., Mistral).
- **Error Handling**: Clear error messages.

## Installation

### Prerequisites

- Python 3.7+
- **Environment Variables**: The project uses **LangChain API** for communication with the models, and the API key should be stored in an `.env` file.

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/qa-chatbot-with-ollama.git
   cd qa-chatbot-with-ollama
   ```
2. Set up a virtual environment (recommended):

   ```bash
   python -m venv venv
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Set up the `.env` file:

   ```dotenv
   LANGCHAIN_API_KEY=your-api-key-here
   ```
5. Run the application:

   ```bash
   streamlit run Q&A_Chatbot_With_Ollama.py
   ```

## Usage

1. Select the language model (e.g., **Mistral**) from the sidebar.
2. Adjust parameters such as **Temperature** and **Max Tokens**.
3. Enter a query and see the assistant's response.
