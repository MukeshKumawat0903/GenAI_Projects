# Conversational Q&A Chatbot With PDF Along With Chat History

## Overview
The "Conversational Q&A Chatbot With PDF Along With Chat History" project leverages Retrieval-Augmented Generation (RAG) to enable users to upload PDF documents and interact with their content through a conversational interface. This tool supports chat history, making interactions contextually aware and seamless.

## Features
- **PDF Uploads**: Upload multiple PDF documents and process their content for interactive querying.
- **Chat History Management**: Maintain context across interactions for coherent conversations.
- **Powered by RAG**: Uses Retrieval-Augmented Generation for efficient and accurate responses.
- **Embeddings Integration**: Employs HuggingFace embeddings for document understanding.
- **Streamlit Interface**: User-friendly web-based interface for interaction.

## Prerequisites
- Python 3.8+
- Streamlit
- HuggingFace Embeddings
- Groq API Key
- Chroma for vector storage
- dotenv for environment variable management

## Installation
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables:
   - Create a `.env` file in the root directory.
   - Add your HuggingFace token and other required keys:
     ```
     HF_TOKEN=<your-huggingface-token>
     ```

## Usage
1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Open the application in your browser (usually at `http://localhost:8501`).
3. Input your Groq API key to activate the chatbot functionality.
4. Upload your PDF files and start interacting with their content via chat.

## How It Works
1. **PDF Processing**: Uploaded PDFs are processed into text chunks using a recursive text splitter.
2. **Vector Store Creation**: Chroma creates a vector store for efficient retrieval using embeddings.
3. **Retrieval-Augmented QA**: Questions are answered using context retrieved from the vector store.
4. **Chat History Awareness**: Maintains and uses chat history for improved contextual understanding.

## Project Structure
```
project-folder/
|-- app.py                 # Main application script
|-- requirements.txt       # Dependencies
|-- .env                   # Environment variables (not included in the repo)
|-- temp.pdf               # Temporary storage for uploaded PDFs
```

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- [LangChain](https://langchain.com) for foundational RAG components.
- [HuggingFace](https://huggingface.co) for embedding models.
- [Streamlit](https://streamlit.io) for the interactive interface.
- [Chroma](https://www.trychroma.com) for vector storage.

---
Feel free to reach out with questions or feedback!
