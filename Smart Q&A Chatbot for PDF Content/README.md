# Smart Q&A Chatbot for PDF Content

## Introduction
A powerful and interactive chatbot (**Conversational RAG Chatbot**) that enables users to upload PDF files and engage in a conversational Q&A experience with the content. This chatbot leverages advanced language models and retrieval-augmented generation (RAG) techniques to provide accurate answers while maintaining a chat history for context-aware interactions.

---

## Features

- **PDF Upload Support**: Upload one or multiple PDF files for content extraction.
- **Conversational Q&A**: Ask questions about the content, and the chatbot provides concise, accurate answers.
- **Chat History**: Retain conversation context for seamless, history-aware interactions.
- **RAG Implementation**: Combines document retrieval and question answering to enhance response accuracy.
- **Streamlit Interface**: User-friendly web-based interface for interaction.
- **Customizable Language Model**: Integrates with HuggingFace and Groq APIs for flexible model usage.

---

## Requirements

### Prerequisites
- Python 3.8+
- A HuggingFace API token (for embeddings)
- A Groq API key (for the language model)
- Libraries: `streamlit`, `dotenv`, `langchain`, `langchain_chroma`, `langchain_community`, `langchain_core`, `langchain_groq`, `langchain_huggingface`, `langchain_text_splitters`

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/smart-qa-chatbot-pdf.git
    cd smart-qa-chatbot-pdf
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up the environment:
    - Create a `.env` file in the root directory.
    - Add your HuggingFace and Groq API tokens:
      ```env
      HF_TOKEN=your_huggingface_token
      ```

---

## Usage

1. Start the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Access the web interface in your browser (default: `http://localhost:8501`).

3. Upload PDF files, enter your Groq API key, and start interacting with the content!

---

## Project Structure

```
smart-qa-chatbot-pdf/
├── app.py                 # Main application script
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not included in repo)
└── README.md              # Project documentation
```

---

## Key Technologies

- **Streamlit**: Provides the web interface for user interaction.
- **LangChain**: Enables seamless integration of language models and document retrieval.
- **HuggingFace**: Offers high-quality embeddings for document processing.
- **Groq API**: Leverages advanced language models for conversational AI.

---

## Future Enhancements

- Add support for other document formats (e.g., Word, Excel).
- Implement multi-user session management.
- Improve response generation with fine-tuned models.
- Add option to export chat history.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- [LangChain](https://www.langchain.com/)
- [HuggingFace](https://huggingface.co/)
- [Streamlit](https://streamlit.io/)
- [Groq](https://groq.com/)
