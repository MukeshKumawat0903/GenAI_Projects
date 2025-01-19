# RAG Document Q&A with Groq and Llama3

## Table of Contents

- [RAG Document Q\&A with Groq and Llama3](#rag-document-qa-with-groq-and-llama3)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Usage](#usage)
  - [Technologies Used](#technologies-used)
  - [Prerequisites](#prerequisites)
  - [Setup and Usage](#setup-and-usage)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Install Dependencies](#2-install-dependencies)
    - [3. Start the Application](#3-start-the-application)
    - [4. Initialize the Vector Database](#4-initialize-the-vector-database)
    - [5. Query the Documents](#5-query-the-documents)
  - [Troubleshooting](#troubleshooting)
  - [Future Improvements](#future-improvements)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

---

## Introduction

This Streamlit-based application enables you to perform **Question Answering (Q&A)** on a set of research papers using **Groq** and **Llama3** models, along with **LangChain** for document retrieval and response generation. It leverages a **vector database** to index the documents, enabling fast and relevant retrieval of information based on user queries.

## Features

- **Document Retrieval**: Efficiently retrieve context from research papers using a vector database (FAISS).
- **Q&A with Groq and Llama3**: Use Groq for efficient AI model access and Llama3 for document-based question answering.
- **Interactive Interface**: Built using **Streamlit** for an intuitive user interface to input queries and get responses.
- **Contextual Responses**: Answers to questions are based on the content of documents, ensuring the relevance of responses.
- **Performance Monitoring**: View response times for queries to monitor efficiency.

## Usage

1. **Input Your Query**: Once the app is running, type a query related to your research papers in the input field.
2. **Initialize Vector Database**: If the vector database has not been initialized yet, click the "Initialize Vector Database" button to begin indexing the documents.
3. **Query Processing**: Upon entering a query, the system will retrieve relevant content from the documents and provide an answer based on the content.
4. **View Relevant Documents**: Relevant documents used to generate the response will be displayed under the "Relevant Documents" section.

## Technologies Used

- **Streamlit**: For the web interface.
- **Groq (Llama3-8b-8192)**: LLM for generating answers.
- **FAISS**: Vector database for document retrieval.
- **Hugging Face Embeddings**: To create embeddings for document chunks.
- **LangChain**: For managing document loaders, text splitters, and retrieval chains.

## Prerequisites

1. **Python (>=3.8)**
   Ensure Python and `pip` are installed on your system.
2. **Install Dependencies**:
   Install required Python packages with the following command:

   ```bash
   pip install -r requirements.txt
   ```
3. **Environment Variables**:
   Create a `.env` file in the root directory with the following content:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   GROQ_API_KEY=your_groq_api_key
   HF_TOKEN=your_huggingface_token
   ```

   Replace `your_openai_api_key`, `your_groq_api_key`, and `your_huggingface_token` with valid API keys.
4. **Research Papers Directory**:
   Create a directory named `research_papers` in the project folder. Add PDF documents to this directory.

## Setup and Usage

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/your-repo-url/rag-document-qa.git
cd rag-document-qa
```

### 2. Install Dependencies

Run the following command to install required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Start the Application

Run the Streamlit app using:

```bash
streamlit run app.py
```

This will open the application in your default web browser.

### 4. Initialize the Vector Database

- On the app interface, click **"Initialize Vector Database"** to process and index the documents.
- Ensure the `research_papers` directory contains valid PDFs before initialization.

### 5. Query the Documents

- Enter your question in the provided input box.
- The application will display a response along with relevant document excerpts.

## Troubleshooting

- **Error: Missing or Empty Research Papers Directory**
  Ensure the `research_papers` directory exists and contains at least one valid PDF document.
- **Slow Query Responses**
  The first query may take longer due to model and database initialization. Subsequent queries will be faster.
- **API Key Issues**
  Verify that valid API keys are set in the `.env` file.

## Future Improvements

- Add support for more document formats (e.g., Word, TXT).
- Enable fine-tuning or customizing LLM responses.
- Enhance UI for better query and document management.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [LangChain](https://www.langchain.com/)
- [Hugging Face](https://huggingface.co/)
- [Groq](https://groq.com/)

---

Feel free to contact [your_email@example.com](mailto:your_email@example.com) for any questions or suggestions.
