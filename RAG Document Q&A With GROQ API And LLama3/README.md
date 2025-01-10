# RAG Document Q&A with Groq and Llama3

## Table of Contents

- [RAG Document Q\&A with Groq and Llama3](#rag-document-qa-with-groq-and-llama3)
  - Table of Contents
  - [Introduction](#introduction)
  - [Features](#features)
  - [Installation](#installation)
    - Prerequisites
    - [Setup Steps](#setup-steps)
  - Usage

---

## Introduction

This Streamlit-based application enables you to perform **Question Answering (Q&A)** on a set of research papers using **Groq** and **Llama3** models, along with **LangChain** for document retrieval and response generation. It leverages a **vector database** to index the documents, enabling fast and relevant retrieval of information based on user queries.

## Features

- **Document Retrieval**: Efficiently retrieve context from research papers using a vector database (FAISS).
- **Q&A with Groq and Llama3**: Use Groq for efficient AI model access and Llama3 for document-based question answering.
- **Interactive Interface**: Built using **Streamlit** for an intuitive user interface to input queries and get responses.
- **Contextual Responses**: Answers to questions are based on the content of documents, ensuring the relevance of responses.
- **Performance Monitoring**: View response times for queries to monitor efficiency.

## Installation

### Prerequisites

- Python 3.7 or higher
- Streamlit
- Groq API key
- HuggingFace API token for embeddings
- **LangChain**, **FAISS**, and other required Python packages installed
- A set of research papers stored in a directory named `research_papers`

### Setup Steps

```bash
1. Clone the repository:
   git clone https://github.com/your-username/research-paper-qa.git
   cd research-paper-qa

2. Install the required dependencies:
   pip install -r requirements.txt

3. Create a `.env` file and add your API keys:
   GROQ_API_KEY=your-groq-api-key
   HF_TOKEN=your-huggingface-token

4. Ensure the `research_papers` directory exists and contains valid PDF documents for processing.

5. Run the Streamlit application:
   streamlit run app.py

   The application will be available at `http://localhost:8501` in your web browser.
```

## Usage

1. **Input Your Query**: Once the app is running, type a query related to your research papers in the input field.
2. **Initialize Vector Database**: If the vector database has not been initialized yet, click the "Initialize Vector Database" button to begin indexing the documents.
3. **Query Processing**: Upon entering a query, the system will retrieve relevant content from the documents and provide an answer based on the content.
4. **View Relevant Documents**: Relevant documents used to generate the response will be displayed under the "Relevant Documents" section.
