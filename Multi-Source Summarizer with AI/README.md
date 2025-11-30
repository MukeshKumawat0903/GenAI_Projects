# Multi-Source Summarizer with AI

## Overview

**Multi-Source Summarizer with AI** is a Streamlit-based application that leverages the LangChain library and Groq LLM to summarize text content from multiple sources. The app supports:

- Summarizing YouTube video transcripts
- Summarizing content from website URLs
- Summarizing content from uploaded PDF files

By utilizing the power of advanced language models, this tool provides concise and meaningful summaries to enhance productivity and understanding.

## Features

- **Summarize YouTube Videos**: Extract and summarize transcripts from YouTube URLs.
- **Summarize Website Content**: Input any valid website URL to generate a concise summary.
- **Summarize PDFs**: Upload PDF files to extract and summarize textual content.

## Requirements

To use this application, ensure you have the following:

1. **Groq API Key**: Required to access the Groq LLM for summarization.
2. **Python Environment**: Python 3.8 or higher.
3. **Dependencies**: See the installation instructions below.

## Installation

Follow these steps to set up the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/multi-source-summarizer.git
   cd multi-source-summarizer
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Enter Groq API Key**:
   - Input your Groq API Key in the sidebar for authentication.

2. **Summarize a YouTube Video or Website**:
   - Enter a valid YouTube or website URL.
   - Click the "Summarize URL" button to generate the summary.

3. **Summarize a PDF File**:
   - Upload a PDF file using the file uploader.
   - Click the "Summarize PDF" button to generate the summary.

## Project Structure

```
.
├── app.py                 # Main application file
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore file
└── LICENSE                # License file
```

## Dependencies

- **Streamlit**: For building the user interface.
- **LangChain**: To integrate LLMs and process text data.
- **Groq LLM**: For generating summaries.
- **PyPDF2**: To extract content from PDF files.
- **validators**: For validating URL inputs.

## API Configuration

To use the Groq LLM, you need to provide an API key. You can:

- Enter the API key in the Streamlit sidebar.
- Alternatively, set it as an environment variable:
  ```bash
  export GROQ_API_KEY=your_api_key_here
  ```

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [Streamlit](https://streamlit.io/): For the interactive user interface.
- [LangChain](https://www.langchain.com/): For enabling advanced LLM workflows.
- [Groq](https://groq.com/): For the powerful AI model used in summarization.