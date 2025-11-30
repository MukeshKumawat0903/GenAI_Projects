import os
import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from PyPDF2 import PdfReader
from langchain.docstore.document import Document

# Streamlit App Configuration
st.set_page_config(
    page_title="LangChain: Summarize Text from YouTube or Website",
    page_icon="🦜",
)
st.title("🦜 LangChain: Summarize Text from YouTube or Website")
st.subheader("Summarize URL or PDF")

# Sidebar Input for Groq API Key
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

# Validate Groq API Key
if not groq_api_key:
    st.error("Please provide your Groq API Key.")
    st.stop()

# LLM Configuration
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

# Prompt Template
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Input for URL and PDF File
url = st.text_input("Enter a YouTube or Website URL", label_visibility="collapsed")
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# Summarize URL Content
if url:
    if st.button("Summarize URL"):
        if not validators.url(url):
            st.error("Invalid URL. Please enter a valid YouTube or website URL.")
        else:
            try:
                with st.spinner("Loading content and summarizing..."):
                    if "youtube.com" in url.lower():
                        loader = YoutubeLoader.from_youtube_url(url)
                    else:
                        loader = UnstructuredURLLoader(
                            urls=[url],
                            ssl_verify=False,
                            headers={
                                "User-Agent": (
                                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                                )
                            },
                        )
                    docs = loader.load()

                    # Summarize Content
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    summary = chain.run(docs)

                    st.success(summary)
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Summarize PDF Content
if uploaded_file:
    if st.button("Summarize PDF"):
        try:
            with st.spinner("Loading PDF content and summarizing..."):
                pdf_reader = PdfReader(uploaded_file)

                # Extract text page by page
                documents = []
                for page_num, page in enumerate(pdf_reader.pages):
                    page_content = page.extract_text()
                    document = Document(
                        metadata={"source": uploaded_file.name, "page": page_num},
                        page_content=page_content
                    )
                    documents.append(document)

                # Summarize Content
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                summary = chain.run(documents)

                st.success(summary)
        except Exception as e:
            st.error(f"An error occurred: {e}")
