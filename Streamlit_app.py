import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from transformers import pipeline
import logging
import warnings
import gc

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper class to simulate the expected structure by text splitter
class Document:
    def __init__(self, text):
        self.page_content = text
        self.metadata = {}

    def to_dict(self):
        return {'page_content': self.page_content, 'metadata': self.metadata}

    @staticmethod
    def from_dict(data):
        return Document(data['page_content'])

# Function to clean and verify that the input is a string and replace newlines
def prepare_text(text):
    if isinstance(text, str) and text.strip():
        return text.replace("\n", " ")
    return ""

# Load and prepare documents using pdfplumber
def load_documents(pdf_files, max_pages_per_pdf=10):
    documents = []
    for pdf_file in pdf_files:
        with pdfplumber.open(pdf_file) as pdf:
            for page_number, page in enumerate(pdf.pages):
                if page_number >= max_pages_per_pdf:
                    break
                text = prepare_text(page.extract_text())
                if text:
                    documents.append(Document(text).to_dict())
                # Free up memory after processing each page
                gc.collect()
        # Free up memory after processing each file
        gc.collect()
    return documents

# Function to split texts into manageable chunks using the custom Document class
def create_splits(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    document_objs = [Document.from_dict(doc) for doc in documents]
    splits = text_splitter.split_documents(document_objs)
    return [{'page_content': split.page_content, 'metadata': split.metadata} for split in splits]

# Function to create embeddings and index
def create_embeddings_and_index(splits, batch_size=4):
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    splits_content = [split['page_content'] for split in splits]
    embeddings = []
    for i in range(0, len(splits_content), batch_size):
        batch = splits_content[i:i + batch_size]
        batch_embeddings = embedding_model.encode(batch)
        embeddings.extend(batch_embeddings)
        gc.collect()  # Clear unused memory
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return embedding_model, index, splits

# Function to query documents based on textual query
def query_documents(embedding_model, index, splits, query, top_k=5):
    query_embedding = embedding_model.encode([prepare_text(query)])
    distances, indices = index.search(query_embedding, top_k)
    return [splits[i] for i in indices[0]]

# Define the summarization chain using Hugging Face Pipelines
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")

class CombineDocsChain:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, documents):
        combined_text = " ".join([doc['page_content'] for doc in documents])
        return self.llm(combined_text, max_length=512, min_length=30, do_sample=False)[0]['summary_text']

# Streamlit app
def main():
    try:
        st.title("MSIRI")

        query = st.text_input("Hello. How may I help?:")
        if query:
            with st.spinner('Processing query...'):
                relevant_docs = query_documents(embedding_model, index, splits, query)
                if relevant_docs:
                    combine_docs_chain = CombineDocsChain(summarization_pipeline)
                    try:
                        summary = combine_docs_chain(relevant_docs)
                        st.markdown(f"<span style='color:black'>{summary}</span>", unsafe_allow_html=True)
                    except Exception as e:
                        logger.error("An error occurred while generating the summary: %s", e)
                        st.markdown("<span style='color:black'>We apologize, but we couldn't generate a response at this time. Please try again later.</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color:black'>We apologize, but we couldn't find any relevant information for your query.</span>", unsafe_allow_html=True)
    except Exception as e:
        logger.error("An error occurred in the main function: %s", e)
        st.error("An unexpected error occurred. Please try again later.")

# Load and prepare documents
pdf_files = [
    "Msiri_one.pdf",
    "Msiri_two_many.pdf"
]

try:
    logger.info("Loading documents...")
    documents = load_documents(pdf_files)
    logger.info("Splitting documents...")
    splits = create_splits(documents)
    logger.info("Creating embeddings and index...")
    embedding_model, index, splits = create_embeddings_and_index(splits)
    logger.info("Documents loaded and processed successfully!")
except Exception as e:
    logger.error("An error occurred while loading and processing documents: %s", e)
    st.error("An error occurred while loading and processing documents. Please check the logs for more details.")

if __name__ == "__main__":
    main()
