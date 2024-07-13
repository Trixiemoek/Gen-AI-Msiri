import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from transformers import pipeline
from transformers import logging
import warnings

# Suppress warnings
logging.set_verbosity_error()
warnings.filterwarnings('ignore', category=UserWarning)

# Helper class to simulate the expected structure by text splitter
class Document:
    def __init__(self, text):
        self.page_content = text  # Text of the document
        self.metadata = {}  # Metadata can be extended as needed

    def to_dict(self):
        return {'page_content': self.page_content, 'metadata': self.metadata}

    @staticmethod
    def from_dict(data):
        return Document(data['page_content'])

# Function to clean and verify that the input is a string and replace newlines
def prepare_text(text):
    if isinstance(text, str) and text.strip():
        return text.replace("\n", " ")
    return ""  # Return an empty string if the input is not valid to prevent errors

# Load and prepare documents using pdfplumber
@st.cache(allow_output_mutation=True)
def load_documents(pdf_files):
    documents = []
    for pdf_file in pdf_files:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = prepare_text(page.extract_text())
                if text:
                    documents.append(Document(text).to_dict())  # Convert to dict
    return documents

# Function to split texts into manageable chunks using the custom Document class
@st.cache(allow_output_mutation=True)
def create_splits(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    document_objs = [Document.from_dict(doc) for doc in documents]  # Convert back to Document objects
    splits = text_splitter.split_documents(document_objs)
    return [{'page_content': split.page_content, 'metadata': split.metadata} for split in splits]

# Function to create embeddings and index
@st.cache(allow_output_mutation=True)
def create_embeddings_and_index(splits):
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    splits_content = [split['page_content'] for split in splits]
    embeddings = embedding_model.encode(splits_content)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return embedding_model, index, splits

# Function to query documents based on textual query
def query_documents(embedding_model, index, splits, query, top_k=5):
    query_embedding = embedding_model.encode([prepare_text(query)])
    distances, indices = index.search(query_embedding, top_k)
    return [splits[i] for i in indices[0]]

# Load and prepare documents
pdf_files = [
    "Msiri_one.pdf",
    "Msiri_two_many.pdf"
]

with st.spinner('Loading and processing documents...'):
    documents = load_documents(pdf_files)
    splits = create_splits(documents)
    embedding_model, index, splits = create_embeddings_and_index(splits)

# Define the summarization and question answering chains using Hugging Face Pipelines
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")

class CombineDocsChain:
    def __init__(self, llm):
        self.llm = llm
    def __call__(self, documents):
        combined_text = " ".join([doc['page_content'] for doc in documents])
        return self.llm(combined_text, max_length=512, min_length=30, do_sample=False)[0]['summary_text']

# Streamlit interface
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
                st.markdown("<span style='color:black'>We apologize, but we couldn't generate a response at this time. Please try again later.</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:black'>We apologize, but we couldn't find any relevant information for your query.</span>", unsafe_allow_html=True)
