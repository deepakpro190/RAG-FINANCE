import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document  # To handle API responses

def create_embeddings(doc_paths=None, api_responses=None, persist_directory="chroma_store"):
    """
    Creates embeddings for given document paths or API responses and stores them in Chroma.

    Args:
        doc_paths (list): List of file paths to documents (PDFs or text files).
        api_responses (list): List of strings (API responses).
        persist_directory (str): Directory to store the Chroma database.
    """
    # Initialize embedding model and Chroma vectorstore
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

    # Process document paths (files)
    if doc_paths:
        for doc_path in doc_paths:
            if doc_path.endswith(".pdf"):
                loader = PyPDFLoader(doc_path)
            elif doc_path.endswith(".txt"):
                loader = TextLoader(doc_path)
            else:
                print(f"Unsupported file type: {doc_path}")
                continue
            
            # Load and split document
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = splitter.split_documents(documents)
            
            # Add embeddings to Chroma
            vectorstore.add_documents(splits)

    # Process API responses
    if api_responses:
        # Convert API response strings into Document objects for processing
        documents = [Document(page_content=resp) for resp in api_responses]
        
        # Split and add documents to Chroma
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(documents)
        
        # Add embeddings to Chroma
        vectorstore.add_documents(splits)

    # Persist embeddings to disk
    vectorstore.persist()
    print(f"Embeddings created and stored in: {persist_directory}")

# Example usage (if running the script standalone)
if __name__ == "__main__":
    doc_paths = ["data/raw/11.txt", "data/raw/22.txt", "data/raw/33.txt"]  # Add paths to your documents
    api_responses = [
        "This is a sample API response containing information about finance.",
        "Here is another API response discussing stock market trends."
    ]
    create_embeddings(doc_paths=doc_paths, api_responses=api_responses)
