'''from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os

def retrieve_relevant_content(query, persist_directory="chroma_store", top_k=3, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Retrieves the most relevant content for a query from the Chroma database using HuggingFace embeddings.

    Args:
        query (str): The query string.
        persist_directory (str): Directory where the Chroma database is stored.
        top_k (int): Number of top results to retrieve.
        model_name (str): HuggingFace model name for embedding generation.

    Returns:
        list: List of relevant content and metadata.
    """
    # Load stored embeddings
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

    # Perform similarity search
    results = vectorstore.similarity_search(query, k=top_k)
    return results

# Example usage
if __name__ == "__main__":
    query = "What is machine learning?"
    results = retrieve_relevant_content(query)

    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Content: {result.page_content}")
        print(f"Metadata: {result.metadata}")
'''

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def retrieve_relevant_content(query, persist_directory="chroma_store", top_k=3, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Retrieves the most relevant content for a query from the Chroma database using HuggingFace embeddings.

    Args:
        query (str): The query string.
        persist_directory (str): Directory where the Chroma database is stored.
        top_k (int): Number of top results to retrieve.
        model_name (str): HuggingFace model name for embedding generation.

    Returns:
        list: List of relevant content and metadata.
    """
    # Load stored embeddings
    embedding_model = HuggingFaceEmbeddings(model_name=model_name,model_kwargs={"device": "cuda"})
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

    # Perform similarity search
    results = vectorstore.similarity_search(query, k=top_k)
    return results

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def generate_answer_with_llm(query, relevant_docs, model_name="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"):
    """
    Generates an answer to the query based on the relevant documents using the Mistral-7B model on GPU.

    Args:
        query (str): The input query string.
        relevant_docs (list): List of relevant documents retrieved from the Chroma database.
        model_name (str): Pre-trained LLM model name (default: "mistralai/Mistral-7B-v0.1").

    Returns:
        str: Generated answer.
    """
    # Load Mistral-7B model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")  # Move model to GPU

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token if not already defined

    # Combine relevant document content into a context
    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    # Tokenize input with attention mask and move to GPU
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    input_ids = inputs["input_ids"].to("cuda")  # Move input IDs to GPU
    attention_mask = inputs["attention_mask"].to("cuda")  # Move attention mask to GPU

    # Generate the answer
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )

    # Decode the generated text
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


# Example usage
if __name__ == "__main__":
    query = "What is the MACHINE LEARNING?"

    # Retrieve relevant documents
    results = retrieve_relevant_content(query)

    # Generate an answer using Mistral-7B based on the retrieved documents
    answer = generate_answer_with_llm(query, results)

    print("Generated Answer:", answer)


'''
from transformers import GPT2Tokenizer, GPT2Model
from langchain_community.vectorstores import Chroma
import torch

class Retriever:
    def __init__(self, model_name="gpt2-large", chroma_directory="chroma_store"):
        """
        Initializes the retriever with a GPT-2 model and Chroma database.
        
        Args:
            model_name (str): Name of the Hugging Face model for embeddings.
            chroma_directory (str): Directory where the Chroma database is stored.
        """
        # Initialize the tokenizer and model (GPT-2)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)
        self.vectorstore = Chroma(persist_directory=chroma_directory)

    def embed_query(self, query):
        """
        Encodes the query into an embedding using GPT-2 model.
        
        Args:
            query (str): The input query string.
        
        Returns:
            numpy.ndarray: Query embedding.
        """
        # Tokenize the input query
        encoded_input = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        
        # Get the embeddings from the GPT-2 model
        with torch.no_grad():
            output = self.model(**encoded_input)
        
        # We use the last hidden state to form the query embedding by averaging over the token embeddings
        query_embedding = output.last_hidden_state.mean(dim=1)  # Averaging over all token embeddings
        return query_embedding.numpy()

    def retrieve_relevant_docs(self, query, top_k=3):
        """
        Retrieves the most relevant documents from the Chroma database for a given query.
        
        Args:
            query (str): The input query string.
            top_k (int): Number of top results to retrieve.
        
        Returns:
            list: List of relevant documents with content and metadata.
        """
        query_embedding = self.embed_query(query)
        results = self.vectorstore.similarity_search_by_vector(query_embedding[0], k=top_k)
        return results


def generate_answer(model, tokenizer, query, relevant_docs):
    """
    Generates an answer to the query using a causal language model and relevant documents.
    
    Args:
        model: The pre-loaded causal language model.
        tokenizer: Tokenizer for the LLM.
        query (str): The input query string.
        relevant_docs (list): List of relevant documents retrieved from the Chroma database.
    
    Returns:
        str: Generated answer.
    """
    # Combine relevant document content into a context
    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    # Process the prompt and generate an answer
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=512, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Example Usage
if __name__ == "__main__":
    model_name = "gpt2-large"  # GPT-2 model for causal language generation
    retriever = Retriever(model_name)

    # Ensure that padding token is set
    if retriever.tokenizer.pad_token is None:
        retriever.tokenizer.pad_token = retriever.tokenizer.eos_token  # Setting the padding token to eos_token, which is common

    # Query to match
    query = "What is machine learning?"

    # Retrieve relevant documents
    relevant_docs = retriever.retrieve_relevant_docs(query)

    # Load tokenizer and LLM model for answer generation
    llm_model = GPT2Model.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Generate answer
    answer = generate_answer(llm_model, tokenizer, query, relevant_docs)
    print("Generated Answer:", answer)
    '''