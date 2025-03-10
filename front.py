import streamlit as st
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import requests

# --- Import API Keys and Config ---
from config import (
    NEWS_API_KEY,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    StockAPIClient,
    SEC_BASE_URL,
)

# --- Function Definitions ---

def fetch_reddit_posts(query, limit=10):
    try:
        url = f"https://www.reddit.com/search.json?q={query}&limit={limit}"
        headers = {"User-Agent": REDDIT_USER_AGENT}
        auth = (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
        response = requests.get(url, headers=headers, auth=auth)
        response.raise_for_status()
        data = response.json()
        return [
            f"Title: {item['data']['title']}\nContent: {item['data']['selftext']}" 
            for item in data.get("data", {}).get("children", [])
        ]
    except Exception as e:
        st.error(f"Error fetching Reddit data: {e}")
        return []


def fetch_sec_filings(company, limit=5):
    try:
        url = f"{SEC_BASE_URL}/company/{company}/filings?limit={limit}"
        response = requests.get(url)
        response.raise_for_status()
        filings = response.json()
        return [f"Filing: {filing['title']} Date: {filing['date']}" for filing in filings]
    except Exception as e:
        st.error(f"Error fetching SEC data: {e}")
        return []


def fetch_news_articles(query, limit=5):
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&pageSize={limit}&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        return [f"Title: {article['title']} Source: {article['source']['name']}" for article in articles]
    except Exception as e:
        st.error(f"Error fetching news data: {e}")
        return []

@st.cache_resource
def load_chroma_and_embeddings(persist_directory="chroma_store", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"})
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectorstore, embeddings

@st.cache_resource
def load_llm(model_name="meta-llama/Llama-3.2-1B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return tokenizer, model

# Summarizer Function
def summarize_text(text, max_length=20000):
    from transformers import AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    if len(tokenizer(text)["input_ids"]) > tokenizer.model_max_length:
        text_chunks = split_text(text, tokenizer.model_max_length)
        summarized_chunks = [
            summarizer(chunk, max_length=100, min_length=50, do_sample=False)
            for chunk in text_chunks
        ]
        summarized = " ".join([chunk[0]["summary_text"] for chunk in summarized_chunks])
    else:
        summarized = text
    return summarized


# Truncate Function for Context
def truncate_text(text, max_length=131072):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    return tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)

# Generate Answer using LLM
def generate_answer_with_llm(query, context, tokenizer, model, max_length=50123):
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(model.device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=3,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Streamlit UI ---
st.set_page_config(layout="wide")

with st.container():
    col1, col2 = st.columns([2, 3])
    with col1:
        st.header("Chat History")
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        for i, message in enumerate(st.session_state.history):
            preview = message['text'][:30] + ("..." if len(message['text']) > 30 else "")
            if st.button(f"Chat {i+1}: {preview}", key=f"chat-{i}"):
                st.session_state.current_chat = st.session_state.history[i]
                st.session_state.history = st.session_state.history[:i+1]
        
        if 'current_chat' in st.session_state:
            st.subheader(f"Chat {i+1} - Full Conversation")
            for message in st.session_state.current_chat:
                st.write(f"**{message['role']}**: {message['text']}")
    
    with col2:
        st.header("Current Chat")
        vectorstore, embeddings = load_chroma_and_embeddings()
        tokenizer, model = load_llm()
        query = st.text_area("Enter your finance-related query:", height=200)
        
        if st.button("Get Answer"):
            if query:
                with st.spinner("Processing..."):
                    reddit_data = fetch_reddit_posts(query)
                    news_data = fetch_news_articles(query)
                    sec_data = fetch_sec_filings(query, limit=3)
                    
                    documents = reddit_data + news_data + sec_data
                    if documents:
                        embeddings_list = embeddings.embed_documents(documents)
                        for doc, emb in zip(documents, embeddings_list):
                            vectorstore.add_texts([doc], embeddings=[emb])
                    
                        search_results = vectorstore.similarity_search(query, k=5)
                        context = "\n".join([result.page_content for result in search_results])
                        
                        if len(context.split()) > 500:
                            context = summarize_text(context)
                        else:
                            context = truncate_text(context)
                        
                        answer = generate_answer_with_llm(query, context, tokenizer, model)
                        st.session_state.history.append({"role": "User", "text": query})
                        st.session_state.history.append({"role": "Bot", "text": answer})
                        st.write(answer)
                    else:
                        st.warning("No relevant data found.")
            else:
                st.warning("Please enter a query to proceed.")

st.header("Stock Price Lookup")
stock_symbol = st.text_input("Enter a stock symbol (e.g., AAPL):")
if stock_symbol:
    with st.spinner("Fetching stock data..."):
        try:
            stock_data = fetch_stock_data(stock_symbol)
            st.write(stock_data)
        except Exception as e:
            st.error(f"Error fetching stock data: {e}")
else:
    st.warning("Please enter a stock symbol to retrieve data.")
