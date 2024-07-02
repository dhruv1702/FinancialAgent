import os
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import MarkdownTextSplitter
from supabase import create_client, Client
from edgar import Company, set_identity

# Set your SEC identity
set_identity("Your Name your.email@example.com")

def chunk_markdown_files(directory, chunk_size=1500, chunk_overlap=100):
    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            with open(os.path.join(directory, filename), 'r') as file:
                markdown_content = file.read()
                chunks = splitter.split_text(markdown_content)
                chunked_data[filename] = chunks
    return chunked_data

def store_embeddings(chunked_data, model, client):
    for filename, chunks in chunked_data.items():
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()
            data = {
                "filename": filename,
                "chunk_id": i,
                "content": chunk,
                "embedding": embedding
            }
            client.table("sec_500").insert(data).execute()

def retrieve_and_process_filings(tickers, years, model, client):
    chunked_data = {}
    for ticker in tickers:
        company = Company(ticker)
        filings = company.get_filings(forms=["10-K", "10-Q"], years=years)
        for filing in filings:
            filename = f"{ticker}_{filing.date}.md"
            markdown_content = filing.markdown()
            chunks = chunk_markdown_files(markdown_content)
            chunked_data[filename] = chunks
    store_embeddings(chunked_data, model, client)

def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)

def main():
    # Load environment variables
    supabase_client = get_supabase_client()
    model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
    
    # List of top 50 public companies by ticker symbols
    top_50_companies = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V",
        "UNH", "HD", "PG", "PYPL", "DIS", "MA", "NFLX", "KO", "PEP", "XOM",
        "CSCO", "MRK", "INTC", "T", "PFE", "BA", "WMT", "VZ", "ADBE", "CMCSA",
        "CRM", "ABT", "NKE", "LLY", "ORCL", "TMO", "MCD", "MDT", "NEE", "HON",
        "PM", "AMGN", "COST", "CVX", "TXN", "AVGO", "DHR", "QCOM", "UPS", "BMY"
    ]
    
    # Years to retrieve filings for
    years = range(2013, 2023)

    # Retrieve and process filings
    retrieve_and_process_filings(top_50_companies, years, model, supabase_client)

if __name__ == "__main__":
    main()