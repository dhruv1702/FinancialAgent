import os
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import MarkdownTextSplitter

def chunk_markdown_files(directory):
    splitter = MarkdownTextSplitter(chunk_size=1500, chunk_overlap=100)
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
            client.table("top50").insert(data).execute()