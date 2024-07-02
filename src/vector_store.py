import os
from supabase import create_client, Client
from langchain.vectorstores import VectorStore
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

class SupabaseVectorStore(VectorStore):
    def __init__(self, client: Client, embedding_model: SentenceTransformer):
        self.client = client
        self.embedding_model = embedding_model

    def add_texts(self, texts, metadatas=None):
        embeddings = self.embedding_model.encode(texts)
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            self.client.table("top50").insert({
                "content": text,
                "embedding": embeddings[i].tolist(),
                **metadata
            }).execute()

    @classmethod
    def from_texts(cls, texts, embedding_model, metadatas=None, client=None):
        store = cls(client, embedding_model)
        store.add_texts(texts, metadatas)
        return store

    def similarity_search(self, query, k=5):
        query_embedding = self.embedding_model.encode(query).tolist()
        response = self.client.rpc("match_documents", {"input_embedding": query_embedding}).execute()
        return [
            Document(page_content=item["content"], metadata={"filename": item["filename"], "chunk_id": item["chunk_id"]})
            for item in response.data[:k]
        ]

def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)