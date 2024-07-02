import os
from vector_store import SupabaseVectorStore, get_supabase_client
from data_prep import chunk_markdown_files, store_embeddings
from retrieval_qa import create_qa_chain
import streamlit as st
from sentence_transformers import SentenceTransformer

def main():
    # Load environment variables
    # ollama_api_key = os.getenv("OLLAMA_API_KEY")
    supabase_client = get_supabase_client()
    #embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    #embedding_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    embedding_model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)

    # Chunk and store data
    chunked_data = chunk_markdown_files("data")
    store_embeddings(chunked_data, embedding_model, supabase_client)

    # Create QA chain
    vector_store = SupabaseVectorStore(client=supabase_client, embedding_model=embedding_model)
    qa_chain = create_qa_chain(vector_store, model='llama3')

    # Streamlit interface
    st.title("Intelligent Financial Agent")

    st.write("""
        I am an intelligent financial agent. I will try to answer financial questions grounded in the 10K, 10Q reports of public companies. When the answer does not exist in the context, I will declare that it is not found in my SEC data.
    """)

    query = st.text_input("Enter your question about financial reports:")

    if query:
        response = qa_chain.invoke({"input": query})
        answer = response['answer']
        matching_docs = vector_store.similarity_search(query, k=5)
        sources = [(doc.metadata["filename"], doc.metadata["chunk_id"]) for doc in matching_docs]

        st.write(f"Answer: {answer}")
        st.write("Sources:")
        for filename, chunk_id in sources:
            st.write(f"- {filename}, chunk {chunk_id}")

if __name__ == "__main__":
    main()