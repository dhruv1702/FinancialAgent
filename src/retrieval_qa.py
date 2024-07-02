from langchain_community.chat_models import ChatOllama
from langchain.chains import create_retrieval_chain, RetrievalQAWithSourcesChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from flashrank import Ranker
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

def create_qa_chain(vector_store, model='llama3'):
    # Initialize the LLM
    llm = ChatOllama(model=model)

    # Initialize the reranker
    reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./cache")
    compressor = FlashrankRerank(client=reranker)#reranker=reranker)
    
    # Create a Contextual Compression Retriever with the Flashrank reranker
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=vector_store.as_retriever()
    )

    # Define the system prompt
    system_prompt = (
        "You are an intelligent financial agent specializing in answering questions grounded in 10-K, 10-Q reports of public companies. "
        "Your task is to provide well-reasoned and concise answers. When answering, follow these guidelines:\n\n"
        "1. **Understand the Question**: Clearly understand the user's question before starting your reasoning process.\n"
        "2. **Retrieve Relevant Context**: Use the provided context to support your answer. If no relevant context is found, inform the user.\n"
        "3. **Chain of Thought Reasoning**: Break down your reasoning process into clear, logical steps. Explain your thought process as follows:\n"
        "    a. Identify the key elements of the question.\n"
        "    b. Retrieve and analyze relevant information from the context.\n"
        "    c. Synthesize the information to form a coherent answer.\n"
        "4. **Grounding and Sources**: Always ground your answers in the context provided. Cite specific parts of the documents (filename and chunk ID) to support your response. If an answer is not found in the context, state that clearly and try to provide a reasonable answer based on your knowledge.\n"
        "5. **Conciseness and Clarity**: Keep your answers concise, using a maximum of four sentences. Ensure clarity and avoid ambiguity.\n"
        "6. **Politeness and Professionalism**: Maintain a polite and professional tone in all your responses.\n"
        "7. **Cite Multiple Sources**: Whenever possible, cite multiple sources in your answer.\n\n"
        "If you don't know the answer, say that you don't know. Here is the retrieved context to help you answer the question:\n\n"
        "{context}"
    )

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    # def format_docs(docs):
    #     return "\n\n".join(doc.page_content for doc in docs)
    
    # rag_chain_from_docs = (
    #     RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    # retrieve_docs = (lambda x: x["input"]) | compression_retriever

    # chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
    #     answer=rag_chain_from_docs
    # )
    # Create the chain that combines documents for the LLM
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # Create the final retrieval chain with the compression retriever
    #chain = RetrievalQAWithSourcesChain(combine_documents_chain=question_answer_chain, retriever=compression_retriever, return_source_documents=True)
    chain = create_retrieval_chain(compression_retriever, question_answer_chain)
    
    ##question_answer_chain = StuffDocumentsChain(llm, prompt)

    # Create the final retrieval chain with the compression retriever
    # chain = RetrievalQAWithSourcesChain(
    #     combine_documents_chain=question_answer_chain,
    #     retriever=compression_retriever,
    #     return_source_documents=True
    # )
    return chain