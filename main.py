import os
import arxiv
import faiss
from langchain_huggingface import HuggingFaceEmbeddings  # Corrected import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.storage import InMemoryStore
from transformers import pipeline
import numpy as np

# Set Hugging Face API Key (if needed)
os.environ["HF_API_KEY"] = "enter api token"

# Initialize Hugging Face Summarizer (without 'use_auth_token')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")

# FAISS Vector Store Setup (Ensure dimensionality matches the embeddings)
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Example embedding model (768 dimensions)

# Initialize FAISS index with the embedding function
faiss_index = FAISS.from_texts([""], embedding=embedding_function)  # Initialize with an empty text

def store_summary(title, summary):
    # Extract embeddings
    embeddings = embedding_function.embed_documents([summary])

    # Print the structure of embeddings to understand it
    print(f"Embeddings: {embeddings}")  # Inspect the structure of embeddings

    # Convert embeddings list to numpy array (if needed) and check shape
    embedding_array = np.array(embeddings[0])  # Extract the first (and only) embedding for this document
    print(f"Embedding shape: {embedding_array.shape}")  # Now you can access the shape

    # Add the embeddings to FAISS
    faiss_index.add_texts([summary], embeddings=[embedding_array])

def query_research(question):
    retriever = faiss_index.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=summarizer, retriever=retriever)  # Using Hugging Face summarizer here
    return qa_chain.run(question)

# Research Paper Search Agent
def search_papers(query, max_results=3):
    search = arxiv.Client()
    results = search.results(arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance))
    paper_list = []
    for result in results:
        paper_list.append({
            "title": result.title,
            "summary": result.summary,
            "url": result.entry_id
        })
    return paper_list

# Summarization Agent
def summarize_text(text):
    summary = summarizer(text, max_length=200, min_length=50, do_sample=False)
    return summary[0]["summary_text"]

# Main Execution
if __name__ == "__main__":
    user_query = input("Enter your research topic: ")
    papers = search_papers(user_query)
    for paper in papers:
        summary = summarize_text(paper["summary"])
        store_summary(paper["title"], summary)
        print(f"\n📄 Title: {paper['title']}\n🔗 URL: {paper['url']}\n📝 Summary: {summary}\n")

    while True:
        user_question = input("Enter your question about the research (or type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            break
        print("\n🤖 Research Answer:", query_research(user_question))