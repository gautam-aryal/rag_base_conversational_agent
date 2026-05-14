from langchain.schema import Document
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS, Chroma
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

class Retriever:
  def __init__(self, chunks: list[Document], model_name = "BAAI/bge-large-en-v1.5"):
    self.embedding_model = HuggingFaceEmbeddings(
        model_name=model_name
    )
    self.bm25 = BM25Retriever.from_documents(chunks)
    self.vectorstore = FAISS.from_documents(chunks, self.embedding_model)


  # 1. BM25 Retrieval (keyword-based)
  def bm25_retrieval(self, query: str, k: int = 3):
    """Retrieve documents using BM25 keyword-based search."""
    self.bm25.k = k
    return self.bm25.get_relevant_documents(query)

  #2. Dense Retrieval (FAISS semantic)
  def dense_retrieval(self, query: str, k: int = 3):
    """Retrieve documents using FAISS dense (semantic) retrieval."""
    return self.vectorstore.similarity_search(query, k=k)

  # 3. Hybrid Retrieval (BM25 + FAISS)
  def hybrid_retrieval(self, query: str, k: int = 3):
    """Combine BM25 (keywords) and FAISS (semantic) retrieval."""
    faiss_retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k})
    hybrid = EnsembleRetriever(
        retrievers=[self.bm25, faiss_retriever],
        weights=[0.3, 0.7]  # Adjust balance: 30% keyword, 70% semantic
    )
    return hybrid.get_relevant_documents(query)


  def display_retireved_result(results):
    for i, r in enumerate(results):
      print(f"{i}:", r.page_content)
