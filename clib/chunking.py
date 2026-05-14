# RAG and LLM
# Retreival
from langchain_text_splitters import MarkdownTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter
from data_cleanner import DataCleanner
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


class Chunking:

    def __init__(self):
        print("Chunking")

    def fix_size_chunking(self, text):
        splitter = CharacterTextSplitter(
            separator="\n",       # Try to split on newlines first
            chunk_size=1000,        # Max size per chunk
            chunk_overlap=0       # Overlap between chunks
        )

        docs = splitter.create_documents(
            [t.page_content if len(text) > 0 else text.page_content for t in text])
        return DataCleanner.clean_docs(docs)

    def overlaped_chunking(self, text):

        #separator="\n",       # Try to split on newlines first

        splitter = CharacterTextSplitter(
            separator="",
            chunk_size=1000,        # Max size per chunk
            chunk_overlap=150       # Overlap between chunks
        )

        docs = splitter.create_documents(
            [t.page_content if len(text) > 0 else text.page_content for t in text])
        return DataCleanner.clean_docs(docs)

    def recursive_chunking(self, text):
        # Instantiate the splitter with custom options
        markdown_splitter = MarkdownTextSplitter(
            chunk_size=1500,      # Max ~100 characters per chunk
            chunk_overlap=50,    # 20-character overlap
            is_separator_regex=True  # Treat separators as regex for proper Markdown splitting
        )
        # Split into documents
        docs = markdown_splitter.create_documents([text])
        return DataCleanner.clean_docs(docs)

    def sememantic_chunking(self, text = "", embedding_model="all-MiniLM-L6-v2"):
        # Specify the model explicitly

        if embedding_model == "gemma300":
            embeddings = OllamaEmbeddings(
                model="gemma-3-0-0", model_type="embedding")

        elif embedding_model == "gemma400":
            embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
        elif embedding_model == "qwen3":
            embeddings = OllamaEmbeddings(
                model="dengcao/Qwen3-Embedding-4B:Q5_K_M")
        else:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2")

        splitter = SemanticChunker(embeddings)
        # chunks = splitter.split_text(text)
        docs = splitter.create_documents(
            [t.page_content if len(text) > 0 else text.page_content for t in text])

        return DataCleanner.clean_docs(docs)

    def store_chunks_to_excel(self, chunks, output_file: str = "chunks"):
        import pandas as pd
        # Sample data
        data = {
            'Chunk Id': [i for i in range(len(chunks))],
            'Chunk_Content': [chunk.page_content for chunk in chunks],
        }

        # Create DataFrame
        df = pd.DataFrame(data)

        # Save to Excel
        df.to_excel(output_file + '.xlsx', index=False)
