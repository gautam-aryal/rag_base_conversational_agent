import pandas as pd
from data_loader import DataLoader
from chunking import Chunking
from retriever import Retriever


import pandas as pd
from langchain.schema import Document  # For proper LangChain compatibility


class UserAccessExl:
    def __init__(self, output_file_path, model_name = "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        self.output_file_path = output_file_path
        self.excel_path = None
        self.chunks = None

    def load_chunks_from_excel(self, excel_path = None):
        self.excel_path = excel_path
        self.chunks = None
        if not self.excel_path:
            raise ValueError("Excel file path not provided.")
        # Read Excel file
        df = pd.read_excel(self.excel_path)

        # Convert to LangChain Document objects
        documents = [
            Document(page_content=str(
                row['Chunk_Content']).strip(), metadata={"row_index": row['Chunk Id']})
            for idx, row in df.iterrows()
        ]

        return documents

    def read_qa_from_excel(self, file_name='qa.xlsx'):
        df = pd.read_excel(file_name)
        self.queries = df['Queries'].tolist()
        self.answers = df['Answers'].tolist()
        self.fix_chunking_references = df['fix_chunking_reference'].tolist()
        self.overlapping_chunking_references = df['overlapping_chunking_reference'].tolist(
        )
        self.semantic_chunking_references = df['semantic_chunking_refernces'].tolist(
        )
        self.recursive_chunkign_refernces = df['recursive_chunking_refernces'].tolist(
        )

        print(len(self.queries), len(self.answers),
              len(self.fix_chunking_references))
        return (
            self.queries,
            self.answers,
            self.fix_chunking_references,
            self.semantic_chunking_references,
            self.recursive_chunkign_refernces
        )

    # Fixsize Chunking

    def fix_chunking_sparse_embed(self, output_path):
        self.chunks = self.load_chunks_from_excel(self.output_file_path + 'overlapped_chunks')
        return self.chunk_retriever(
            retriever_model=Retriever(self.chunks, self.model_name).bm25_retrieval
        )

    def fix_chunking_dense_embed(self):
        self.chunks = Chunking().fix_size_chunking(text)
        return self.chunk_retriever(
            retriever_model=Retriever(self.chunks, self.model_name).dense_retrieval
        )

    def fix_chunking_hyrid_embed(self, text):
        self.chunks = Chunking().fix_size_chunking(text)
        return self.chunk_retriever(
            retriever_model=Retriever(self.chunks, self.model_name).hybrid_retrieval
        )

    # Overalped Chunking
    def overlapped_chunking_sparse_embed(self):
        self.chunks = self.load_chunks_from_excel(self.output_file_path + 'overlapped_chunks.xlsx')
        return self.chunk_retriever(
            retriever_model=Retriever(self.chunks, self.model_name).bm25_retrieval
        )

    def overlapped_chunking_dense_embed(self):
        self.chunks = self.load_chunks_from_excel(self.output_file_path + 'overlapped_chunks.xlsx')
        return self.chunk_retriever(
            retriever_model=Retriever(self.chunks, self.model_name).dense_retrieval
        )

    def overlapped_chunking_hyrid_embed(self):
        self.chunks = self.load_chunks_from_excel(self.output_file_path + 'overlapped_chunks.xlsx')
        return self.chunk_retriever(
            retriever_model=Retriever(self.chunks, self.model_name).hybrid_retrieval
        )

    # Semantic Chunking
    def semantic_chunking_sparse_embed(self):
        self.chunks = self.load_chunks_from_excel(self.output_file_path + 'qwen3_semantic_chunks.xlsx')
        return self.chunk_retriever(
            retriever_model=Retriever(self.chunks, self.model_name).bm25_retrieval
        )

    def semantic_chunking_dense_embed(self):
        self.chunks = self.load_chunks_from_excel(self.output_file_path + 'qwen3_semantic_chunks.xlsx')
        return self.chunk_retriever(
            retriever_model=Retriever(self.chunks, self.model_name).dense_retrieval
        )

    def semantic_chunking_hyrid_embed(self):
        self.chunks = self.load_chunks_from_excel(self.output_file_path + 'qwen3_semantic_chunks.xlsx')
        return self.chunk_retriever(
            retriever_model=Retriever(self.chunks, self.model_name).hybrid_retrieval
        )

    # Recursive Chunking
    def recursive_chunking_sparse_embed(self):
        self.chunks = self.load_chunks_from_excel(self.output_file_path + 'recursive_chunks.xlsx')
        return self.chunk_retriever(
            retriever_model=Retriever(self.chunks, self.model_name).bm25_retrieval
        )

    def recursive_chunking_dense_embed(self):
        self.chunks = self.load_chunks_from_excel(self.output_file_path + 'recursive_chunks.xlsx')
        return self.chunk_retriever(
            retriever_model=Retriever(self.chunks, self.model_name).dense_retrieval
        )

    def recursive_chunking_hyrid_embed(self):
        self.chunks = self.load_chunks_from_excel(self.output_file_path + 'recursive_chunks.xlsx')
        return self.chunk_retriever(
            retriever_model=Retriever(self.chunks, self.model_name).hybrid_retrieval
        )

    def chunk_retriever(self, retriever_model, k=5):
        self.chunks_text_only = [chunk.page_content for chunk in self.chunks]
        all_retrievals = []
        all_retrieved_ids = []
        all_queries = []
        for query in self.queries:
            results = retriever_model(query, k)
            retrived_ids = ",".join([str(self.find_index_or_default(
                result.page_content, default=9999)) for result in results])
            result_in_text = "|||".join(
                [result.page_content for result in results])
            all_retrievals.append(result_in_text)
            all_retrieved_ids.append(retrived_ids)
            all_queries.append(query)

        df = self.store_retrieval_in_excell(
            all_queries,
            all_retrievals,
            all_retrieved_ids
        )
        return df

    def store_retrieval_in_excell(self, queries, retrival, retrived_ids):

        df = pd.DataFrame({
            'query': queries,
            'retrival': retrival,
            'actual_fix_chunks_ids': self.fix_chunking_references,
            'actual_overlapping_chunks_ids': self.overlapping_chunking_references,
            'actual_semantic_chunks_ids': self.semantic_chunking_references,
            'actual_recursive_chunk_ids': self.recursive_chunkign_refernces,
            'retrived_ids': retrived_ids
        })
        df.to_excel('retrival.xlsx', index=False)
        print("Success")
        return df

    def find_index_or_default(self, item, default=9999):
        try:
            return self.chunks_text_only.index(item)
        except ValueError:
            return default
