from typing import Any

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

from backend.src.core.config.infrastructure import EMBEDDING_DIM, RAG_QUERY_TOP_K, RAG_WORKING_DIR
from backend.src.infra.chat.embeddings import Embeddings
from langchain_core.documents import Document


class RAG:
    def __init__(self) -> None:
        embeddings = Embeddings()
        self.retriever = LightRAG(
            working_dir=RAG_WORKING_DIR,
            embedding_func=EmbeddingFunc(
                embedding_dim=EMBEDDING_DIM,
                func=embeddings.embed_query,
            ),
        )

    def query(self, query: str) -> list[Document]:
        raw_data = self.retriever.query_data(
            query=query,
            param=QueryParam(
                top_k=RAG_QUERY_TOP_K,
                enable_rerank=False,
            ),
        )

        return self._clean_rag_output(raw_data)

    def _clean_rag_output(self, raw_data: dict[str, Any]) -> list[Document]:
        documents = []
        chunks = raw_data.get("data", {}).get("chunks", [])
        for chunk in chunks:
            content = chunk.get("content", "")
            file_path = chunk.get("file_path", {})

            doc = Document(
                page_content=content,
                metadata={
                    "file_path": self._clean_source_path(file_path),
                },
            )
            documents.append(doc)

        return documents

    def _clean_source_path(self, file_path: str) -> str:
        if not file_path:
            return file_path

        prefixes_to_remove = ["data/raw/", "data\\raw\\", "./data/raw/"]
        for prefix in prefixes_to_remove:
            if file_path.startswith(prefix):
                return file_path[len(prefix) :]

        return file_path
