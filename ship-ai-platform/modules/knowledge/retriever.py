"""두 ChromaDB 컬렉션을 대상으로 문서 적재·검색·삭제를 제공합니다.

서버 부팅 시 모델을 읽지 않습니다. ChromaDB 연결은 필요할 때 생성하고,
임베딩과 선택적 Cross-Encoder는 검색/색인 요청이 들어온 시점에만 로드합니다.
"""

from __future__ import annotations

import logging
import uuid
from threading import Lock
from typing import Any

from config import Config
from modules.knowledge.embedding import encode_query, encode_texts

logger = logging.getLogger(__name__)
PUBLIC_COLLECTION_NAME = "shipbuilding_docs"
INTERNAL_COLLECTION_NAME = "company_internal_docs"
INDEX_VERSION = "RAG_V3"

_STORE = None
_STORE_LOCK = Lock()


class KnowledgeStore:
    """공통 기술 문서와 사내 문서를 분리해 관리하는 벡터 저장소입니다."""

    def __init__(self) -> None:
        """PersistentClient와 두 문서 컬렉션만 초기화합니다."""
        import chromadb

        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        self.public_collection = self.client.get_or_create_collection(
            name=PUBLIC_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "description": "Public shipbuilding and welding standards"},
        )
        self.internal_collection = self.client.get_or_create_collection(
            name=INTERNAL_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "description": "Authenticated internal shipyard documents"},
        )
        self._reranker = None
        self._reranker_lock = Lock()

    def _collection(self, access_level: str):
        """접근 등급에 해당하는 ChromaDB 컬렉션을 선택합니다."""
        normalized = (access_level or "PUBLIC").upper()
        if normalized == "INTERNAL":
            return self.internal_collection
        if normalized == "PUBLIC":
            return self.public_collection
        raise ValueError("access_level은 PUBLIC 또는 INTERNAL이어야 합니다.")

    def counts(self) -> dict[str, int]:
        """공통 기술 문서와 사내 문서의 현재 청크 수를 반환합니다."""
        return {
            "public": int(self.public_collection.count()),
            "internal": int(self.internal_collection.count()),
        }

    def count(self) -> int:
        """공통 기술 문서 컬렉션의 청크 수를 반환합니다."""
        return int(self.public_collection.count())

    def count_internal(self) -> int:
        """사내 문서 컬렉션의 청크 수를 반환합니다."""
        return int(self.internal_collection.count())

    def _get_reranker(self):
        """설정에서 활성화한 경우에만 Cross-Encoder 모델을 지연 로딩합니다."""
        if not Config.RAG_RERANK_ENABLED:
            return None
        if self._reranker is None:
            with self._reranker_lock:
                if self._reranker is None:
                    from sentence_transformers import CrossEncoder

                    logger.info("Cross-Encoder 로딩: %s", Config.RAG_RERANK_MODEL)
                    self._reranker = CrossEncoder(Config.RAG_RERANK_MODEL, max_length=512)
        return self._reranker

    def search(
        self,
        query: str,
        top_k: int | None = None,
        rag_mode: str = "ALL",
    ) -> list[dict[str, Any]]:
        """선택한 컬렉션에서 벡터 검색 후 필요할 때만 재정렬합니다."""
        query = (query or "").strip()
        mode = (rag_mode or "ALL").upper()
        if not query:
            return []
        if mode not in {"ALL", "PUBLIC", "INTERNAL"}:
            raise ValueError("rag_mode는 ALL, PUBLIC, INTERNAL 중 하나여야 합니다.")

        limit = max(1, min(int(top_k or Config.RAG_DEFAULT_TOP_K), 20))
        candidate_limit = max(limit, min(Config.RAG_CANDIDATE_K, 30))

        collections = []
        if mode in {"ALL", "PUBLIC"}:
            collections.append((self.public_collection, "PUBLIC", "공통기술RAG"))
        if mode in {"ALL", "INTERNAL"}:
            collections.append((self.internal_collection, "INTERNAL", "사내문서RAG"))

        # 검색 대상 컬렉션에 문서가 없으면 임베딩 모델을 로드하지 않습니다.
        populated_collections = [
            (collection, access_level, rag_type, int(collection.count()))
            for collection, access_level, rag_type in collections
        ]
        populated_collections = [item for item in populated_collections if item[3] > 0]
        if not populated_collections:
            return []

        query_vector = list(encode_query(query))
        if not query_vector:
            return []

        candidates: dict[str, dict[str, Any]] = {}
        for collection, access_level, rag_type, collection_size in populated_collections:
            result = collection.query(
                query_embeddings=[query_vector],
                n_results=min(candidate_limit, collection_size),
                include=["documents", "metadatas", "distances"],
            )
            documents = (result.get("documents") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            ids = (result.get("ids") or [[]])[0]

            for index, chunk_text in enumerate(documents):
                if not chunk_text:
                    continue
                metadata = metadatas[index] or {}
                stored_access_level = str(metadata.get("access_level") or access_level).upper()
                if stored_access_level != access_level:
                    continue
                document_id = metadata.get("document_id")
                if str(metadata.get("document_status", "ACTIVE")).upper() != "ACTIVE":
                    continue
                if metadata.get("is_active", True) is False:
                    continue
                distance = float(distances[index]) if index < len(distances) else 0.0
                chunk_id = ids[index]
                candidates[f"{access_level}:{chunk_id}"] = {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "chunk_text": str(chunk_text),
                    "document_name": metadata.get("document_name", "이름 없는 문서"),
                    "category": metadata.get("category", "미분류"),
                    "source": metadata.get("source", "출처 미기재"),
                    "revision": metadata.get("revision", "UNKNOWN"),
                    "page": metadata.get("page_start", 1),
                    "page_start": metadata.get("page_start", 1),
                    "page_end": metadata.get("page_end", metadata.get("page_start", 1)),
                    "access_level": access_level,
                    "rag_type": rag_type,
                    "vector_distance": round(distance, 6),
                    "similarity_score": round(max(0.0, min(1.0, 1.0 - distance)), 4),
                }

        results = list(candidates.values())
        if not results:
            return []

        reranker = self._get_reranker()
        if reranker is not None:
            pairs = [[query, item["chunk_text"]] for item in results]
            scores = reranker.predict(pairs, show_progress_bar=False)
            for item, score in zip(results, scores):
                item["rerank_score"] = round(float(score), 5)
            results.sort(key=lambda item: item.get("rerank_score", float("-inf")), reverse=True)
        else:
            results.sort(key=lambda item: item["vector_distance"])

        return results[:limit]

    def replace_document_chunks(
        self,
        *,
        document_id: int,
        document_name: str,
        category: str,
        source: str,
        revision: str,
        file_hash: str,
        chunks: list[dict[str, Any]],
        access_level: str = "PUBLIC",
        document_status: str = "ACTIVE",
    ) -> int:
        """문서의 기존 청크를 교체하고 지정 컬렉션에 새 청크를 일괄 적재합니다."""
        collection = self._collection(access_level)
        valid_chunks = [chunk for chunk in chunks if str(chunk.get("text", "")).strip()]
        if not valid_chunks:
            return 0

        texts = [str(chunk["text"]).strip() for chunk in valid_chunks]
        embeddings = encode_texts(texts)
        document_id = int(document_id)
        collection.delete(where={"document_id": {"$eq": document_id}})
        other_collection = self.internal_collection if collection is self.public_collection else self.public_collection
        other_collection.delete(where={"document_id": {"$eq": document_id}})

        ingest_id = uuid.uuid4().hex[:12]
        ids = [f"DOC{document_id}_CHUNK{index:06d}" for index in range(1, len(texts) + 1)]
        metadatas = []
        for index, chunk in enumerate(valid_chunks, start=1):
            page_start = int(chunk.get("page_start", 1) or 1)
            page_end = int(chunk.get("page_end", page_start) or page_start)
            metadatas.append(
                {
                    "document_id": document_id,
                    "document_name": str(document_name)[:300],
                    "category": str(category or "미분류")[:50],
                    "source": str(source or "출처 미기재")[:300],
                    "revision": str(revision or "UNKNOWN")[:100],
                    "file_hash": str(file_hash),
                    "chunk_index": index,
                    "page_start": page_start,
                    "page_end": page_end,
                    "chunk_type": str(chunk.get("chunk_type", "TEXT"))[:30],
                    "access_level": access_level.upper(),
                    "document_status": document_status.upper(),
                    "is_active": True,
                    "index_version": INDEX_VERSION,
                    "ingest_run_id": ingest_id,
                }
            )

        batch_size = 64
        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )
        return len(texts)

    def is_document_indexed(self, document_id: int, file_hash: str, access_level: str = "PUBLIC") -> bool:
        """문서 ID와 파일 해시가 같은 색인 청크가 있는지 확인합니다."""
        collection = self._collection(access_level)
        result = collection.get(
            where={
                "$and": [
                    {"document_id": {"$eq": int(document_id)}},
                    {"file_hash": {"$eq": str(file_hash)}},
                    {"access_level": {"$eq": access_level.upper()}},
                ]
            },
            include=["metadatas"],
        )
        return bool(result.get("ids"))

    def delete_document_chunks(self, document_id: int, access_level: str = "INTERNAL") -> int:
        """문서 ID에 연결된 청크를 두 컬렉션 모두에서 찾아 제거합니다."""
        self._collection(access_level)  # 접근 등급 입력값을 먼저 검증합니다.
        deleted_count = 0
        for collection in (self.public_collection, self.internal_collection):
            condition = {"document_id": {"$eq": int(document_id)}}
            existing = collection.get(where=condition, include=[])
            count = len(existing.get("ids", []))
            if count:
                collection.delete(where=condition)
                deleted_count += count
        return deleted_count


def get_knowledge_store() -> KnowledgeStore:
    """프로세스 전역 KnowledgeStore 인스턴스를 지연 생성해 반환합니다."""
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = KnowledgeStore()
    return _STORE
