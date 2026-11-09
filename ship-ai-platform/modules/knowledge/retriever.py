"""두 개의 독립 Chroma 컬렉션을 권한 경계와 함께 검색한다.

- shipbuilding_docs: 공개 조선·용접 표준 자료
- company_internal_docs: 회사별 사내 시방서/내부 기술문서
Similarity는 Chroma cosine distance를 1-distance로 환산한 값이며 CrossEncoder logit과 혼동하지 않는다.
"""
from __future__ import annotations
import threading
from typing import Any
import chromadb
from config import Config
from modules.knowledge.embedding import encode_query

_store = None
_lock = threading.Lock()

class KnowledgeStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        self.public = self.client.get_or_create_collection(name=Config.PUBLIC_COLLECTION, metadata={"hnsw:space": "cosine"})
        self.internal = self.client.get_or_create_collection(name=Config.INTERNAL_COLLECTION, metadata={"hnsw:space": "cosine"})

    def counts(self) -> dict[str, int]:
        return {"standards": int(self.public.count()), "internal": int(self.internal.count())}

    @staticmethod
    def _normalise(raw: dict[str, Any], collection_type: str, top_k: int) -> list[dict[str, Any]]:
        docs = (raw.get("documents") or [[]])[0]
        ids = (raw.get("ids") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        items = []
        for idx, text in enumerate(docs):
            if not text:
                continue
            meta = metas[idx] or {}
            distance = float(distances[idx]) if idx < len(distances) and distances[idx] is not None else 1.0
            similarity = max(0.0, min(1.0, 1.0 - distance))
            items.append({
                "chunk_id": ids[idx],
                "document_id": int(meta.get("document_id", 0) or 0),
                "document_name": meta.get("document_name", "자료명 미상"),
                "category": meta.get("category", "GENERAL"),
                "source": meta.get("source", "출처 미상"),
                "revision": meta.get("revision", ""),
                "page": int(meta.get("page_start", 1) or 1),
                "page_start": int(meta.get("page_start", 1) or 1),
                "similarity_score": round(similarity, 5),
                "collection_type": collection_type,
                "chunk_text": str(text)[:12000],
            })
        items.sort(key=lambda x: x["similarity_score"], reverse=True)
        return items[:top_k]

    def _query(self, collection, query: str, top_k: int, where: dict | None = None) -> list[dict[str, Any]]:
        count = int(collection.count())
        if count == 0:
            return []
        vector = encode_query(query)
        if not vector:
            return []
        args = {"query_embeddings": [vector], "n_results": min(max(top_k * 4, top_k), count), "include": ["documents", "metadatas", "distances"]}
        if where:
            args["where"] = where
        return collection.query(**args)

    def search_public(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        top_k = top_k or Config.RAG_TOP_K
        raw = self._query(self.public, query, top_k)
        items = self._normalise(raw, "PUBLIC", top_k)
        return [x for x in items if x["similarity_score"] >= Config.RAG_MIN_SIMILARITY]

    def search_internal(self, query: str, company_code: str | None, is_sys_admin: bool = False, top_k: int | None = None) -> list[dict[str, Any]]:
        top_k = top_k or Config.RAG_TOP_K
        if is_sys_admin:
            raw = self._query(self.internal, query, top_k)
            items = self._normalise(raw, "INTERNAL", top_k)
        else:
            # 회사별 문서와 전체 공통 사내문서를 별도 쿼리해 컬렉션 필터 문법 차이를 피한다.
            items = []
            if company_code:
                raw_company = self._query(self.internal, query, top_k, {"owner_company_code": company_code})
                items.extend(self._normalise(raw_company, "INTERNAL", top_k))
            raw_global = self._query(self.internal, query, top_k, {"owner_company_code": "GLOBAL"})
            items.extend(self._normalise(raw_global, "INTERNAL", top_k))
            dedup = {x["chunk_id"]: x for x in items}
            items = sorted(dedup.values(), key=lambda x: x["similarity_score"], reverse=True)[:top_k]
        return [x for x in items if x["similarity_score"] >= Config.RAG_MIN_SIMILARITY]

    def upsert_document_chunks(self, *, document_id: int, document_name: str, category: str, source: str,
                               revision: str, file_hash: str, chunks: list[dict], collection_type: str,
                               owner_company_code: str | None = None) -> int:
        from modules.knowledge.embedding import encode_texts
        clean_chunks = [c for c in chunks if str(c.get("text", "")).strip()]
        if not clean_chunks:
            return 0
        texts = [str(c["text"]).strip() for c in clean_chunks]
        embeddings = encode_texts(texts)
        collection = self.public if collection_type == "PUBLIC" else self.internal
        prefix = "PUB" if collection_type == "PUBLIC" else "INT"
        ids = [f"{prefix}_D{document_id}_C{i+1:05d}" for i in range(len(texts))]
        company_code = "PUBLIC" if collection_type == "PUBLIC" else (owner_company_code or "GLOBAL")
        metadatas = [{
            "document_id": int(document_id), "document_name": document_name[:300],
            "category": category[:100], "source": (source or "").strip()[:200],
            "revision": (revision or "")[:80], "file_hash": file_hash,
            "page_start": int(c.get("page_start", 1) or 1),
            "owner_company_code": company_code,
        } for c in clean_chunks]
        # 재색인 시 구 버전 청크를 먼저 지우되 다른 문서의 데이터에는 영향이 없게 한다.
        self.delete_document_chunks(document_id, collection_type)
        collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        return len(ids)

    def delete_document_chunks(self, document_id: int, collection_type: str) -> int:
        collection = self.public if collection_type == "PUBLIC" else self.internal
        found = collection.get(where={"document_id": int(document_id)}, include=[])
        ids = found.get("ids", [])
        if ids:
            collection.delete(ids=ids)
        return len(ids)

def get_knowledge_store() -> KnowledgeStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = KnowledgeStore()
    return _store
