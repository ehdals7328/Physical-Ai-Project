"""
[파일 역할] ChromaDB를 기반으로 다중 RAG(1. 공통 기술/법규 컬렉션, 2. 회사별 사내 내규/보안 컬렉션)를 관리하며,
           불필요한 카테고리 반복 루프를 제거하고 단일 벡터 검색 + 권한 필터링을 수행하는 핵심 모듈입니다.
"""
from __future__ import annotations

import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

import chromadb

from config import Config
from modules.knowledge.embedding import encode_query, encode_texts

INDEX_VERSION = "RAG_V2"
COLLECTION_NAME = Config.CHROMA_GLOBAL_COLLECTION
INTERNAL_COLLECTION_NAME = Config.CHROMA_INTERNAL_COLLECTION


class KnowledgeStore:
    """
    다중 ChromaDB 컬렉션(공통 기술 지식 vs 사내 보안 내규)을 제어하고
    문서 적재 시 무중단 트랜잭션(Blue-Green 교체)을 보장하는 벡터 스토어 매니저 클래스입니다.
    """

    def __init__(self):
        """디스크에 ChromaDB 경로를 생성하고 공통 컬렉션과 사내 보안 컬렉션 2개를 초기화합니다."""
        chroma_path = os.path.abspath(Config.CHROMA_DB_PATH)
        os.makedirs(chroma_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=chroma_path)

        # [컬렉션 1] 공통 제조기술/선급법규/안전보건 (기존 데이터 100% 호환)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "description": "Global shipyard welding standards & laws"},
        )

        # [컬렉션 2] 회사별 사내 내규/로봇 정비 지침/협력사 보안 규정 (다중 RAG 확장용)
        self.internal_collection = self.client.get_or_create_collection(
            name=INTERNAL_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "description": "Company-specific internal rules & ERP policies"},
        )

    def _get_target_collection(self, collection_target: str = "global"):
        """대상 문자열('global' 또는 'internal')에 맞는 ChromaDB 컬렉션 객체를 반환하는 내부 헬퍼 함수입니다."""
        if str(collection_target).lower() == "internal":
            return self.internal_collection
        return self.collection

    def count(self, *, current_index_only: bool = True, collection_target: str = "global") -> int:
        """지정된 컬렉션에 저장된 유효 청크 벡터의 총개수를 반환하는 함수입니다."""
        target_col = self._get_target_collection(collection_target)
        if not current_index_only:
            return int(target_col.count())
        result = target_col.get(
            where={"index_version": {"$eq": INDEX_VERSION}},
            include=["metadatas"],
        )
        return len(result.get("ids", []))

    def is_document_indexed(self, document_id: int, file_hash: str, *, collection_target: str = "global") -> bool:
        """
        특정 PDF 문서가 누락 없이 온전한 세대로 인덱싱되어 있는지 확인하고,
        비정상 종료로 남은 찌꺼기 벡터가 있다면 자동으로 정리하는 정합성 검증 함수입니다.
        """
        target_col = self._get_target_collection(collection_target)
        result = target_col.get(
            where={
                "$and": [
                    {"document_id": {"$eq": int(document_id)}},
                    {"file_hash": {"$eq": str(file_hash)}},
                    {"index_version": {"$eq": INDEX_VERSION}},
                    {"is_active": {"$eq": True}},
                ]
            },
            include=["metadatas"],
        )
        ids = result.get("ids", [])
        metadatas = result.get("metadatas", [])
        if not ids:
            return False

        generations: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for vector_id, metadata in zip(ids, metadatas):
            metadata = metadata or {}
            generation = str(metadata.get("ingest_run_id", "LEGACY"))
            generations[generation].append((vector_id, metadata))

        complete: list[tuple[str, str]] = []
        for generation, rows in generations.items():
            expected = int(rows[0][1].get("expected_chunk_count", -1))
            if expected > 0 and len(rows) == expected:
                indexed_at = max(str(meta.get("indexed_at", "")) for _, meta in rows)
                complete.append((indexed_at, generation))

        if not complete:
            target_col.delete(ids=ids)
            return False

        keep_generation = max(complete)[1]
        stale_ids = [
            vector_id
            for generation, rows in generations.items()
            if generation != keep_generation
            for vector_id, _ in rows
        ]
        if stale_ids:
            target_col.delete(ids=stale_ids)
        return True

    @staticmethod
    def _batches(items: list, size: int):
        """대량 리스트를 지정된 배치 크기로 분할하여 반환하는 제너레이터 함수입니다."""
        for start in range(0, len(items), size):
            yield start, items[start:start + size]

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
        access_level: str = "INTERNAL",
        document_status: str = "ACTIVE",
        company_name: str = "ALL",
        collection_target: str = "global",
    ) -> int:
        """
        신규 문서 청크를 임베딩하여 먼저 적재·검증한 뒤, 구버전 청크를 삭제(Blue-Green 방식)하는
        무중단 벡터 적재 트랜잭션 함수입니다. 사내문서인 경우 internal 컬렉션에도 적재할 수 있습니다.
        """
        if not chunks:
            raise ValueError("청크가 0개입니다. 빈 문서는 인덱싱하지 않습니다.")

        texts = [str(chunk["text"]).strip() for chunk in chunks]
        if any(not text for text in texts):
            raise ValueError("빈 텍스트 청크가 포함되어 있습니다.")

        target_col = self._get_target_collection(collection_target)
        embeddings = encode_texts(texts)
        run_id = uuid.uuid4().hex[:12]
        ids = [f"D{int(document_id)}_H{file_hash[:12]}_R{run_id}_C{i + 1:05d}" for i in range(len(chunks))]
        now = datetime.now(timezone.utc).isoformat()

        metadatas = []
        for idx, chunk in enumerate(chunks):
            metadata = {
                "document_id": int(document_id),
                "document_name": str(document_name)[:300],
                "category": str(category)[:30],
                "source": str(source or "미확인")[:300],
                "revision": str(revision or "UNKNOWN")[:100],
                "file_hash": str(file_hash),
                "chunk_index": int(idx + 1),
                "page_start": int(chunk.get("page_start", 1)),
                "page_end": int(chunk.get("page_end", chunk.get("page_start", 1))),
                "chunk_type": str(chunk.get("chunk_type", "TEXT"))[:30],
                "section_path": str(chunk.get("section_path", ""))[:500],
                "asset_path": str(chunk.get("asset_path", ""))[:1000],
                "access_level": str(access_level or "INTERNAL").upper()[:20],
                "company_name": str(company_name or "ALL")[:100],
                "document_status": str(document_status or "ACTIVE").upper()[:20],
                "is_active": True,
                "index_version": INDEX_VERSION,
                "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
                "indexed_at": now,
                "ingest_run_id": run_id,
                "expected_chunk_count": int(len(chunks)),
            }
            metadatas.append(metadata)

        previous = target_col.get(
            where={"document_id": {"$eq": int(document_id)}},
            include=["metadatas"],
        )
        previous_ids = list(previous.get("ids", []))
        batch_size = int(os.getenv("CHROMA_WRITE_BATCH_SIZE", "64"))

        try:
            for start, _ in self._batches(ids, batch_size):
                end = min(start + batch_size, len(ids))
                target_col.upsert(
                    ids=ids[start:end],
                    embeddings=embeddings[start:end],
                    documents=texts[start:end],
                    metadatas=metadatas[start:end],
                )
        except Exception:
            for start, batch_ids in self._batches(ids, batch_size):
                try:
                    target_col.delete(ids=batch_ids)
                except Exception:
                    pass
            raise

        obsolete_ids = [old_id for old_id in previous_ids if old_id not in set(ids)]
        for start, old_batch in self._batches(obsolete_ids, batch_size):
            target_col.delete(ids=old_batch)

        return len(ids)

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        allowed_access_levels: Iterable[str] | None = ("PUBLIC", "INTERNAL"),
        collection_target: str = "global",
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        [핵심 최적화 완료] 불필요한 카테고리 필터링과 count() I/O를 제거하고,
        LRU 캐싱된 쿼리 임베딩으로 전체 유효 문서에서 가장 유사한 청크를 단 1회에 인출하는 함수입니다.
        (**kwargs를 통해 기존 코드에서 category 인자를 넘겨도 에러 없이 흡수합니다.)
        """
        query = (query or "").strip()
        if not query:
            return []
        top_k = max(1, min(int(top_k), 20))
        target_col = self._get_target_collection(collection_target)

        try:
            clauses: list[dict[str, Any]] = [
                {"index_version": {"$eq": INDEX_VERSION}},
                {"is_active": {"$eq": True}},
                {"document_status": {"$eq": "ACTIVE"}},
            ]
            if allowed_access_levels is not None:
                access = [str(value).upper() for value in allowed_access_levels]
                if not access:
                    return []
                clauses.append({"access_level": {"$in": access}})

            where = {"$and": clauses}
            query_embedding = encode_query(query)
            if not query_embedding:
                return []

            results = target_col.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            print(f"[KnowledgeStore.search 경고] 필터 검색 실패, 기본 검색 폴백 수행: {exc}")
            try:
                query_embedding = encode_query(query)
                results = target_col.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception:
                return []

        ids = (results.get("ids") or [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        formatted: list[dict[str, Any]] = []
        for cid, doc_text, meta, dist in zip(ids, docs, metas, dists):
            meta = meta or {}
            similarity = max(0.0, min(1.0, 1.0 - float(dist if dist is not None else 0.2)))
            formatted.append({
                "id": cid,
                "doc_id": meta.get("document_id", cid),
                "document_id": meta.get("document_id", cid),
                "document_name": meta.get("document_name", "기술 규격서"),
                "category": meta.get("category", "공통규격"),
                "source": meta.get("source", "조선소 기술자료실"),
                "revision": meta.get("revision", "Rev.1"),
                "page_start": meta.get("page_start", 1),
                "page_end": meta.get("page_end", 1),
                "chunk_type": meta.get("chunk_type", "TEXT"),
                "access_level": meta.get("access_level", "INTERNAL"),
                "rag_scope": collection_target.upper(),
                "similarity_score": round(similarity, 4),
                "chunk_text": doc_text or "",
                "metadata": meta,
            })
        return formatted

    def search_multi_rag(
        self,
        query: str,
        *,
        user_company_type: str = "MAIN",
        user_role: str = "WORKER",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        사용자의 소속(원청 MAIN vs 협력사 PARTNER)과 직책에 따라
        [컬렉션 1: 공통 기술/선급법규]와 [컬렉션 2: 사내 보안 내규]를 동시에 조회하여 병합 반환하는 다중 RAG 검색 함수입니다.
        """
        # 협력사(PARTNER) 직원은 PUBLIC 문서만 열람 가능, 원청(MAIN) 직원은 INTERNAL까지 열람 가능
        if str(user_company_type).upper() == "PARTNER":
            allowed_levels = ("PUBLIC",)
        elif str(user_role).upper() in ("EXEC", "ADMIN", "MANAGER"):
            allowed_levels = ("PUBLIC", "INTERNAL", "CONFIDENTIAL")
        else:
            allowed_levels = ("PUBLIC", "INTERNAL")

        # 1. 공통 기술/법규 컬렉션 검색
        global_docs = self.search(
            query=query,
            top_k=top_k,
            allowed_access_levels=allowed_levels,
            collection_target="global",
        )

        # 2. 사내 보안/내규 컬렉션 검색 (원청 소속이거나 PUBLIC 내규만 인출)
        internal_docs = self.search(
            query=query,
            top_k=max(2, top_k // 2),
            allowed_access_levels=allowed_levels,
            collection_target="internal",
        )

        # 유사도 점수 순으로 정렬하여 상위 top_k 반환
        combined = global_docs + internal_docs
        combined.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        return combined[:top_k]