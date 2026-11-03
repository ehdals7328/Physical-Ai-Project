# 이 파일은 RAG 시스템에서 텍스트 청크를 ChromaDB에 저장·검색하며,
# 문서 개정이나 시스템 크래시 상황에서도 데이터 정합성을 깨뜨리지 않도록
#  원자적 롤백과 무중단 벡터 교체 배포 방식)를 보장하는 벡터 스토어 트랜잭션 관리자이다
from __future__ import annotations

import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

import chromadb

from config import Config
from modules.knowledge.embedding import encode_texts

INDEX_VERSION = "RAG_V2"
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "shipbuilding_docs")


class KnowledgeStore:
    # 디스크에 ChromaDB 디렉터리를 생성하고 컬렉션(HNSW 코사인 공간)을 초기화하거나 불러오는 생성자.
    def __init__(self):
        chroma_path = os.path.abspath(Config.CHROMA_DB_PATH)
        os.makedirs(chroma_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "description": "Shipyard welding knowledge RAG"},
        )

    # 현재 적용 중인 유효 인덱스 버전(RAG_V2)의 청크 개수(또는 전체 청크 개수)를 반환하는 메서드.
    def count(self, *, current_index_only: bool = True) -> int:
        if not current_index_only:
            return int(self.collection.count())
        result = self.collection.get(
            where={"index_version": {"$eq": INDEX_VERSION}},
            include=["metadatas"],
        )
        return len(result.get("ids", []))

    # 특정 문서가 누락 없이 온전한 세대(Generation)로 적재되었는지 검증하고, 비정상 종료로 남은 파편 벡터는 자동 청소하는 정합성 확인 메서드.
    def is_document_indexed(self, document_id: int, file_hash: str) -> bool:
        result = self.collection.get(
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
            self.collection.delete(ids=ids)
            return False

        keep_generation = max(complete)[1]
        stale_ids = [
            vector_id
            for generation, rows in generations.items()
            if generation != keep_generation
            for vector_id, _ in rows
        ]
        if stale_ids:
            self.collection.delete(ids=stale_ids)
        return True

    # 대량의 텍스트/벡터 리스트를 지정한 크기(batch_size)로 쪼개어 배치 단위로 전달하는 제너레이터 헬퍼.
    @staticmethod
    def _batches(items: list, size: int):
        for start in range(0, len(items), size):
            yield start, items[start:start + size]

    # 신규 버전의 청크들을 먼저 인코딩해 저장·검증한 뒤 구버전 벡터만 안전하게 교체(Blue-Green 방식)하는 무중단 트랜잭션 적재 메서드.
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
    ) -> int:
        if not chunks:
            raise ValueError("청크가 0개입니다. 빈 문서는 인덱싱하지 않습니다.")

        texts = [str(chunk["text"]).strip() for chunk in chunks]
        if any(not text for text in texts):
            raise ValueError("빈 텍스트 청크가 포함되어 있습니다.")

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
                "document_status": str(document_status or "ACTIVE").upper()[:20],
                "is_active": True,
                "index_version": INDEX_VERSION,
                "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
                "indexed_at": now,
                "ingest_run_id": run_id,
                "expected_chunk_count": int(len(chunks)),
            }
            metadatas.append(metadata)

        previous = self.collection.get(
            where={"document_id": {"$eq": int(document_id)}},
            include=["metadatas"],
        )
        previous_ids = list(previous.get("ids", []))
        batch_size = int(os.getenv("CHROMA_WRITE_BATCH_SIZE", "64"))

        try:
            for start, _ in self._batches(ids, batch_size):
                end = min(start + batch_size, len(ids))
                self.collection.upsert(
                    ids=ids[start:end],
                    embeddings=embeddings[start:end],
                    documents=texts[start:end],
                    metadatas=metadatas[start:end],
                )

            verified: set[str] = set()
            for start, batch_ids in self._batches(ids, batch_size):
                response = self.collection.get(ids=batch_ids, include=["metadatas"])
                verified.update(response.get("ids", []))
            missing = set(ids) - verified
            if missing:
                raise RuntimeError(f"ChromaDB 검증 실패: {len(missing)}개 청크가 확인되지 않습니다.")
        except Exception:
            for start, batch_ids in self._batches(ids, batch_size):
                try:
                    self.collection.delete(ids=batch_ids)
                except Exception:
                    pass
            raise

        obsolete_ids = [old_id for old_id in previous_ids if old_id not in set(ids)]
        for start, old_batch in self._batches(obsolete_ids, batch_size):
            self.collection.delete(ids=old_batch)

        return len(ids)

    # 이전 레거시 파이프라인과의 호환을 위해 청크 텍스트를 정규화하고 해시를 생성하여 replace_document_chunks로 넘겨주는 래퍼 메서드.
    def ingest_pdf_chunks(self, document_id, doc_name, category, source, revision, chunks):
        import hashlib
        payload = "\n".join(str(c.get("text", "")) for c in chunks)
        file_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        normalized = [
            {
                "text": c.get("text", ""),
                "page_start": c.get("page", c.get("page_start", 1)),
                "page_end": c.get("page", c.get("page_end", 1)),
                "chunk_type": c.get("chunk_type", "TEXT"),
                "asset_path": c.get("asset_path", ""),
                "section_path": c.get("section_path", ""),
            }
            for c in chunks
        ]
        return self.replace_document_chunks(
            document_id=document_id,
            document_name=doc_name,
            category=category,
            source=source,
            revision=revision,
            file_hash=file_hash,
            chunks=normalized,
        )

    # 단일 텍스트 문자열을 1페이지짜리 단일 청크로 감싸서 적재할 수 있도록 지원하는 테스트/단문용 래퍼 메서드.
    def ingest_sample_text(self, document_id, doc_name, category, source, revision, text):
        return self.ingest_pdf_chunks(
            document_id, doc_name, category, source, revision,
            [{"page": 1, "text": text, "chunk_type": "TEXT"}],
        )

    # 사용자 질문을 벡터화하여 유효 인덱스 버전(RAG_V2), 활성 상태, 권한 수준(allowed_access_levels) 조건에 맞는 상위 유사도 청크들을 검색하는 메서드.
    def search(
        self,
        query: str,
        category: str | None = None,
        top_k: int = 5,
        *,
        allowed_access_levels: Iterable[str] | None = ("PUBLIC", "INTERNAL"),
    ) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        top_k = max(1, min(int(top_k), 20))

        try:
            if self.collection.count() == 0:
                return []
            clauses: list[dict[str, Any]] = [
                {"index_version": {"$eq": INDEX_VERSION}},
                {"is_active": {"$eq": True}},
                {"document_status": {"$eq": "ACTIVE"}},
            ]
            if category:
                clauses.append({"category": {"$eq": str(category)}})
            if allowed_access_levels is not None:
                access = [str(value).upper() for value in allowed_access_levels]
                if not access:
                    return []
                clauses.append({"access_level": {"$in": access}})

            where = {"$and": clauses}
            query_embedding = encode_texts([query])[0]
            response = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            ids = (response.get("ids") or [[]])[0]
            documents = (response.get("documents") or [[]])[0]
            metadatas = (response.get("metadatas") or [[]])[0]
            distances = (response.get("distances") or [[]])[0]

            items: list[dict[str, Any]] = []
            min_similarity = float(os.getenv("RAG_MIN_SIMILARITY", "0.0"))
            for idx, text in enumerate(documents):
                meta = metadatas[idx] or {}
                distance = float(distances[idx]) if idx < len(distances) and distances[idx] is not None else 1.0
                similarity = max(0.0, min(1.0, 1.0 - distance))
                if similarity < min_similarity:
                    continue
                items.append({
                    "chunk_id": ids[idx] if idx < len(ids) else None,
                    "document_id": meta.get("document_id"),
                    "document_name": meta.get("document_name"),
                    "category": meta.get("category"),
                    "source": meta.get("source"),
                    "revision": meta.get("revision"),
                    "chunk_text": text,
                    "similarity_score": round(similarity, 5),
                    "page": meta.get("page_start"),
                    "page_start": meta.get("page_start"),
                    "page_end": meta.get("page_end"),
                    "chunk_type": meta.get("chunk_type", "TEXT"),
                    "section_path": meta.get("section_path", ""),
                    "asset_path": meta.get("asset_path", ""),
                    "file_hash": meta.get("file_hash"),
                })
            return items
        except Exception as exc:
            raise RuntimeError(f"RAG 검색 실패: {exc}") from exc
