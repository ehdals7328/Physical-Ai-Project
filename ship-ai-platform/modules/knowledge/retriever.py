"""
modules/knowledge/retriever.py
고성능 하이브리드 RAG 검색 및 관리자 문서 생명주기 관리 엔진.
"""
from __future__ import annotations

import os
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from kiwipiepy import Kiwi
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import Config
from modules.knowledge.embedding import encode_query, encode_texts

INDEX_VERSION = "RAG_V2"
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "shipbuilding_docs")
INTERNAL_COLLECTION_NAME = os.getenv("CHROMA_INTERNAL_COLLECTION", "company_internal_docs")

_GLOBAL_STORE_INSTANCE = None
_GLOBAL_RERANKER = None
_GLOBAL_KIWI = None

def get_shared_reranker() -> CrossEncoder:
    global _GLOBAL_RERANKER
    if _GLOBAL_RERANKER is None:
        print("[Reranker] Cross-Encoder 로드: BAAI/bge-reranker-base (최초 1회)")
        _GLOBAL_RERANKER = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
    return _GLOBAL_RERANKER

def get_shared_kiwi() -> Kiwi:
    global _GLOBAL_KIWI
    if _GLOBAL_KIWI is None:
        _GLOBAL_KIWI = Kiwi()
    return _GLOBAL_KIWI

class KnowledgeStore:
    def __init__(self):
        chroma_path = os.path.abspath(Config.CHROMA_DB_PATH)
        os.makedirs(chroma_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=chroma_path)

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        self.internal_collection = self.client.get_or_create_collection(
            name=INTERNAL_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

        self.kiwi = get_shared_kiwi()
        self.reranker = get_shared_reranker()

        self.bm25_corpus: List[List[str]] = []
        self.bm25_chunks: List[Dict[str, Any]] = []
        self.bm25_index: BM25Okapi | None = None
        self._load_or_build_bm25_index()

    def count(self, *, current_index_only: bool = True) -> int:
        return int(self.collection.count())

    def count_internal(self) -> int:
        return int(self.internal_collection.count())

    def _tokenize(self, text: str) -> List[str]:
        tokens = []
        for t in self.kiwi.tokenize(text):
            if t.tag.startswith(('NN', 'SL', 'SN')):
                tokens.append(t.form.lower())
        return tokens or text.lower().split()

    def _load_or_build_bm25_index(self):
        cache_file = Path(Config.CHROMA_DB_PATH) / "bm25_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                    self.bm25_corpus = cached.get("corpus", [])
                    self.bm25_chunks = cached.get("chunks", [])
                    self.bm25_index = cached.get("index")
                print(f"[BM25] 캐시 즉시 로드 완료: {len(self.bm25_chunks)}개 청크 (소요: ~0.05초)")
                return
            except Exception as e:
                print(f"[BM25] 캐시 로드 실패 재빌드: {e}")

        data = self.collection.get(include=["documents", "metadatas"])
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        ids = data.get("ids") or []

        self.bm25_corpus = []
        self.bm25_chunks = []
        for i, text in enumerate(docs):
            cleaned = (text or "").strip()
            if len(cleaned) < 30:
                continue
            meta = metas[i] if i < len(metas) else {}
            tokens = self._tokenize(cleaned)
            self.bm25_corpus.append(tokens)
            self.bm25_chunks.append({
                "chunk_id": ids[i],
                "chunk_text": cleaned,
                "document_name": meta.get("document_name", "기술규격서"),
                "category": meta.get("category", "규격"),
                "source": meta.get("source", "조선소 기술표준"),
                "page": meta.get("page_start", 1),
                "page_start": meta.get("page_start", 1),
                "rag_type": "공통규격RAG"
            })

        if self.bm25_corpus:
            self.bm25_index = BM25Okapi(self.bm25_corpus)
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump({"corpus": self.bm25_corpus, "chunks": self.bm25_chunks, "index": self.bm25_index}, f)
                print(f"[BM25] 인덱스 빌드 및 캐시 저장 완료 ({cache_file.name})")
            except Exception:
                pass

    def search(
        self,
        query: str,
        top_k: int = 4,
        rag_mode: str = "ALL",
        min_confidence_threshold: float = 0.65,
        **kwargs
    ) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []

        candidates_map: Dict[str, Dict[str, Any]] = {}
        q_emb = encode_query(query)

        # 1. Dense Vector 검색
        if q_emb and self.collection.count() > 0 and rag_mode in ("ALL", "PUBLIC"):
            raw_res = self.collection.query(
                query_embeddings=[q_emb],
                n_results=min(15, self.collection.count()),
                include=["documents", "metadatas"]
            )
            for idx, text in enumerate((raw_res.get("documents") or [[]])[0]):
                if len(str(text).strip()) < 30: continue
                cid = (raw_res.get("ids") or [[]])[0][idx]
                meta = (raw_res.get("metadatas") or [[]])[0][idx] or {}
                candidates_map[cid] = {
                    "chunk_id": cid, "chunk_text": text,
                    "document_name": meta.get("document_name", "기술규격서"),
                    "category": meta.get("category", "규격"),
                    "source": meta.get("source", "조선소 기술자료실"),
                    "page": meta.get("page_start", 1), "page_start": meta.get("page_start", 1),
                    "rag_type": "공통규격RAG"
                }

        # 사내 보안내규 검색
        if q_emb and self.internal_collection.count() > 0 and rag_mode in ("ALL", "INTERNAL"):
            int_res = self.internal_collection.query(
                query_embeddings=[q_emb],
                n_results=min(6, self.internal_collection.count()),
                include=["documents", "metadatas"]
            )
            for idx, text in enumerate((int_res.get("documents") or [[]])[0]):
                cid = (int_res.get("ids") or [[]])[0][idx]
                meta = (int_res.get("metadatas") or [[]])[0][idx] or {}
                candidates_map[cid] = {
                    "chunk_id": cid, "chunk_text": text,
                    "document_name": meta.get("document_name", "사내보안내규"),
                    "category": meta.get("category", "사내보안내규"),
                    "source": meta.get("source", "생산기술본부 로봇운영팀"),
                    "page": meta.get("page_start", 1), "page_start": meta.get("page_start", 1),
                    "rag_type": "사내내규RAG"
                }

        # 2. Sparse 키워드 검색 (BM25)
        if self.bm25_index and rag_mode in ("ALL", "PUBLIC"):
            tokens = self._tokenize(query)
            scores = self.bm25_index.get_scores(tokens)
            for idx in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]:
                if scores[idx] > 0.05:
                    c = self.bm25_chunks[idx]
                    if c["chunk_id"] not in candidates_map:
                        candidates_map[c["chunk_id"]] = c

        candidate_list = list(candidates_map.values())
        if not candidate_list:
            return []

        # 3. Cross-Encoder 리랭킹
        pairs = [[query, c["chunk_text"]] for c in candidate_list]
        logits = self.reranker.predict(pairs)

        # 4. 시그모이드 확률 스케일링 & 컷오프
        final_results = []
        for idx, logit in enumerate(logits):
            prob = 1.0 / (1.0 + np.exp(-float(logit)))
            if prob < min_confidence_threshold:
                continue
            item = dict(candidate_list[idx])
            item["similarity_score"] = round(prob, 4)
            final_results.append(item)

        final_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return final_results[:top_k]

    def add_internal_document_chunks(self, document_id: int, document_name: str, category: str, source: str, chunks: list[dict[str, Any]]):
        texts = [c["text"].strip() for c in chunks if c.get("text", "").strip()]
        if not texts: return
        embs = encode_texts(texts)
        ids = [f"INT_D{document_id}_C{i+1:04d}" for i in range(len(texts))]
        metas = [{
            "document_id": document_id, "document_name": document_name, "category": category,
            "source": source, "page_start": c.get("page_start", 1), "index_version": INDEX_VERSION
        } for c in chunks]
        self.internal_collection.upsert(ids=ids, embeddings=embs, documents=texts, metadatas=metas)

    def delete_internal_document_chunks(self, document_id: int):
        res = self.internal_collection.get(where={"document_id": {"$eq": int(document_id)}}, include=[])
        ids = res.get("ids", [])
        if ids:
            self.internal_collection.delete(ids=ids)

def get_knowledge_store() -> KnowledgeStore:
    global _GLOBAL_STORE_INSTANCE
    if _GLOBAL_STORE_INSTANCE is None:
        _GLOBAL_STORE_INSTANCE = KnowledgeStore()
    return _GLOBAL_STORE_INSTANCE
