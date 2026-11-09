"""
modules/knowledge/embedding.py
BGE-M3 싱글톤 임베딩 및 단일 쿼리 LRU 캐싱 모듈.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Sequence

@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    device = os.getenv("EMBEDDING_DEVICE", "").strip() or None
    print(f"[Embedding] 모델 적재: {model_name} (device={device or 'auto'})")
    return SentenceTransformer(model_name, device=device)

def warmup_embedder() -> None:
    model = get_embedder()
    model.encode(["warmup"], normalize_embeddings=True)
    print("[Embedding] BGE-M3 모델 사전 예열(Warm-up) 완료.")

@lru_cache(maxsize=256)
def encode_query(query: str) -> list[float]:
    cleaned = (query or "").strip()
    if not cleaned: return []
    res = encode_texts([cleaned])
    return res[0] if res else []

def encode_texts(texts: Sequence[str], batch_size: int | None = None) -> list[list[float]]:
    if not texts: return []
    import numpy as np
    model = get_embedder()
    batch = batch_size or int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
    vectors = model.encode(list(texts), batch_size=batch, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(vectors, dtype=np.float32).tolist()
