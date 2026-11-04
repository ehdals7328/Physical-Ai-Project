from __future__ import annotations

import os
from functools import lru_cache
from typing import Sequence

@lru_cache(maxsize=1)
def get_embedder():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers 미설치") from exc

    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    device = os.getenv("EMBEDDING_DEVICE", "").strip() or None
    print(f"[Embedding] 모델 로딩: {model_name} (device={device or 'auto'})")
    model = SentenceTransformer(model_name, device=device)
    return model

def warmup_embedder() -> None:
    """서버 구동 시점에 사전 호출하여 Cold-Start 지연을 제거"""
    model = get_embedder()
    model.encode(["warmup_query"], normalize_embeddings=True)
    print("[Embedding] BGE-M3 모델 사전 예열(Warm-up) 완료.")

@lru_cache(maxsize=256)
def encode_query(query: str) -> list[float]:
    """동일 질의어에 대해 반복 임베딩 연산 방지 (LRU 캐시 적용)"""
    cleaned = (query or "").strip()
    if not cleaned:
        return []
    res = encode_texts([cleaned], show_progress_bar=False)
    return res[0] if res else []

def encode_texts(
    texts: Sequence[str],
    *,
    batch_size: int | None = None,
    show_progress_bar: bool = False,
):
    if not texts:
        return []
    import numpy as np

    model = get_embedder()
    effective_batch = batch_size or int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
    vectors = model.encode(
        list(texts),
        batch_size=effective_batch,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors.tolist()