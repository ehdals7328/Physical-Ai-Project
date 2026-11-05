"""
[파일 역할] BAAI/bge-m3 임베딩 모델을 싱글톤으로 메모리에 적재하고,
           반복되는 사용자 질의에 대해 LRU 캐싱을 적용하여 검색 속도를 0.1초 단위로 단축합니다.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Sequence


@lru_cache(maxsize=1)
def get_embedder():
    """
    임베딩 모델(BAAI/bge-m3)을 프로세스 메모리에 단 1회만 로드하여 재사용하는 싱글톤 함수입니다.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers가 설치되지 않았습니다. "
            "pip install sentence-transformers 를 실행하세요."
        ) from exc

    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    device = os.getenv("EMBEDDING_DEVICE", "").strip() or None
    print(f"[Embedding] 모델 메모리 적재 중: {model_name} (device={device or 'auto'})")
    model = SentenceTransformer(model_name, device=device)
    return model


@lru_cache(maxsize=512)
def encode_query(query: str) -> list[float]:
    """
    단일 검색 질의어를 벡터로 변환하며, 최근 512개 질의를 메모리에 캐싱하여
    동일 질의나 멀티턴 재검색 시 모델 추론 시간을 0초로 단축하는 함수입니다.
    """
    cleaned = (query or "").strip()
    if not cleaned:
        return []
    vectors = encode_texts([cleaned], batch_size=1, show_progress_bar=False)
    return vectors[0] if vectors else []


def warmup_embedder() -> None:
    """
    Flask 웹 서버 기동 시점에 모델을 미리 메모리에 올리고 더미 인코딩을 1회 수행하여,
    첫 번째 사용자의 질문 시 발생하는 Cold Start(5~10초 멈춤) 현상을 방지하는 예열 함수입니다.
    """
    try:
        encode_query("조선소 용접 규격 예열 테스트")
        print("[Embedding] BGE-M3 모델 Warm-up 완료 (Cold Start 방지 적용)")
    except Exception as exc:
        print(f"[Embedding] Warm-up 건너뜀: {exc}")


def encode_texts(
    texts: Sequence[str],
    *,
    batch_size: int | None = None,
    show_progress_bar: bool = False,
) -> list[list[float]]:
    """
    다수의 문서 청크 텍스트 리스트를 배치(Batch) 단위로 묶어 정규화된 고차원 벡터로 일괄 변환하는 함수입니다.
    """
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