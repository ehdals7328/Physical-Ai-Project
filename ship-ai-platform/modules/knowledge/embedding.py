# 공유 BGE-M3 임베딩 모델 로더 및 배치 인코더.
from __future__ import annotations

import os
from functools import lru_cache
from typing import Sequence

# @lru_cache는 단일 프로세스 내부에서만 유효하기 때문에 만약 Gunicorn 멀티 프로세스 워커 배포 시 메모리 폭증
# 임베딩 모델을 @lru_cache를 통해 프로세스 메모리에 단 1회만 적재해 재사용하는 싱글톤 로더 함수.
@lru_cache(maxsize=1)
def get_embedder():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers가 설치되지 않았습니다. "
            "pip install -r requirements-rag.txt 를 실행하세요."
        ) from exc

    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    device = os.getenv("EMBEDDING_DEVICE", "").strip() or None
    print(f"[Embedding] 모델 로딩: {model_name} (device={device or 'auto'})")
    model = SentenceTransformer(model_name, device=device)
    return model


# 입력된 텍스트 묶음을 배치 단위로 나눠 고차원 벡터로 변환하고 L2 정규화(normalize_embeddings=True)를 거쳐 리스트 형태로 반환하는 일괄 인코딩 함수.
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
