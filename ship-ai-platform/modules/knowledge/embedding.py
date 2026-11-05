"""임베딩 모델을 지연 로딩하고 질의 벡터를 재사용합니다."""

from __future__ import annotations

import logging
from functools import lru_cache
from threading import Lock
from typing import Sequence

from config import Config

logger = logging.getLogger(__name__)
_MODEL_LOCK = Lock()
_MODEL = None


def get_embedder():
    """SentenceTransformer 모델을 프로세스에서 한 번만 생성해 반환합니다."""
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise RuntimeError("sentence-transformers 패키지가 설치되지 않았습니다.") from exc
                logger.info("임베딩 모델 로딩: %s", Config.EMBEDDING_MODEL)
                _MODEL = SentenceTransformer(
                    Config.EMBEDDING_MODEL,
                    device=Config.EMBEDDING_DEVICE,
                )
    return _MODEL


def warmup_embedder() -> None:
    """최초 검색 전 선택적으로 임베딩 모델을 예열합니다."""
    get_embedder().encode(["model warmup"], normalize_embeddings=True, show_progress_bar=False)
    logger.info("임베딩 모델 예열 완료")


@lru_cache(maxsize=512)
def encode_query(query: str) -> tuple[float, ...]:
    """동일한 질의의 임베딩 결과를 LRU 캐시에서 재사용합니다."""
    cleaned = (query or "").strip()
    if not cleaned:
        return ()
    vector = encode_texts([cleaned])[0]
    return tuple(float(value) for value in vector)


def encode_texts(
    texts: Sequence[str],
    *,
    batch_size: int | None = None,
    show_progress_bar: bool = False,
) -> list[list[float]]:
    """텍스트 목록을 정규화된 임베딩 벡터 목록으로 변환합니다."""
    if not texts:
        return []
    model = get_embedder()
    vectors = model.encode(
        list(texts),
        batch_size=batch_size or Config.EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )
    return vectors.tolist()
