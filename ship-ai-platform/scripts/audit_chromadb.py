"""공통 기술 RAG와 사내 RAG의 인덱스 상태를 읽기 전용으로 진단합니다."""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from config import Config
from modules.knowledge.retriever import PUBLIC_COLLECTION_NAME, INTERNAL_COLLECTION_NAME, INDEX_VERSION


def main() -> int:
    """두 컬렉션의 인덱스 버전·문서·카테고리별 청크 분포를 출력합니다."""
    client = chromadb.PersistentClient(path=os.path.abspath(Config.CHROMA_DB_PATH))
    for collection_name in (PUBLIC_COLLECTION_NAME, INTERNAL_COLLECTION_NAME):
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            print(f"컬렉션 없음: {collection_name}")
            continue

        result = collection.get(include=["metadatas"])
        ids = result.get("ids", [])
        versions = Counter()
        docs = Counter()
        categories = Counter()
        for metadata in result.get("metadatas", []):
            metadata = metadata or {}
            versions[metadata.get("index_version", "LEGACY_OR_UNKNOWN")] += 1
            docs[metadata.get("document_id", "UNKNOWN")] += 1
            categories[metadata.get("category", "UNKNOWN")] += 1

        print(f"\n[{collection_name}]")
        print(f"전체 벡터: {len(ids)}")
        print(f"현재 인덱스({INDEX_VERSION}): {versions.get(INDEX_VERSION, 0)}")
        print(f"이전/기타 인덱스: {len(ids) - versions.get(INDEX_VERSION, 0)}")
        print("인덱스 버전:")
        for key, value in versions.most_common():
            print(f"  {key}: {value}")
        print("문서별 청크 수 - 상위 20:")
        for key, value in docs.most_common(20):
            print(f"  DOCUMENT_ID={key}: {value}")
        if categories:
            print("분류별 청크 수:")
            for key, value in categories.most_common():
                print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
