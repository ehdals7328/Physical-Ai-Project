
# 인덱스 버전별(RAG_V2 vs 레거시), 카테고리별, 문서별 청크 분포 현황을 터미널에 요약 출력하는 읽기 전용(Read-Only) 진단/모니터링 스크립트
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from config import Config
from modules.knowledge.retriever import COLLECTION_NAME, INDEX_VERSION


# 로컬 ChromaDB에 접속해 메타데이터를 읽어온 뒤 버전·카테고리·문서 ID별 분포를 집계하여 콘솔에 리포팅하는 진단 진입점 함수.
def main() -> int:
    client = chromadb.PersistentClient(path=os.path.abspath(Config.CHROMA_DB_PATH))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"컬렉션이 없습니다: {COLLECTION_NAME}")
        return 0

    result = collection.get(include=["metadatas"])
    ids = result.get("ids", [])
    metadatas = result.get("metadatas", [])
    versions = Counter()
    docs = Counter()
    categories = Counter()

    for metadata in metadatas:
        metadata = metadata or {}
        versions[metadata.get("index_version", "LEGACY_OR_UNKNOWN")] += 1
        docs[metadata.get("document_id", "UNKNOWN")] += 1
        categories[metadata.get("category", "UNKNOWN")] += 1

    print(f"Collection: {COLLECTION_NAME}")
    print(f"전체 벡터: {len(ids)}")
    print(f"검색 대상({INDEX_VERSION}): {versions.get(INDEX_VERSION, 0)}")
    print(f"구형/기타 인덱스: {len(ids) - versions.get(INDEX_VERSION, 0)}")
    print("\n[인덱스 버전]")
    for key, value in versions.most_common():
        print(f"  {key}: {value}")
    print("\n[카테고리]")
    for key, value in categories.most_common():
        print(f"  {key}: {value}")
    print("\n[문서별 청크 수 - 상위 30]")
    for key, value in docs.most_common(30):
        print(f"  DOCUMENT_ID={key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
