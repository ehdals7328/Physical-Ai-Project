"""두 RAG 컬렉션에서 지원하지 않는 구형 인덱스만 안전하게 정리합니다.

미리보기: python scripts/cleanup_legacy_vectors.py
실행:     python scripts/cleanup_legacy_vectors.py --confirm
Flask/Waitress/배치 프로세스를 종료한 뒤 실행하세요.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from modules.knowledge.retriever import PUBLIC_COLLECTION_NAME, INTERNAL_COLLECTION_NAME, INDEX_VERSION

# RAG_V2는 기존 정상 색인일 수 있으므로 명시적인 마이그레이션 전까지 보존합니다.
SUPPORTED_INDEX_VERSIONS = {"RAG_V2", INDEX_VERSION}


def main() -> int:
    """두 RAG 컬렉션의 구형 벡터를 미리 확인하고 명시적 승인 시 삭제합니다."""
    parser = argparse.ArgumentParser(description="두 RAG 컬렉션에서 지원하지 않는 구형 벡터 정리")
    parser.add_argument("--confirm", action="store_true", help="미리보기 대상 벡터 삭제")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=os.path.abspath(Config.CHROMA_DB_PATH))
    total_legacy = 0
    for collection_name in (PUBLIC_COLLECTION_NAME, INTERNAL_COLLECTION_NAME):
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            print(f"컬렉션 없음: {collection_name}")
            continue

        data = collection.get(include=["metadatas"])
        legacy_ids = [
            vector_id
            for vector_id, metadata in zip(data.get("ids", []), data.get("metadatas", []))
            if (metadata or {}).get("index_version") not in SUPPORTED_INDEX_VERSIONS
        ]
        total_legacy += len(legacy_ids)
        print(f"[{collection_name}] 전체={collection.count()} / 유지={collection.count() - len(legacy_ids)} / 정리 대상={len(legacy_ids)}")

        if args.confirm:
            for start in range(0, len(legacy_ids), 500):
                collection.delete(ids=legacy_ids[start:start + 500])
            print(f"[{collection_name}] 삭제 완료: {len(legacy_ids)}개")

    if not args.confirm:
        print(f"[DRY RUN] 총 {total_legacy}개가 정리 후보입니다. 삭제하려면 --confirm을 지정하세요.")
    else:
        print(f"정리 완료: 총 {total_legacy}개 구형 벡터")
        print("Oracle 테이블과 DOCUMENT_METADATA는 변경하지 않았습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
