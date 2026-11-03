"""Delete legacy character-chunk vectors while keeping all RAG_V2 vectors.

Preview (default): python scripts/cleanup_legacy_vectors.py
Execute:          python scripts/cleanup_legacy_vectors.py --confirm
Stop Flask/Waitress/worker processes before cleanup.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from config import Config
from modules.knowledge.retriever import COLLECTION_NAME, INDEX_VERSION


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG_V2가 아닌 구형 ChromaDB 벡터만 정리")
    parser.add_argument("--confirm", action="store_true", help="legacy vector 삭제 실행")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=os.path.abspath(Config.CHROMA_DB_PATH))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"컬렉션이 없습니다: {COLLECTION_NAME}")
        return 0

    data = collection.get(include=["metadatas"])
    legacy_ids = []
    for vector_id, metadata in zip(data.get("ids", []), data.get("metadatas", [])):
        metadata = metadata or {}
        if metadata.get("index_version") != INDEX_VERSION:
            legacy_ids.append(vector_id)

    print(f"전체 벡터: {collection.count()}")
    print(f"유지할 {INDEX_VERSION} 벡터: {collection.count() - len(legacy_ids)}")
    print(f"정리 대상 legacy 벡터: {len(legacy_ids)}")
    if not args.confirm:
        print("[DRY RUN] 삭제하지 않았습니다. 실행하려면 --confirm을 추가하세요.")
        return 0

    for start in range(0, len(legacy_ids), 500):
        collection.delete(ids=legacy_ids[start:start + 500])
    print(f"삭제 완료: {len(legacy_ids)}개 legacy 벡터")
    print("Oracle 테이블 및 DOCUMENT_METADATA는 변경하지 않았습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
