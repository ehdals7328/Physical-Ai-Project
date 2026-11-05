"""PUBLIC 컬렉션에 잘못 저장된 INTERNAL 청크를 임베딩 재계산 없이 이동합니다."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from config import Config
from modules.knowledge.retriever import PUBLIC_COLLECTION_NAME, INTERNAL_COLLECTION_NAME


def main() -> int:
    """오접근 등급 청크를 미리 확인하고 --confirm 승인 시 사내 컬렉션으로 이동합니다."""
    parser = argparse.ArgumentParser(description="PUBLIC 컬렉션에 들어간 INTERNAL 벡터 이동")
    parser.add_argument("--confirm", action="store_true", help="실제 이동 실행")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
    try:
        public_collection = client.get_collection(PUBLIC_COLLECTION_NAME)
        internal_collection = client.get_or_create_collection(
            name=INTERNAL_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "description": "Authenticated internal shipyard documents"},
        )
    except Exception as exc:
        print(f"컬렉션 접근 실패: {exc}")
        return 2

    candidates = public_collection.get(
        where={"access_level": {"$eq": "INTERNAL"}},
        include=["metadatas"],
    )
    ids = candidates.get("ids", [])
    metadata = candidates.get("metadatas", [])
    print(f"PUBLIC에서 발견한 INTERNAL 청크: {len(ids)}개")
    for record_id, record_metadata in list(zip(ids, metadata))[:20]:
        data = record_metadata or {}
        print(f" - {record_id} | DOCUMENT_ID={data.get('document_id')} | {data.get('document_name', '')}")

    if not args.confirm:
        print("[DRY RUN] 이동하지 않았습니다. 검토 후 --confirm을 추가하세요.")
        return 0

    moved = 0
    for start in range(0, len(ids), 64):
        batch_ids = ids[start:start + 64]
        batch = public_collection.get(
            ids=batch_ids,
            include=["documents", "metadatas", "embeddings"],
        )
        internal_collection.upsert(
            ids=batch.get("ids", []),
            documents=batch.get("documents", []),
            metadatas=batch.get("metadatas", []),
            embeddings=batch.get("embeddings", []),
        )
        public_collection.delete(ids=batch_ids)
        moved += len(batch_ids)
        print(f"진행: {moved}/{len(ids)}개")

    print(f"이동 완료: {moved}개. 임베딩 모델 재계산은 수행하지 않았습니다.")
    print("Oracle DOCUMENT_METADATA는 변경하지 않았습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
