# ChromaDB의 특정 벡터 컬렉션만 안전하게 날리고 빈 컬렉션으로 재생성

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# CLI 옵션(--confirm, --no-backup)을 검증하고, DB 폴더를 타임스탬프 디렉터리로 백업한 뒤 ChromaDB의 지정 컬렉션을 안전하게 삭제·재생성하는 메인 진입점 함수.
def main() -> int:
    parser = argparse.ArgumentParser(description="ChromaDB shipbuilding_docs 컬렉션 초기화")
    parser.add_argument("--confirm", action="store_true", help="실제 컬렉션 삭제 실행")
    parser.add_argument("--no-backup", action="store_true", help="초기화 전 DB 폴더 백업 생략")
    args = parser.parse_args()

    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from config import Config
        db_path = Path(Config.CHROMA_DB_PATH).resolve()
    except Exception as exc:
        print(f"Config 로드 실패: {exc}")
        return 2

    collection_name = os.getenv("CHROMA_COLLECTION", "shipbuilding_docs")
    if not db_path.exists():
        print(f"ChromaDB 경로가 없습니다: {db_path}")
        return 0

    if not args.confirm:
        print("[DRY RUN] 실제 삭제하지 않았습니다.")
        print(f"대상 경로: {db_path}")
        print(f"대상 컬렉션: {collection_name}")
        print("실행하려면: python scripts/reset_chromadb.py --confirm")
        print("주의: 이 작업은 해당 컬렉션의 모든 벡터·문서·메타데이터를 삭제합니다.")
        print("Oracle USERS / WORK_ORDERS / DOCUMENT_METADATA 등은 삭제하지 않습니다.")
        return 0

    if not args.no_backup:
        backup_root = PROJECT_ROOT / "data" / "backups"
        backup_root.mkdir(parents=True, exist_ok=True)
        backup_path = backup_root / f"chromadb_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            shutil.copytree(db_path, backup_path)
            print(f"백업 완료: {backup_path}")
        except Exception as exc:
            print(f"백업 실패로 삭제를 중단합니다: {exc}")
            return 3

    client = chromadb.PersistentClient(path=str(db_path))
    try:
        listed = client.list_collections()
        existing = {item if isinstance(item, str) else item.name for item in listed}
        if collection_name in existing:
            collection = client.get_collection(collection_name)
            print(f"삭제 대상 벡터 수: {collection.count()}")
            client.delete_collection(name=collection_name)
            print(f"컬렉션 삭제 완료: {collection_name}")
        else:
            print(f"컬렉션이 이미 없습니다: {collection_name}")

        client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "description": "Shipyard welding knowledge RAG"},
        )
        print("빈 컬렉션 생성 완료.")
        print("Oracle 테이블과 PDF 원본/캐시는 변경하지 않았습니다.")
        return 0
    except Exception as exc:
        print(f"초기화 실패: {exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
