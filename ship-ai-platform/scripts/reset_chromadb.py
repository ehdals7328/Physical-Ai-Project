"""대상 ChromaDB 컬렉션을 백업 후 초기화하는 유지보수 도구입니다."""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """선택한 RAG 컬렉션을 미리 확인하고 승인 시 백업 후 초기화합니다."""
    parser = argparse.ArgumentParser(description="ChromaDB RAG 컬렉션 초기화")
    parser.add_argument("--collection", choices=("public", "internal", "all"), default="public")
    parser.add_argument("--confirm", action="store_true", help="실제 컬렉션 삭제 실행")
    parser.add_argument("--no-backup", action="store_true", help="초기화 전 DB 폴더 백업 생략")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from config import Config
    from modules.knowledge.retriever import PUBLIC_COLLECTION_NAME, INTERNAL_COLLECTION_NAME

    db_path = Path(Config.CHROMA_DB_PATH).resolve()
    if not db_path.exists():
        print(f"ChromaDB 경로가 없습니다: {db_path}")
        return 0

    targets = {
        "public": (PUBLIC_COLLECTION_NAME, "공통 기술 문서 RAG"),
        "internal": (INTERNAL_COLLECTION_NAME, "사내 문서 RAG"),
    }
    selected = list(targets) if args.collection == "all" else [args.collection]
    target_names = [targets[key][0] for key in selected]

    if not args.confirm:
        print("[DRY RUN] 실제 삭제하지 않았습니다.")
        print(f"대상 경로: {db_path}")
        print("대상 컬렉션:")
        for key in selected:
            print(f" - {targets[key][0]} ({targets[key][1]})")
        print("실행하려면 --confirm을 추가하세요. 초기화 전 서버와 배치 프로세스를 종료하세요.")
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
        for collection_name in target_names:
            try:
                collection = client.get_collection(collection_name)
                print(f"삭제 대상 벡터 수: {collection.count()} ({collection_name})")
                client.delete_collection(name=collection_name)
            except Exception:
                pass
            description = (
                "Public shipbuilding and welding standards"
                if collection_name == PUBLIC_COLLECTION_NAME
                else "Authenticated internal shipyard documents"
            )
            client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine", "description": description},
            )
            print(f"초기화 완료: {collection_name}")
        print("Oracle 테이블과 원본 PDF/파싱 캐시는 변경하지 않았습니다.")
        return 0
    except Exception as exc:
        print(f"초기화 실패: {exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
