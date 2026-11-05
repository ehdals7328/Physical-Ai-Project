"""기존 테스트 실행 중 자동 삽입된 데모 사내 규정 벡터를 선택적으로 제거합니다."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config


def main() -> int:
    """데모 규정 ID를 확인하고 --confirm 옵션이 있을 때만 삭제합니다."""
    parser = argparse.ArgumentParser(description="사내 RAG에 저장된 데모 규정 벡터 정리")
    parser.add_argument("--confirm", action="store_true", help="확인된 데모 ID를 실제 삭제")
    args = parser.parse_args()

    import chromadb

    client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
    try:
        collection = client.get_collection("company_internal_docs")
    except Exception:
        print("사내 문서 컬렉션이 없습니다.")
        return 0

    demo_ids = ["INT_RULE_001", "INT_RULE_002", "INT_RULE_003"]
    existing = set(collection.get(ids=demo_ids, include=[]).get("ids", []))
    print(f"정리 대상 데모 벡터: {len(existing)}개")
    for vector_id in sorted(existing):
        print(f" - {vector_id}")
    if not args.confirm:
        print("미리보기만 수행했습니다. 삭제하려면 --confirm을 추가하세요.")
        return 0
    if existing:
        collection.delete(ids=sorted(existing))
    print(f"삭제 완료: {len(existing)}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
