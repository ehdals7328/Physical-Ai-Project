#!/usr/bin/env python3
"""
CAD RAG 적재 및 AI Agent 검색 연동 검증 스크립트 (읽기 전용).
1) Oracle DOCUMENT_METADATA 내 CATEGORY_CODE='CAD_DRAWING' 행 검증
2) ChromaDB shipbuilding_docs 내 RAG_V2 + CAD_DRAWING 벡터 수 검증
3) 기존 AI Agent가 실제로 사용하는 retriever_tools.search_knowledge() 호출 테스트
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from web import create_web  # noqa: E402
from web.models.schema import DocumentMetadata  # noqa: E402
from modules.knowledge.retriever import KnowledgeStore, INDEX_VERSION  # noqa: E402
from modules.agent.tools.retriever_tools import search_knowledge  # noqa: E402

TEST_QUERIES = [
    "R/V Gunnerus 갑판 Deck 높이 및 이중저 구조 도면 정보 알려줘",
    "KCS 컨테이너선 프로펠러 KP505 제원과 직경 알려줘",
    "KVLCC2 초대형 유조선 선체 3D CAD 모델 정보 알려줘",
]


def main() -> int:
    print("=" * 68)
    print(" 1단계: Oracle DB (DOCUMENT_METADATA) CAD 도면 등록 현황 점검")
    print("=" * 68)
    web = create_web()
    with web.app_context():
        cad_docs = (
            DocumentMetadata.query
            .filter_by(category_code="CAD_DRAWING")
            .order_by(DocumentMetadata.document_id.desc())
            .all()
        )
        active_completed = [d for d in cad_docs if d.is_active == "Y" and d.ingest_status == "COMPLETED"]
        print(f" - 전체 CAD_DRAWING 행 수: {len(cad_docs)}건")
        print(f" - 활성(Y) + 적재완료(COMPLETED): {len(active_completed)}건")
        for d in active_completed[:10]:
            print(f"   * [ID={d.document_id}] [{d.collection_type}] {d.document_title} -> {d.storage_uri}")

    print("\n" + "=" * 68)
    print(" 2단계: ChromaDB 벡터 스토어 CAD 청크 분포 점검")
    print("=" * 68)
    store = KnowledgeStore()
    res = store.collection.get(
        where={
            "$and": [
                {"index_version": {"$eq": INDEX_VERSION}},
                {"category": {"$eq": "CAD_DRAWING"}},
                {"is_active": {"$eq": True}},
            ]
        },
        include=["metadatas"],
    )
    cad_vec_count = len(res.get("ids") or [])
    print(f" - ChromaDB 활성 CAD_DRAWING 벡터 수 ({INDEX_VERSION}): {cad_vec_count}개")
    if cad_vec_count == 0:
        print(" [경고] 적재된 CAD 벡터가 없습니다. 먼저 `python scripts/ingest_cads.py`를 실행하세요.")
        return 1

    print("\n" + "=" * 68)
    print(" 3단계: AI Agent 실제 검색 경로 (retriever_tools.search_knowledge) 테스트")
    print("=" * 68)
    for q in TEST_QUERIES:
        print(f"\n[질의] {q}")
        hits = search_knowledge(q, top_k=3)
        if not hits:
            print("  -> 검색 결과 없음")
            continue
        for rank, item in enumerate(hits, 1):
            snippet = str(item.get("chunk_text", "")).replace("\n", " ")[:120]
            print(
                f"  {rank}. ({item.get('similarity_score', 0):.3f}) "
                f"[{item.get('category')}] {item.get('document_name')} | {snippet}..."
            )

    print("\n[완료] CAD RAG 적재 및 Agent 검색 연동 검증이 정상 완료되었습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
