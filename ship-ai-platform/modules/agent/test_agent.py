"""
스마트 선박 자율 제조 멀티에이전트 단독 통합 테스트 (test_agent.py)
"""
import os
import sys
import json
from dotenv import load_dotenv

# 루트 경로 인식 보정 (ship-ai-platform 기준)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# modules/agent/test_agent.py
import sys
from pathlib import Path
from modules.knowledge.retriever import KnowledgeStore

store = KnowledgeStore()

print("=" * 60)
print("1. KnowledgeStore 속성 점검")
print("   - persist_directory:", getattr(store, "persist_directory", getattr(store, "db_path", "없음")))
print("   - collection_name  :", getattr(store, "collection_name", "없음"))

# ChromaDB 컬렉션 접근
col = getattr(store, "collection", getattr(store, "_collection", None))
if col is None:
    print("❌ 컬렉션 객체를 찾을 수 없습니다.")
    sys.exit(0)

count = col.count()
print(f"2. ChromaDB 적재 청크 총 개수: {count}건")

if count == 0:
    print("❌ ChromaDB 컬렉션 데이터가 0건입니다. (PDF 인제스천이 안 되어 있거나 DB 경로 불일치)")
    sys.exit(0)

# 데이터가 있다면 1건 샘플 출력
sample = col.peek(limit=1)
print("3. 샘플 문서 메타데이터:")
print("   - metadatas:", sample.get("metadatas", []))
print("   - text 미리보기:", [str(doc)[:100] for doc in sample.get("documents", [])])

# store.search 직접 호출 시 에러 확인
print("4. store.search('선박') 직접 실행 테스트:")
try:
    res = store.search(query="선박", top_k=2)
    print(f"   -> 반환 결과: {len(res)}건")
except Exception as e:
    print(f"   ❌ search() 내부 에러 발생: {e}")
print("=" * 60)