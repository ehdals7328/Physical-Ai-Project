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

# supervisor 메인 그래프 앱 임포트
from modules.agent.supervisor import ship_agent_app


def run_unit_test(case_num: int, title: str, query: str):
    print(f"\n{'='*75}")
    print(f"▶ [Test Case {case_num}] {title}")
    print(f"  - 질의: \"{query}\"")
    print(f"{'='*75}")

    initial_state = {
        "messages": [],
        "user_query": query,
        "task_type": None,
        "target_categories": [],
        "need_weather": False,
        "extracted_params": {},
        "is_param_complete": None,
        "missing_params": [],
        "search_status": "idle",
        "search_queries": [],
        "retrieved_docs": [],
        "weather_data": None,
        "generated_document": None,
        "final_answer": None,
        "audit_saved": False
    }

    # ⭕ 체크포인터용 thread_id 설정 추가 (케이스별 독립 세션 부여)
    config = {"configurable": {"thread_id": f"test_case_session_{case_num}"}}

    try:
        # invoke 호출 시 config 전달
        result = ship_agent_app.invoke(initial_state, config=config)

        print("\n[에이전트 실행 상태 요약]")
        print(f" • Task Type        : {result.get('task_type')}")
        print(f" • Target Categories: {result.get('target_categories')}")
        print(f" • Search Status    : {result.get('search_status')}")
        print(f" • Retrieved Docs   : {len(result.get('retrieved_docs', []))}건 인출")
        print(f" • Extracted Params : {result.get('extracted_params')}")
        print(f" • Is Param Complete: {result.get('is_param_complete')}")

        print("\n[최종 에이전트 응답]:")
        print(result.get("final_answer"))

        doc = result.get("generated_document")
        if doc:
            print("\n[발행된 공식 지시서/계획서 문서]:")
            print(json.dumps(doc, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"\n❌ [테스트 실패]: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n🚀 [Multi-Agent System] 자율 의도 분류 및 범용 RAG 단독 통합 테스트 시작")

    # 1. 일상 대화 (chat)
    run_unit_test(
        1,
        "일상 대화 및 시스템 기능 안내",
        "안녕하세요! 선박 작업 어시스턴트 기능에 대해 소개해줘."
    )

    # 2. 전문 지식 질의 (knowledge_qa - 산소농도 및 RT 기준)
    run_unit_test(
        2,
        "전문 규격/지침 Q&A (산소농도 및 RT 비파괴검사 기준)",
        "밀폐구역 작업 시 적정 산소농도 기준과 방사선투과(RT) 결함 합격 기준을 알려줘."
    )

    # 3. 제원 누락 문서 생성 (doc_generation - 역질문 유도)
    run_unit_test(
        3,
        "문서 작성 제원 누락 역질문 검증 (하드코딩 디폴트값 차단 확인)",
        "울산 야외 독에서 용접 작업지시서 한 부 작성해줘."
    )

    print(f"\n{'='*75}\n✅ 단독 테스트 완료\n{'='*75}\n")