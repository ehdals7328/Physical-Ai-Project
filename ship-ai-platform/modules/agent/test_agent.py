# agent_system/run_test.py
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from .supervisor import ship_agent_app

OUTPUT_DIR = Path("./generated_docs")

def print_separator(title: str):
    print("\n" + "=" * 70)
    print(f"🚀 {title}")
    print("=" * 70)

def execute_stream_turn(turn_label: str, query: str, config: dict = None) -> dict:
    """단일 질의 스트림 실행 및 세부 단계 출력 래퍼"""
    print(f"\n----------------------------------------------------------------------")
    print(f"▶ [{turn_label}] 질의: \"{query}\"")
    print(f"----------------------------------------------------------------------")

    turn_input = {
        "user_query": query,
        "messages": [HumanMessage(content=query)],
        "task_type": None,
        "is_param_complete": None,
        "retrieved_docs": [],
        "generated_document": None,
        "final_answer": None,
    }

    step_count = 1
    final_state = {}

    for output in ship_agent_app.stream(
        turn_input,
        config=config,
        recursion_limit=20,
        stream_mode="updates",
        subgraphs=True,
    ):
        if isinstance(output, tuple):
            subgraph_path, node_updates = output
        else:
            subgraph_path, node_updates = (), output

        for node_name, state_update in (node_updates or {}).items():
            if isinstance(state_update, dict):
                final_state.update(state_update)

            prefix = f"[{'/'.join(subgraph_path)} > {node_name}]" if subgraph_path else f"[{node_name}]"
            print(f"  [{step_count:02d}단계] 👉 {prefix}")

            # 1. 최상위 오케스트레이션
            if node_name == "supervisor":
                print(f"    └─ [Supervisor] 다음 팀 배정: '{state_update.get('next')}'")

            # 2. Team 1 분석팀
            elif node_name == "intent_classifier":
                print(f"    └─ [Team 1 분류] task_type: {state_update.get('task_type')}")
            elif node_name == "parameter_checker":
                print(f"    └─ [Team 1 검증] is_param_complete: {state_update.get('is_param_complete')}")

            # 3. Team 2 검색팀
            elif node_name == "rag_retriever":
                docs = state_update.get("retrieved_docs", [])
                print(f"    └─ [Team 2 RAG] 규격 인출 완료 ({len(docs)}건)")
            elif node_name == "weather_collector":
                weather = state_update.get("weather_info", {})
                print(f"    └─ [Team 2 기상] {weather.get('location')} | 기온: {weather.get('temperature')}℃ | 풍속: {weather.get('wind_speed')}m/s")
            elif node_name == "doc_grader":
                print(f"    └─ [Team 2 Grader] 적합성 판정: {state_update.get('is_docs_relevant')} (재시도: {state_update.get('retry_count', 0)}회)")

            # 4. Team 3 문서생성팀
            elif node_name == "output_validator":
                print(f"    └─ [Team 3 검증] 서식 무결성 통과 및 마크다운 파일 저장")
            elif node_name == "save_audit":
                audit_msg = (state_update or {}).get("audit_result", "기록 완료")
                print(f"    └─ [Team 3 감사] {audit_msg}")

            step_count += 1

    # 최종 요약 출력
    print(f"\n[산출물 결과 요약]")
    doc = final_state.get("generated_document")
    if doc:
        print(f"• 태스크: {final_state.get('task_type')}")
        print(f"• 생성 문서 JSON:\n{json.dumps(doc, indent=2, ensure_ascii=False)}")
    else:
        print(f"• 에이전트 응답:")
        print(final_state.get("final_answer"))

    return final_state

def run_full_system_test():
    """모든 핵심 기능 통합 검증 벤치마크"""
    print_separator("스마트 선박 제조 멀티에이전트 시스템 전체 기능 통합 벤치마크 테스트")

    # =========================================================================
    # Part 1. 멀티턴 & 역질문 & 용접지시서(weld_order) 발행 검증
    # =========================================================================
    print("\n[PART 1] 필수 조건 누락 역질문 및 멀티턴 세션 유지 검증")
    session_config = {"configurable": {"thread_id": "integration_test_session"}}

    # Turn 1: 강종/두께 누락 -> 역질문 유도
    execute_stream_turn(
        turn_label="Turn 1: 필수 정보 누락 질의",
        query="울산 야외 독에서 용접 작업지시서 작성해줘",
        config=session_config
    )

    # Turn 2: 누락 조건 보충 -> 이전 세션(울산)과 결합하여 정상 지시서 발행
    execute_stream_turn(
        turn_label="Turn 2: 누락 조건 보충 답변",
        query="SS400 10T 맞대기로 진행해",
        config=session_config
    )

    # =========================================================================
    # Part 2. 중량물 작업계획서(heavy_plan) 및 강풍 안전 중지 검증
    # =========================================================================
    print("\n[PART 2] 중량물 안전계획서 발행 및 강풍 환경 작업 중지 판정 검증")
    execute_stream_turn(
        turn_label="Case 3: 강풍(풍속 12m/s) 인양 계획 질의",
        query="풍속이 12m/s일 때 크레인 블록 인양 작업 안전 기준 및 계획서 작성해줘",
        config={"configurable": {"thread_id": "heavy_plan_session"}}
    )

    # =========================================================================
    # Part 3. 사내 규격 기반 일반 기술 QA(general_qa) 검증
    # =========================================================================
    print("\n[PART 3] 사내 규격 기반 전문 기술 QA 검증")
    execute_stream_turn(
        turn_label="Case 4: 결함 방지 수칙 전문 기술 질의",
        query="선박 후판 용접 시 언더컷 결함 방지 기준이 뭐야?",
        config={"configurable": {"thread_id": "general_qa_session"}}
    )

    # =========================================================================
    # Part 4. 로컬 물리 디스크 산출물 확인
    # =========================================================================
    print_separator("물리 파일 저장 및 감사 이력 적재 확인")
    if OUTPUT_DIR.exists():
        md_files = list(OUTPUT_DIR.glob("*.md"))
        audit_file = OUTPUT_DIR / "audit_history.log"

        print(f"📁 저장 디렉토리: {OUTPUT_DIR.resolve()}")
        print(f"📄 생성된 마크다운(.md) 문서 수: {len(md_files)}개")
        for f in md_files[-3:]:
            print(f"   └─ {f.name}")

        if audit_file.exists():
            print(f"📋 감사 이력 로그(audit_history.log) 확인 완료 (최근 기록):")
            with audit_file.open("r", encoding="utf-8") as af:
                lines = af.readlines()
                for line in lines[-3:]:
                    print(f"   {line.strip()}")
        else:
            print("⚠️ audit_history.log 파일이 아직 생성되지 않았습니다.")
    else:
        print("⚠️ generated_docs 폴더가 생성되지 않았습니다.")

    print("\n" + "=" * 70)
    print("🎉 전체 통합 벤치마크 테스트 완료")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_full_system_test()