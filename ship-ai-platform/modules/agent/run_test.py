"""
test_agent.py
Oracle SHIP_ERP / SHIP_USER 실시간 연동 및 Multi-Agent 핵심 기능 단독 통합 테스트
"""
import os
import sys
import json
from dotenv import load_dotenv
from pathlib import Path
# 루트 경로 인식 보정
CURRENT_FILE = Path(__file__).resolve()
PLATFORM_ROOT = CURRENT_FILE.parents[2]

if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))

load_dotenv()

from modules.agent.supervisor import ship_agent_app


def execute_test_case(case_num: int, title: str, state_input: dict, thread_id: str = "test_session_01"):
    print(f"\n{'='*75}")
    print(f"▶ [Test Case {case_num}] {title}")
    print(f"  • 사용자 질의: \"{state_input.get('user_query')}\"")
    print(f"  • 권한 컨텍스트: ERP={state_input.get('can_access_erp')}, RAG={state_input.get('can_access_internal_rag')}, 야드={state_input.get('branch_code')}")
    print(f"{'='*75}")

    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = ship_agent_app.invoke(state_input, config=config)

        print("\n[🎯 실행 상태 결과]")
        print(f" • Task Type        : {result.get('task_type')}")
        print(f" • 누락 제원 목록    : {result.get('missing_params')}")
        print(f" • 외주 전환 필요    : {result.get('is_outsourcing_required')}")
        print(f" • RAG 인출 건수    : {len(result.get('retrieved_docs', []))}건")

        erp_status = result.get("erp_resource_status") or {}
        if erp_status:
            worker_qual = erp_status.get("worker_qualification") or {}
            print(f" • 작업자 자격 적격  : {erp_status.get('worker_qualified')} (유효: {worker_qual.get('is_valid')})")
            print(f" • 배정 자재 Lot     : {erp_status.get('assigned_lot_no')}")
            print(f" • 배차 장비 코드    : {erp_status.get('assigned_vehicle_code')}")

        print("\n[💬 최종 에이전트 답변]:")
        print(result.get("final_answer"))

        doc = result.get("generated_document")
        if doc:
            print("\n[📄 생성된 작업지시서 메타데이터 (WORK_INSTRUCTIONS 규격)]:")
            print(f" • 관리 번호 : {doc.get('work_order_number')}")
            print(f" • 승인 상태 : {doc.get('approval_status')}")
            welding_params = doc.get("welding_params_json") or {}
            print(f" • 용접 궤적 : 시작점={welding_params.get('start_point')}, 끝점={welding_params.get('end_point')}")
            print(f" • 예상 길이 : {welding_params.get('estimated_length_mm')} mm")
            print(f" • 안전거리  : {welding_params.get('safety_clearance_yn')}")

        return result

    except Exception as e:
        print(f"\n❌ [테스트 실패]: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("\n🚀 [Multi-Agent System] Oracle DB 연동 4대 핵심 시나리오 단독 검증 시작")

    # -------------------------------------------------------------------------
    # Case 1: RBAC 권한 인터록 차단 (CAN_ACCESS_ERP = 'N')
    # -------------------------------------------------------------------------
    execute_test_case(
        1,
        "유저 권한 제어 검증 (CAN_ACCESS_ERP='N' 계정의 ERP 재고 조회 차단)",
        {
            "user_query": "창고에 남아있는 강판이랑 용접 와이어 재고 알려줘",
            "user_id": "99",
            "user_role": "WORKER",
            "can_access_erp": "N",
            "can_access_internal_rag": "Y",
            "branch_code": "BR-ULSAN-01",
        },
        thread_id="session_rbac_test"
    )

    # -------------------------------------------------------------------------
    # Case 2: Oracle 실시간 인사 원장(ERP_WORKER_CERT) 표 인출
    # -------------------------------------------------------------------------
    execute_test_case(
        2,
        "사내외 인사 원장(ERP_WORKER_CERT) 실시간 작업자 현황 표 조회",
        {
            "user_query": "등록된 작업자 전체 명단이랑 자격증 상태 보여줘",
            "user_id": "1",
            "user_role": "MANAGER",
            "can_access_erp": "Y",
            "can_access_internal_rag": "Y",
            "branch_code": None,  # 전체 사업장(울산/거제) 전체 작업자 8명 인출
        },
        thread_id="session_worker_test"
    )

    # -------------------------------------------------------------------------
    # Case 3: 정상 발행 (B-104 블록 제원 자동 인출 + 김철수 투입 + 비전 궤적)
    # -------------------------------------------------------------------------
    execute_test_case(
        3,
        "정상 작업지시서 초안 발행 (B-104 블록 제원 자동 인출 + YOLOv11 궤적)",
        {
            "user_query": "오늘 B-104 블록 김철수 반장 투입해서 용접 작업지시서 작성해줘",
            "user_id": "1",
            "user_role": "ENGINEER",
            "can_access_erp": "Y",
            "can_access_internal_rag": "Y",
            "default_workspace": "ZONE-B-BAY1",
            "branch_code": "BR-GEOJE-01",  # 김철수 실제 상주 야드 매핑
        },
        thread_id="session_order_test"
    )

    # -------------------------------------------------------------------------
    # Case 4: 안전 인터록 차단 (선급 자격 만료 작업자 '조객원' 투입 시 HOLD)
    # -------------------------------------------------------------------------
    execute_test_case(
        4,
        "작업자 선급 자격 만료에 따른 발행 원천 차단 (조객원 작업자 투입)",
        {
            "user_query": "B-104 블록에 조객원 작업자 투입해서 작업지시서 써줘",
            "user_id": "1",
            "user_role": "ENGINEER",
            "can_access_erp": "Y",
            "can_access_internal_rag": "Y",
            "default_workspace": "ZONE-A-BAY1",
            "branch_code": "BR-ULSAN-01",
        },
        thread_id="session_expired_test"
    )

    print(f"\n{'='*75}\n✅ 단독 검증 완료\n{'='*75}\n")