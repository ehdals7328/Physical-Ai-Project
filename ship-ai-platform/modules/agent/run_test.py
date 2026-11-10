"""
run_test.py
PostgreSQL B2B 협력사 DB 연동 및 분산 트랜잭션 전용 검증 스크립트
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


CURRENT_FILE = Path(__file__).resolve()
PLATFORM_ROOT = CURRENT_FILE.parents[2]

if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))

from supervisor import ship_agent_app
from sqlalchemy import create_engine, text
from config import Config

def execute_b2b_test(case_no: int, title: str, state_input: dict, thread_id: str):
    print("\n" + "=" * 75)
    print(f"▶ [Test Case {case_no}] {title}")
    print(f"  • 사용자 질의: \"{state_input.get('user_query')}\"")
    print(f"  • 권한 컨텍스트: ERP={state_input.get('can_access_erp')}, RAG={state_input.get('can_access_internal_rag')}, 야드={state_input.get('branch_code')}")
    print("=" * 75)

    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        result = ship_agent_app.invoke(state_input, config=config)
        
        print("\n[🎯 실행 상태 결과]")
        print(f" • Task Type        : {result.get('task_type')}")
        print(f" • 외주 전환 필요   : {result.get('is_outsourcing_required')}")
        
        partner_plan = result.get("partner_erp_plan") or {}
        selected = partner_plan.get("selected_partner") or {}
        if selected:
            print(f" • 선정된 협력사    : {selected.get('partner_name')} ({selected.get('partner_id')})")
            print(f" • 조회 출처        : {partner_plan.get('source')}")
        
        print("\n[💬 최종 에이전트 답변]:")
        print(result.get("final_answer"))
        return result
    except Exception as e:
        print(f"\n❌ [실행 중 예외 발생]: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_postgres_sales_orders():
    """PostgreSQL b2b_sales_orders 테이블에 실제 적재된 주문 레코드 물리 검증"""
    pg_uri = Config.SQLALCHEMY_BINDS.get("partner_pg")
    if not pg_uri:
        print("⚠️ PARTNER_PG_DATABASE_URL이 설정되지 않아 PostgreSQL 테이블 조회를 건너뜁니다.")
        return

    print("\n" + "=" * 75)
    print("🔍 [PostgreSQL b2b_sales_orders 테이블 실시간 적재 데이터 검증]")
    try:
        engine = create_engine(pg_uri, pool_pre_ping=True)
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT b2b_order_no, client_company, hq_wo_number, partner_id,
                           order_type, ordered_ref_code, order_qty, allocated_lot_or_id,
                           delivery_yard_zone, order_status, 
                           TO_CHAR(created_at, 'YYYY-MM-DD HH24:MI:SS') AS created_at_str
                    FROM b2b_sales_orders
                    ORDER BY created_at DESC
                    LIMIT 3
                """)
            ).mappings().all()

            if not rows:
                print("   ⚠️ (현재 등록된 B2B 수주 내역이 없습니다)")
            else:
                for idx, r in enumerate(rows, 1):
                    print(f"   [{idx}] 주문번호: {r['b2b_order_no']} | 상태: {r['order_status']}")
                    print(f"       발주처: {r['client_company']} -> 협력사: {r['partner_id']}")
                    print(f"       오더구분: {r['order_type']} | 품목코드: {r['ordered_ref_code']} | 수량: {r['order_qty']}")
                    print(f"       배정Lot/ID: {r['allocated_lot_or_id']} | 입고구역: {r['delivery_yard_zone']}")
                    print(f"       등록일시: {r['created_at_str']}")
    except Exception as e:
        print(f"   ⚠️ PostgreSQL 조회 실패: {e}")
    print("=" * 75)

if __name__ == "__main__":
    print("===========================================================================")
    print("🚀 [Multi-Agent System] 외부 협력사 PostgreSQL B2B 연동 전용 검증 시작")
    print("===========================================================================")

    # -------------------------------------------------------------------------
    # Test Case 1: 대한특수철강 B2B 자재 카탈로그 View 실시간 조회 (v_b2b_daehan_materials)
    # -------------------------------------------------------------------------
    execute_b2b_test(
        case_no=1,
        title="대한특수철강 B2B 공개 자재 카탈로그 실시간 조회 (v_b2b_daehan_materials)",
        state_input={
            "user_query": "사외 협력사(대한특수철강)에서 긴급 납품받을 수 있는 FCAW 용접 와이어 재고랑 단가 알려줘",
            "user_id": "1",
            "user_role": "MANAGER",
            "can_access_erp": "Y",
            "can_access_internal_rag": "Y",
            "branch_code": "BR-ULSAN-01",
        },
        thread_id="test_b2b_mat_view_01"
    )

    # -------------------------------------------------------------------------
    # Test Case 2: 동해블록용접 B2B 가용 외주 인력 View 실시간 조회 (v_b2b_donghae_workers)
    # -------------------------------------------------------------------------
    execute_b2b_test(
        case_no=2,
        title="동해블록용접 B2B 가용 외주 용접사 명단 실시간 조회 (v_b2b_donghae_workers)",
        state_input={
            "user_query": "협력업체(동해블록용접)에 투입 가능한 외주 용접 작업자 명단이랑 일당 단가 보여줘",
            "user_id": "1",
            "user_role": "MANAGER",
            "can_access_erp": "Y",
            "can_access_internal_rag": "Y",
            "branch_code": "BR-ULSAN-01",
        },
        thread_id="test_b2b_wkr_view_02"
    )

    # -------------------------------------------------------------------------
    # Test Case 3: 외주 발주 및 PostgreSQL 수주 창구(b2b_sales_orders) 동시 커밋
    # -------------------------------------------------------------------------
    execute_b2b_test(
        case_no=3,
        title="본사 자재 부족 시 B2B 외주 연계 및 수주 창구(b2b_sales_orders) 자동 발주 커밋",
        state_input={
            "user_query": "B-104 블록 김철수 반장 투입해서 작업지시서 써줘. 본사 와이어 재고가 없으니 외주 협력사로 긴급 발주 연계해줘",
            "user_id": "1",
            "user_role": "MANAGER",
            "can_access_erp": "Y",
            "can_access_internal_rag": "Y",
            "default_workspace": "ZONE-A-BAY1",
            "branch_code": "BR-ULSAN-01",
            "is_action_requested": True,  # 👈 관리자 승인 상태(원장 커밋 실행)
        },
        thread_id="test_b2b_commit_03"
    )

    # 최종 PostgreSQL DB 적재 내역 물리 검증
    check_postgres_sales_orders()

    print("\n===========================================================================")
    print("✅ PostgreSQL B2B 연동 단독 검증 완료")
    print("===========================================================================")