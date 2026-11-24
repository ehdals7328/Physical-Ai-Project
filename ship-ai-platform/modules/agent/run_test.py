# modules/agent/run_test.py
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(PROJECT_ROOT / ".env")

import modules.agent.tools.erp_tools as erp_mod
from modules.agent.supervisor import ship_agent_app
from modules.agent.tools.cad_tools import confirm_cad_work_plan
from modules.agent.tools.vision_tools import user_engine

erp_engine = getattr(erp_mod, "erp_engine", None) or getattr(erp_mod, "engine", None)
if erp_engine is None:
    erp_uri = os.getenv("SHIP_ERP_DATABASE_URL") or os.getenv("ORACLE_ERP_DATABASE_URL")
    if erp_uri:
        erp_engine = create_engine(erp_uri, pool_pre_ping=True)

# 재실행 편의를 위한 이전 테스트 데이터 정리 (SHIP_USER + SHIP_ERP 임시 GUN-% 블록)
if user_engine is not None:
    with user_engine.begin() as conn:
        conn.execute(text("DELETE FROM WORK_PLANS WHERE BRANCH_CODE = 'BR-ULSAN-01'"))
        conn.execute(text("DELETE FROM CAD_DRAWINGS WHERE BRANCH_CODE = 'BR-ULSAN-01'"))

if erp_engine is not None:
    with erp_engine.begin() as e_conn:
        e_conn.execute(text("DELETE FROM ERP_BLOCK_MASTER WHERE BLOCK_ID LIKE 'GUN-%'"))

base = str(PROJECT_ROOT / "data/cad_drawings")
config = {"configurable": {"thread_id": "test_team4_session_01"}}

# ==============================================================================
# [테스트 1] CAD 도면 업로드 -> Top Supervisor -> Team 1 -> Team 4 (AI 블록 분할 + APS 일정)
# ==============================================================================
print("\n>>> [테스트 1] CAD 도면 업로드 및 Team 4 AI 블록 분할·APS 일정 편성 실행...")
res1 = ship_agent_app.invoke(
    {
        "user_query": "Gunnerus 갑판 및 이중저 도면 공정 계획 수립해줘",
        "user_id": "admin",
        "user_name": "정관리",
        "user_role": "ADMIN",
        "can_access_erp": "Y",
        "branch_code": "BR-ULSAN-01",
        "default_workspace": "ZONE-A-BAY1",
        "cad_dxf_path": f"{base}/gunnerus/extended/CJOB/Construction Plan Deck and Double Bottom Rev0.dxf",
        "cad_pdf_path": f"{base}/gunnerus/extended/CJOB/Construction Plan Deck and Double Bottom Rev0.pdf",
    },
    config=config,
)

staging = res1.get("cad_staging_result") or {}
print("1-1. 라우팅 task_type:", res1.get("task_type"))
print("1-2. STAGING 상태:", staging.get("status"), "| Gate2:", staging.get("gate2_status"))
print("1-3. AI 도면 구조 해석:", staging.get("drawing_interpretation"))
print("1-4. AI 블록 분할 기준:", staging.get("block_division_criterion"))
print("1-5. AI 분할·편성 블록 목록:")
print(json.dumps(staging.get("work_packages"), ensure_ascii=False, indent=2))

# ==============================================================================
# [테스트 2] 관리자 확정(DB 커밋) + 단기 자재(P1~P2) MRP 점검
# ==============================================================================
if staging.get("status") == "STAGING_SUCCESS":
    print("\n>>> [테스트 2] 공정 계획 확정(DB 커밋) 및 단기 MRP 점검 실행...")
    confirmed = confirm_cad_work_plan(
        drawing_meta=staging["drawing_meta"],
        edited_packages=staging["work_packages"],
        confirm_mode="ACTIVE",
        uploaded_by="admin",
        user_engine=user_engine,
        erp_engine=erp_engine,
    )
    print("2-1. 확정 도면 ID:", confirmed["drawing_id"])
    print("2-2. 단기 MRP 진단:", json.dumps(confirmed["mrp_diagnostics"], ensure_ascii=False, indent=2))
    print("2-3. 선제 PO 초안:", json.dumps(confirmed["proactive_po_drafts"], ensure_ascii=False, indent=2))

# ==============================================================================
# [테스트 3] 동일 세션(thread_id)에서 자연어 질의 -> Team 1 -> Team 4 일일 브리핑
# ==============================================================================
print("\n>>> [테스트 3] 동일 세션 자연어 질의('오늘 작업계획 알려줘') 멀티턴 격리 및 브리핑 확인...")
res3 = ship_agent_app.invoke(
    {
        "user_query": "오늘 울산 야드 작업계획 우선순위 알려줘",
        "user_id": "admin",
        "user_name": "정관리",
        "user_role": "ADMIN",
        "can_access_erp": "Y",
        "branch_code": "BR-ULSAN-01",
        "default_workspace": "ZONE-A-BAY1",
    },
    config=config,
)
print("3-1. 라우팅 task_type:", res3.get("task_type"))
print("3-2. 에이전트 브리핑 응답:\n" + str(res3.get("final_answer")))

# ==============================================================================
# [테스트 4] 잘못된 파일(STL 더미) 업로드 시 Gate 1 Fail-Closed 차단 검증
# ==============================================================================
print("\n>>> [테스트 4] 비구조 파일(Stanford_Bunny.stl) Gate 1 차단 검증...")
res4 = ship_agent_app.invoke(
    {
        "user_query": "이 파일로 작업계획 세워줘",
        "user_id": "admin",
        "can_access_erp": "Y",
        "cad_dxf_path": f"{base}/github_pygem/tests/test_datasets/Stanford_Bunny.stl",
    },
    config={"configurable": {"thread_id": "test_team4_session_02"}},
)
print("4-1. 차단 응답:\n" + str(res4.get("final_answer")))

# ==============================================================================
# [테스트 5] Team 4 P1 블록 -> Team 1~3 자율 작업지시서(DRAFT) 발행 및 소요량 정합성 검증
# ==============================================================================
print("\n>>> [테스트 5] 1순위 블록 작업지시서 자동 편성 실행...")
res5 = ship_agent_app.invoke(
    {
        "user_query": "1순위 블록 작업지시서 바로 편성해서 써줘",
        "user_id": "admin",
        "user_name": "정관리",
        "user_role": "ADMIN",
        "can_access_erp": "Y",
        "branch_code": "BR-ULSAN-01",
        "default_workspace": "ZONE-A-BAY1",
    },
    config=config,
)
doc5 = res5.get("generated_document") or {}
print("5-1. 문서 상태:", doc5.get("status"), "| plan_id:", doc5.get("plan_id"), "| 선점 와이어(kg):", doc5.get("reserve_wire_kg"))
print("5-2. 생성 파일:", doc5.get("files"))
print("5-3. 브리핑 응답:\n" + str(res5.get("final_answer")))