"""
modules/agent/state.py
Agentic AI + SHIP_USER(웹/비전) & SHIP_ERP(전사 자원) 통합 상태 구조체 (ShipAgentState)
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

TaskType = Literal[
    "chat",             # 일상 대화 및 인사
    "erp_resource_qa",  # 사내 자재(ERP_BOM_INVENTORY), 강판 Lot, 와이어 잔량, 대기 차량 확인
    "hr_safety_qa",     # 특정 작업자 선급(KR/AWS) 자격증 단건 유효기간 확인
    "hr_worker_qa",     # 사내/하청/프리랜서 작업자 전체 명단 실시간 DB 조회 (ERP_WORKER_CERT)
    "scm_partner_qa",   # 사외 협력사(ERP_COMPANIES) 가용성 및 외주 실행 문의
    "knowledge_qa",     # 사내 규정(WPS), 시방서, KOSHA 안전 지침, 기상 단순 문의
    "doc_generation",   # 공식 작업지시서(WORK_INSTRUCTIONS) / 중량물 작업계획서 발행 요청
]

DocCategory = Literal[
    "manufacturing_tech",  # 제조기술 (WPS, 용접조건)
    "safety_health",       # 안전보건 (KOSHA, 밀폐공간, 보호구)
    "class_rules",         # 선급법규 (KR, DNV, ABS)
    "heavy_equipment",     # 장비중량물 (크레인, 트랜스포터)
    "quality_inspection",  # 품질검사 (NDT, 비파괴, 허용공차)
    "work_weather",        # 작업환경기상 (예열, 방풍막, 풍속제한)
    "internal_docs",       # 사내문서 (표준양식, SOP)
]

SearchStatus = Literal["idle", "completed", "empty", "error"]


class ShipAgentState(TypedDict, total=False):
    # 0. Supervisor 라우팅 제어
    next: Optional[str]

    # 1. 멀티턴 대화 히스토리 및 최신 질의
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str

    # 2. SHIP_USER.USERS 권한(RBAC) 및 작업 오더(WORK_ORDERS) 컨텍스트
    user_id: Optional[str]                        # USERS.USER_ID
    user_role: Optional[str]                      # WORKER | MANAGER | ADMIN
    user_type: Optional[str]                      # INTERNAL | EXTERNAL
    can_access_erp: Optional[str]                 # 'Y' | 'N'
    can_access_internal_rag: Optional[str]        # 'Y' | 'N'
    can_control_robot: Optional[str]              # 'Y' | 'N'
    default_workspace: Optional[str]              # 예: 'ZONE-A-BAY1'
    branch_code: Optional[str]                    # 예: 'BR-ULSAN-01', 'BR-GEOJE-01'
    work_id: Optional[str]                        # WORK_ORDERS.WORK_ID
    work_no: Optional[str]                        # WORK_ORDERS.WORK_NO (MES_WORK_ORDER_DISPATCH.WO_NUMBER와 1:1 매핑)

    # 3. Team 1 (4-Node 분업: 보안게이트 -> 의도분석 -> ERP진단 -> 완결성판정) 상태
    task_type: Optional[TaskType]
    is_action_requested: bool                     # 단순 조회(False) vs 원장 커밋/발주 실행(True)
    preferred_execution_mode: Optional[Literal["AUTO", "IN_HOUSE", "OUTSOURCED_B2B"]]
    outsourcing_domain: Optional[Literal["HEAVY_TRANSPORT", "WELDING_MATERIAL", "OUTSOURCED_WORKER"]]
    target_categories: List[DocCategory]
    need_weather: bool
    extracted_params: Dict[str, Any]              # 누적 보존 제원
    is_param_complete: Optional[str]              # "yes" | "no"
    missing_params: List[str]                     # 누락된 현장 필수 입력 제원
    search_queries: List[str]                     # 공학 표준 검색어 리스트

    # 4. SHIP_ERP & B2B SCM 연동 상태
    erp_resource_status: Optional[Dict[str, Any]]  # SHIP_ERP 통합 진단 결과
    is_outsourcing_required: bool                  # B2B 협력사 호출 플래그
    partner_erp_plan: Optional[Dict[str, Any]]     # PostgreSQL B2B 협력사 조회 결과
    erp_commit_result: Optional[Dict[str, Any]]    # 최종 재고 선점 및 MES_WORK_ORDER_DISPATCH 커밋 결과
    hitl_approved: Optional[bool]                  # 관리자 최종 승인 여부

    # 5. Team 2 (Supervisor 기반 지식·환경·비전·공급망 자율 감리) 상태
    team2_plan: List[str]                          # Team 2 Supervisor가 수립한 동적 실행 대기열
    search_status: SearchStatus
    retrieved_docs: List[Dict[str, Any]]
    doc_grade: Optional[str]                       # "yes" | "no"
    retry_count: int                               # 쿼리 재작성 루프 카운터
    weather_data: Optional[Dict[str, Any]]
    vision_data: Optional[Dict[str, Any]]          # VISION_ANALYSES / DETECTIONS / INSPECTIONS 통합 데이터

    # 6. Team 3 (거버넌스 라우터 & 문서 생성 & 트랜잭션 마감) 산출물
    generated_document: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    audit_saved: bool