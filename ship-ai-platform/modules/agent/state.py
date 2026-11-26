"""
modules/agent/state.py
Agentic AI + SHIP_USER(웹/비전) & SHIP_ERP(전사 자원) 통합 상태 구조체 (ShipAgentState)
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

TaskType = Literal[
    "chat",                  # 일상 대화 및 인사
    "cad_planning",          # CAD 도면 기반 작업계획 수립·분석, 공정 순서·일정 편성 및 일일 브리핑
    "vision_inspection_qa",  # 현장 비전 탐지 현황, 용접 대기 구역, 결함 및 로봇 관제 상태 조회
    "erp_resource_qa",       # 사내 자재(ERP_BOM_INVENTORY), 강판 Lot, 와이어 잔량, 대기 차량 확인
    "hr_safety_qa",          # 특정 작업자 선급(KR/AWS) 자격증 단건 유효기간 확인
    "hr_worker_qa",          # 사내/하청/프리랜서 작업자 전체 명단 실시간 DB 조회 (ERP_WORKER_CERT)
    "scm_partner_qa",        # 사외 협력사(ERP_COMPANIES) 가용성 및 외주 실행 문의
    "knowledge_qa",          # 사내 규정(WPS), 시방서, KOSHA 안전 지침, 기상 단순 문의
    "doc_generation",        # 공식 작업지시서(WORK_INSTRUCTIONS) / 중량물 작업계획서 발행 요청
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
    user_id: Optional[str]
    user_name: Optional[str]
    user_role: Optional[str]                      # WORKER | MANAGER | ADMIN
    user_type: Optional[str]                      # INTERNAL | EXTERNAL
    can_access_erp: Optional[str]                 # 'Y' | 'N'
    can_access_internal_rag: Optional[str]        # 'Y' | 'N'
    can_control_robot: Optional[str]              # 'Y' | 'N'
    default_workspace: Optional[str]
    branch_code: Optional[str]
    work_id: Optional[str]
    work_no: Optional[str]

    # 3. Team 1 (RBAC 초기화 -> 의도/목표 추출 -> ERP 사전진단 -> 완결성/자율편성 판정) 상태
    task_type: Optional[TaskType]
    is_action_requested: bool
    preferred_execution_mode: Optional[Literal["AUTO", "IN_HOUSE", "OUTSOURCED_B2B"]]
    outsourcing_domain: Optional[
        Literal[
            "ALL_PARTNERS",
            "HEAVY_TRANSPORT",
            "WELDING_MATERIAL",
            "OUTSOURCED_WORKER",
            "NOT_APPLICABLE",
        ]
    ]
    worker_scope: Optional[Literal["ALL", "HQ_ONLY", "EXTERNAL_ONLY"]]

    # Agentic 자율 편성 및 Self-Correction 제어 상태
    auto_assign_mode: bool                        # 블록/작업자 미지정 시 에이전트 자율 탐색·배정 허용
    replan_count: int                             # Team 3 Gatekeeper 차단 후 자율 재편성 횟수
    replan_needed: bool                           # Team 3 -> Team 2 복구 백엣지 트리거 플래그
    replan_feedback: Optional[str]                # 차단 사유 및 재편성 지시 피드백
    excluded_blocks: List[str]                    # 자율 탐색 중 불가 판정된 블록 제외 목록
    excluded_workers: List[str]                   # 결격 판정된 작업자 제외 목록
    agent_reasoning_summary: Optional[str]        # 에이전트 자율 선택·복구 판단 근거 요약
    tool_trace: List[Dict[str, Any]]              # ReAct 도구 호출·관찰 이력
    drafted_blocks_history: List[str]             # 현재 세션에서 이미 초안 발행된 블록 목록
    drafted_workers_history: List[str]            # 현재 세션에서 이미 초안 배정된 작업자 목록

    target_categories: List[DocCategory]
    need_weather: bool
    extracted_params: Dict[str, Any]
    is_param_complete: Optional[str]              # "yes" | "no"
    missing_params: List[str]
    search_queries: List[str]

    # 4. SHIP_ERP & B2B SCM 연동 상태
    erp_resource_status: Optional[Dict[str, Any]]
    is_outsourcing_required: bool
    partner_erp_plan: Optional[Dict[str, Any]]
    erp_commit_result: Optional[Dict[str, Any]]
    hitl_approved: Optional[bool]

    # 5. Team 2 (ReAct 자율 탐색 에이전트) 수집 결과
    team2_plan: Optional[List[str]]
    search_status: SearchStatus
    retrieved_docs: List[Dict[str, Any]]
    doc_grade: Optional[str]                      # "yes" | "no"
    retry_count: int
    weather_data: Optional[Dict[str, Any]]
    vision_data: Optional[Dict[str, Any]]

    # 6. Team 3 (거버넌스 게이트키퍼 & 문서 생성 & 트랜잭션 마감) 산출물
    generated_document: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    audit_saved: bool

    # 7. Team 4 (CAD 공정 계획 수립 & APS 스케줄링) 산출물
    cad_dxf_path: Optional[str]
    cad_pdf_path: Optional[str]
    cad_staging_result: Optional[Dict[str, Any]]

    prev_turn_waiting_clarification: bool
    hitl_target_doc_id: Optional[str]
    hitl_target_doc_version: Optional[str]