"""
[파일 역할] LangGraph 멀티에이전트가 사용하는 공유 상태(ShipAgentState) 정의입니다.
           기존 필드와 100% 호환되면서, 사용자 권한(`user_context`), ERP 가용 자원(`erp_data`),
           Vision 객체/용접선 분석 결과(`vision_data`)를 추가로 담을 수 있도록 확장되었습니다.
"""
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# 작업 의도: 단순 대화 / 규격 전문 Q&A / 공식 문서(작업지시서, 계획서 등) 생성
TaskType = Literal["chat", "knowledge_qa", "doc_generation"]

# 기존 에이전트 서브그래프 호환용 카테고리 타입
DocCategory = Literal[
    "manufacturing_tech",
    "safety_health",
    "class_rules",
    "heavy_equipment",
    "quality_inspection",
    "work_weather",
    "internal_docs",
]

SearchStatus = Literal["idle", "completed", "empty", "error"]


class ShipAgentState(TypedDict, total=False):
    """
    AI-Agent 팀원의 서브그래프(Team 1, 2, 3)와 Flask 백엔드가 데이터를 주고받는 표준 상태 그릇입니다.
    total=False로 설정하여 일부 키가 생략되어도 KeyError 없이 유연하게 동작합니다.
    """
    # 대화 히스토리 (멀티턴 컨텍스트 보존)
    messages: Annotated[List[BaseMessage], add_messages]

    # 최신 사용자 입력 질의
    user_query: str

    # [신규 확장 1] 로그인한 사용자의 회사(원청/협력사), 담당 지부(울산 1야드 등), 직책 정보
    user_context: Optional[Dict[str, Any]]

    # [신규 확장 2] Vision(YOLOv11)이 검출한 용접선 궤적, 루트 간격(mm), 간섭 위험물 정보
    vision_data: Optional[Dict[str, Any]]

    # [신규 확장 3] ERP DB(`ship_erp`)에서 조회한 해당 지부의 가용 로봇, 대기 기술자, 자재 재고 현황
    erp_data: Optional[Dict[str, Any]]

    # Team 1: 분석 및 라우팅 상태
    task_type: Optional[TaskType]
    target_categories: List[DocCategory]
    need_weather: bool
    extracted_params: Dict[str, Any]
    is_param_complete: Optional[str]
    missing_params: List[str]

    # Team 2: 지식 및 환경 수집 상태
    search_status: SearchStatus
    search_queries: List[str]
    retrieved_docs: List[Dict[str, Any]]
    weather_data: Optional[Dict[str, Any]]

    # Team 3: 생성 결과
    generated_document: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    audit_saved: bool