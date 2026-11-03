# agent_system/state.py
from typing import Annotated, Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ShipAgentState(TypedDict, total=False):
    # =========================================================================
    # 0. 멀티턴 대화 세션 히스토리 (🌟 핵심 추가!)
    # - add_messages 리듀서를 적용하여 턴이 바뀔 때 대화가 덮어써지지 않고 누적됨
    # =========================================================================
    messages: Annotated[List[BaseMessage], add_messages]

    # =========================================================================
    # 1. Supervisor 라우팅 및 단일 턴 입력
    # =========================================================================
    next: Optional[str]  # 최상위 Supervisor의 다음 이동 노드
    user_query: str  # 현재 턴에서 사용자가 입력한 자연어 질의

    # =========================================================================
    # 2. Team 1 (요구사항 분석팀) 산출물
    # =========================================================================
    task_type: Optional[
        Literal["general_qa", "weld_order", "heavy_plan"]
    ]  # 의도 분류
    extracted_params: Dict[str, Any]  # 누적/추출된 엔티티 (강종, 두께, 위치 등)
    search_queries: List[str]  # Vector DB 검색 쿼리 리스트

    # =========================================================================
    # 3. Team 2 (지식·환경 검색팀) 산출물
    # =========================================================================
    retrieved_docs: List[str]  # WPS/KOSHA 사내 규격 인출 본문
    weather_info: Optional[Dict[str, Any]]  # 기상 데이터 (온도, 풍속, 습도 등)
    is_docs_relevant: Optional[Literal["yes", "no"]]  # 문서 적합성 평가 결과
    retry_count: int  # 재검색 횟수 카운터

    # =========================================================================
    # 4. Team 3 (맞춤형 문서 생성팀) 산출물
    # =========================================================================
    final_answer: Optional[str]  # 최종 생성된 답변 요약 텍스트
    generated_document: Optional[Dict[str, Any]]  # 표준 작업지시서 / 계획서 JSON

    is_param_complete: Optional[Literal["yes", "no"]]  # 필수 정보 완비 여부

    safety_evaluation: Optional[Dict[str, Any]]  # 안전 룰 검증 결과 (PASS/STOP, 주의사항)