from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# 작업 의도: 단순 대화 / 7대 규격 전문 Q&A / 공식 문서(작업지시서, 계획서 등) 생성
TaskType = Literal["chat", "knowledge_qa", "doc_generation"]

# 7대 RAG 문서 카테고리 규격
DocCategory = Literal[
    "manufacturing_tech",  # 1. 제조기술
    "safety_health",  # 2. 안전보건
    "class_rules",  # 3. 선급법규
    "heavy_equipment",  # 4. 장비중량물
    "quality_inspection",  # 5. 품질검사
    "work_weather",  # 6. 작업환경기상
    "internal_docs",  # 7. 사내문서
]

SearchStatus = Literal["idle", "completed", "empty", "error"]


class ShipAgentState(TypedDict):
  # 대화 히스토리 (멀티턴 컨텍스트 보존)
  messages: Annotated[List[BaseMessage], add_messages]

  # 최신 사용자 입력 질의
  user_query: str

  # Team 1: 분석 및 라우팅 상태
  task_type: Optional[TaskType]
  target_categories: List[DocCategory]  # 질의에 맞는 RAG 카테고리 목록
  need_weather: bool  # 야외/기상 연동 필요 여부
  extracted_params: Dict[str, Any]  # 사용자가 직접 언급한 제원(하드코딩 기본값 없음)
  is_param_complete: Optional[str]  # "yes" | "no" (doc_generation 시 필수)
  missing_params: List[str]  # 누락된 필수 파라미터 목록

  # Team 2: 지식 및 환경 수집 상태
  search_status: SearchStatus  # idle -> completed / empty / error
  search_queries: List[str]  # 생성된 RAG 검색 쿼리들
  retrieved_docs: List[Dict[str, Any]]  # 인출된 청크 (출처, 카테고리, 본문)
  weather_data: Optional[Dict[str, Any]]  # 기상청 MCP 데이터

  # Team 3: 생성 결과
  generated_document: Optional[Dict[str, Any]]  # 정형화된 문서 결과물
  final_answer: Optional[str]  # 최종 사용자 응답 문장
  audit_saved: bool