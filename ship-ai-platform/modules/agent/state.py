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
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    task_type: Optional[TaskType]
    target_categories: List[DocCategory]
    need_weather: bool
    extracted_params: Dict[str, Any]
    is_param_complete: Optional[str]
    missing_params: List[str]
    search_status: SearchStatus
    search_queries: List[str]
    retrieved_docs: List[Dict[str, Any]]
    weather_data: Optional[Dict[str, Any]]
    vision_data: Optional[Dict[str, Any]]
    generated_document: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    audit_saved: bool