"""
Team 2: 지식 및 환경 수집 서브그래프 (search.py)
Team 1에서 도출한 target_categories(7대 카테고리) 및 사용자 질의를 바탕으로
사내 규격 문서를 동적으로 인출하고, 조건부로 실시간 기상 데이터를 수집합니다.
"""
from typing import Any, Dict, List
from langgraph.graph import StateGraph, START, END
from modules.agent.state import ShipAgentState

from modules.agent.tools.retriever_tools import search_knowledge_by_categories
from modules.agent.tools.weather_mcp_client import get_realtime_weather_from_mcp

# 도구 및 클라이언트 안전 임포트
try:
    from modules.agent.tools.retriever_tools import (
        search_knowledge_by_category,
        wps_retriever_tool,
        safety_retriever_tool,
    )
except ImportError:
    # 패키지 내부 상대 경로 폴백
    from ..tools.retriever_tools import (
        search_knowledge_by_category,
        wps_retriever_tool,
        safety_retriever_tool,
    )

try:
    from modules.agent.tools.weather_mcp_client import get_realtime_weather_from_mcp
except ImportError:
    try:
        from ..tools.weather_mcp_client import get_realtime_weather_from_mcp
    except ImportError:
        def get_realtime_weather_from_mcp(location: str):
            return {"location": location, "temperature": 18.0, "wind_speed": 2.5}


# ==============================================================================
# 1. RAG 규격 문서 동적 인출 노드
# ==============================================================================
def dynamic_rag_retriever_node(state: ShipAgentState) -> Dict[str, Any]:
    """
    질의 의도 및 7대 카테고리(제조기술, 안전보건, 선급법규, 장비중량물, 품질검사, 작업환경기상, 사내문서)에
    맞춰 ChromaDB 문서를 동적으로 인출하는 노드
    """
    query = state.get("user_query", "")
    target_categories = state.get("target_categories", [])
    collected_docs: List[Dict[str, Any]] = []

    try:
        # 단 1회의 통합 호출로 개선
        raw_items = search_knowledge_by_categories(query, categories=target_categories, top_k=4)
        for item in raw_items:
            collected_docs.append({
                "category": item.get("category", "규격"),
                "content": item.get("chunk_text", "") or str(item),
                "metadata": {
                    "document_name": item.get("document_name"),
                    "revision": item.get("revision"),
                    "page": item.get("page_start"),
                    "similarity": item.get("similarity_score"),
                    "source": item.get("source"),
                    "doc_id": item.get("chunk_id")
                }
            })
        status = "completed" if collected_docs else "empty"
        return {"retrieved_docs": collected_docs, "search_status": status}
    except Exception as e:
        print(f"[Team 2 Search Error]: {e}")
        return {"retrieved_docs": [], "search_status": "error"}


# ==============================================================================
# 2. 조건부 실시간 기상 수집 노드
# ==============================================================================
def conditional_weather_node(state: ShipAgentState) -> Dict[str, Any]:
    """
    need_weather가 True일 때 실시간 기상을 수집하며,
    위치 정보가 전혀 없을 경우 하드코딩 기본값을 쓰지 않고 사용자에게 역질문 유도
    """
    need_weather = state.get("need_weather", False)
    query = state.get("user_query", "")
    params = state.get("extracted_params", {}) or {}

    is_weather_in_query = any(k in query for k in ["날씨", "기온", "풍속", "기상", "바람", "습도"])
    if not (need_weather or is_weather_in_query):
        return {"weather_data": None}

    location = params.get("location")
    if not location:
        return {
            "weather_data": {
                "status": "NEED_LOCATION",
                "message": "기상 정보 확인을 위해 작업 위치를 입력해 주세요."
            },
            "need_weather": True
        }

    try:
        weather_info = get_realtime_weather_from_mcp(location)
        return {"weather_data": weather_info, "need_weather": True}
    except Exception as e:
        return {
            "weather_data": {"location": location, "error": str(e)},
            "need_weather": True
        }

# ==============================================================================
# 3. 서브그래프 빌드
# ==============================================================================
builder = StateGraph(ShipAgentState)

builder.add_node("rag_retriever", dynamic_rag_retriever_node)
builder.add_node("weather_collector", conditional_weather_node)

# START -> 2개 노드로 동시 Fan-Out
builder.add_edge(START, "rag_retriever")
builder.add_edge(START, "weather_collector")

# 2개 노드가 모두 끝나면 END로 병합 (Fan-In)
builder.add_edge("rag_retriever", END)
builder.add_edge("weather_collector", END)

team2_subgraph = builder.compile()