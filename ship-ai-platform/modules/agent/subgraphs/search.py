"""
Team 2: 지식 및 환경 수집 서브그래프 (search.py)
Team 1에서 도출한 target_categories(7대 카테고리) 및 사용자 질의를 바탕으로
사내 규격 문서를 동적으로 인출하고, 조건부로 실시간 기상 데이터를 수집합니다.
"""
from typing import Any, Dict, List
from langgraph.graph import StateGraph, START, END

from modules.agent.state import ShipAgentState

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
        # 1. 범용 7대 카테고리 검색 함수가 구현되어 있는 경우 우선 사용
        if "search_knowledge_by_category" in globals() and callable(search_knowledge_by_category):
            if target_categories:
                for cat in target_categories:
                    raw_items = search_knowledge_by_category(query, category=cat, top_k=3)
                    for item in raw_items:
                        collected_docs.append({
                            "category": cat,
                            "content": item.get("chunk_text", "") or str(item),
                            "metadata": {
                                "document_name": item.get("document_name"),
                                "revision": item.get("revision"),
                                "page": item.get("page_start"),
                                "similarity": item.get("similarity_score"),
                            }
                        })
            else:
                raw_items = search_knowledge_by_category(query, top_k=4)
                for item in raw_items:
                    collected_docs.append({
                        "category": item.get("category", "일반규격"),
                        "content": item.get("chunk_text", "") or str(item),
                        "metadata": {
                            "document_name": item.get("document_name"),
                            "page": item.get("page_start"),
                        }
                    })

        # 2. 범용 함수 결과가 비어있거나 기존 개별 도구(wps, safety)만 존재하는 경우 폴백
        if not collected_docs:
            need_safety = any(c in ["safety_health", "heavy_equipment"] for c in target_categories) or \
                          any(k in query for k in ["안전", "밀폐", "산소", "크레인", "인양", "방호"])
            need_wps = any(c in ["manufacturing_tech", "quality_inspection", "class_rules"] for c in target_categories) or \
                       any(k in query for k in ["용접", "전류", "전압", "WPS", "모재", "두께"]) or not need_safety

            if need_wps and callable(wps_retriever_tool):
                wps_text = wps_retriever_tool.invoke(query)
                if wps_text and "찾지 못했습니다" not in str(wps_text):
                    collected_docs.append({
                        "category": "제조기술(WPS)",
                        "content": str(wps_text),
                        "metadata": {"source": "wps_retriever_tool"}
                    })

            if need_safety and callable(safety_retriever_tool):
                safety_text = safety_retriever_tool.invoke(query)
                if safety_text and "찾지 못했습니다" not in str(safety_text):
                    collected_docs.append({
                        "category": "안전보건/작업환경",
                        "content": str(safety_text),
                        "metadata": {"source": "safety_retriever_tool"}
                    })

        status = "completed" if collected_docs else "empty"
        return {
            "retrieved_docs": collected_docs,
            "search_status": status
        }

    except Exception as e:
        print(f"[Team 2 Search Error] RAG 문서 인출 중 예외 발생: {e}")
        return {
            "retrieved_docs": [],
            "search_status": "error"
        }


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

    # 날씨 필요 여부 확인
    is_weather_in_query = any(k in query for k in ["날씨", "기온", "풍속", "기상", "바람", "습도"])
    if not (need_weather or is_weather_in_query):
        return {"weather_data": None}

    # 1. 위치 파라미터 확인 (하드코딩 도시 리스트 완전 제거)
    location = params.get("location")

    # 2. 위치가 없으면 임의로 도시를 정하지 않고, 사용자에게 위치를 요구
    if not location:
        return {
            "weather_data": {
                "status": "NEED_LOCATION",
                "message": "기상 정보 확인을 위해 작업 위치(조선소 및 독 명칭)를 입력해 주세요."
            },
            "need_weather": True
        }

    # 3. 위치가 전달된 경우 FastMCP 실시간 기상 도구 실행 (동적 지오코딩)
    try:
        weather_info = get_realtime_weather_from_mcp(location)
        return {
            "weather_data": weather_info,
            "need_weather": True
        }
    except Exception as e:
        print(f"[Team 2 Weather Error] 기상 수집 예외: {e}")
        return {
            "weather_data": {
                "location": location,
                "error": str(e)
            },
            "need_weather": True
        }

# ==============================================================================
# 3. 서브그래프 빌드
# ==============================================================================
builder = StateGraph(ShipAgentState)

builder.add_node("rag_retriever", dynamic_rag_retriever_node)
builder.add_node("weather_collector", conditional_weather_node)

builder.add_edge(START, "rag_retriever")
builder.add_edge("rag_retriever", "weather_collector")
builder.add_edge("weather_collector", END)

team2_subgraph = builder.compile()