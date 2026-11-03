# agent_system/tools/retriever_tools.py
from typing import Any, List
from langchain_core.tools import tool
# 친구가 작성한 retriever.py(KnowledgeStore) 임포트 경로에 맞게 지정
from modules.knowledge.retriever import KnowledgeStore

# 1. 싱글톤 형태의 KnowledgeStore 인스턴스 생성
store = KnowledgeStore()


def _format_search_results(items: List[dict[str, Any]]) -> str:
    """
    ChromaDB 검색 결과(Dictionary List)를 LLM 프롬프트에 주입 가능한 표준 텍스트 포맷으로 직렬화합니다.
    """
    if not items:
        return "관련된 규격 및 지침 문서를 찾지 못했습니다."

    formatted = []
    for item in items:
        doc_name = item.get("document_name") or "규격 문서"
        revision = item.get("revision") or "UNKNOWN"
        sim = item.get("similarity_score", 0.0)
        page = item.get("page_start", 1)
        text = str(item.get("chunk_text", "")).strip()

        header = f"[{doc_name} / Rev.{revision} (p.{page}, 유사도: {sim:.2f})]"
        formatted.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(formatted)


@tool
def search_wps_standard(query: str) -> str:
    """
    용접절차시방서(WPS) 규격을 검색합니다.
    강종(SS400, AH36 등), 판 두께(T), 이음 형상에 따른 표준 전류, 전압, 주행속도, 루트 갭 기준을 반환합니다.
    """
    # 친구의 search() 메서드 호출: category='WPS' 또는 적절한 필터 지정
    results = store.search(
        query=query,
        category="WPS",  # 실제 적재된 카테고리명에 맞춤
        top_k=4,
        allowed_access_levels=["INTERNAL", "PUBLIC"],
    )
    # 카테고리 필터링 결과가 없으면 전체 검색으로 폴백
    if not results:
        results = store.search(query=query, top_k=3)

    return _format_search_results(results)


@tool
def search_safety_rules(query: str) -> str:
    """
    선박 제조 현장 안전 지침, KOSHA 규격 및 중량물 인양/크레인 안전 기준을 검색합니다.
    """
    results = store.search(
        query=query,
        category="SAFETY",  # 안전 카테고리 지정
        top_k=3,
        allowed_access_levels=["INTERNAL", "PUBLIC"],
    )
    if not results:
        results = store.search(query=query, top_k=3)

    return _format_search_results(results)


# 기존 subgraphs/search.py에서 import하던 변수명과 일치시킴
wps_retriever_tool = search_wps_standard
safety_retriever_tool = search_safety_rules