# modules/agent/tools/retriever_tools.py
from typing import Any, List, Optional
from langchain_core.tools import tool
from modules.knowledge.retriever import KnowledgeStore

# 싱글톤 인스턴스
store = KnowledgeStore()


def _format_search_results(items: List[dict[str, Any]]) -> str:
    """ChromaDB 검색 결과(Dictionary List)를 LLM 프롬프트에 주입 가능한 표준 텍스트 포맷으로 직렬화합니다."""
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


def search_knowledge(
    query: str,
    category: Optional[str] = None,
    top_k: int = 4
) -> List[dict[str, Any]]:
    """ChromaDB 지식베이스에서 카테고리 필터(선택)를 반영하여 유사도 상위 문서를 인출합니다."""
    search_kwargs: dict[str, Any] = {
        "query": query,
        "top_k": top_k,
        "allowed_access_levels": ["INTERNAL", "PUBLIC"],
    }
    if category:
        search_kwargs["category"] = category

    try:
        results = store.search(**search_kwargs)
    except TypeError:
        # store.search가 category 인자를 받지 않는 구버전일 경우 호환 처리
        search_kwargs.pop("category", None)
        results = store.search(**search_kwargs)

    # 카테고리 지정 검색 결과가 비어있다면 전체 카테고리에서 재검색(Fallback)
    if not results and category:
        search_kwargs.pop("category", None)
        results = store.search(**search_kwargs)

    return results or []


def search_knowledge_by_category(
    query: str,
    category: Optional[str] = None,
    top_k: int = 4
) -> List[dict[str, Any]]:
    return search_knowledge(query=query, category=category, top_k=top_k)


def search_knowledge_by_categories(
    query: str,
    categories: Optional[List[str]] = None,
    top_k: int = 4
) -> List[dict[str, Any]]:
    """복수 카테고리가 지정된 경우 각 카테고리 결과를 병합 및 중복 제거하여 상위 top_k개를 반환합니다."""
    if not categories:
        return search_knowledge(query=query, top_k=top_k)

    merged: List[dict[str, Any]] = []
    seen_ids = set()
    for cat in categories:
        items = search_knowledge(query=query, category=cat, top_k=top_k)
        for item in items:
            cid = item.get("chunk_id") or item.get("id") or str(item.get("chunk_text", ""))[:80]
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(item)

    merged.sort(key=lambda x: float(x.get("similarity_score") or 0.0), reverse=True)
    return merged[:top_k] if merged else search_knowledge(query=query, top_k=top_k)


@tool
def search_wps_standard(query: str) -> str:
    """용접절차시방서(WPS) 규격을 검색합니다."""
    results = search_knowledge_by_category(query, category="제조기술", top_k=4)
    return _format_search_results(results)


@tool
def search_safety_rules(query: str) -> str:
    """선박 제조 현장 안전 지침, KOSHA 규격 및 중량물 인양/크레인 안전 기준을 검색합니다."""
    results = search_knowledge_by_category(query, category="안전보건", top_k=3)
    return _format_search_results(results)


wps_retriever_tool = search_wps_standard
safety_retriever_tool = search_safety_rules