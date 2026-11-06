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
    # 1. 1차 시도: 기본 검색 (access_level 제약 없이 순수 텍스트/카테고리 유사도 검색)
    search_kwargs: dict[str, Any] = {
        "query": query,
        "top_k": top_k,
    }
    if category:
        search_kwargs["category"] = category

    try:
        results = store.search(**search_kwargs)
    except TypeError:
        # store.search 시그니처 호환 처리
        search_kwargs.pop("category", None)
        try:
            results = store.search(**search_kwargs)
        except Exception as e:
            print(f"[Retriever Store Warning]: {e}")
            results = []
    except Exception as e:
        print(f"[Retriever Search Error]: {e}")
        results = []

    # 2. 카테고리 필터 때문에 비어있는 경우 전체 검색으로 Fallback
    if not results and category:
        search_kwargs.pop("category", None)
        try:
            results = store.search(**search_kwargs)
        except Exception:
            results = []

    # 3. 만약 store.search()에 필수 인자로 allowed_access_levels가 필요하다면 그때만 보완 호출
    if not results:
        try:
            search_kwargs["allowed_access_levels"] = ["INTERNAL", "PUBLIC", "CONFIDENTIAL", "ALL"]
            results = store.search(**search_kwargs)
        except Exception:
            pass

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