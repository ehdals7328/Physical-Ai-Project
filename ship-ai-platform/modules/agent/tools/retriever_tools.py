# modules/agent/tools/retriever_tools.py
from typing import Any, List, Optional, Union
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


# ------------------------------------------------------------------------------
# ⭕ [신규 추가] 7대 카테고리를 동적으로 지원하는 범용 검색 함수
# ------------------------------------------------------------------------------
def search_knowledge_by_categories(
        query: str,
        categories: Optional[Union[str, List[str]]] = None,
        top_k: int = 4
) -> List[dict[str, Any]]:
    """단일 카테고리 또는 다중 카테고리 목록을 전달받아 단 1회의 검색으로 인출"""
    try:
        cat_mapping = {
            "manufacturing_tech": "제조기술",
            "safety_health": "안전보건",
            "class_rules": "선급법규",
            "heavy_equipment": "장비중량물",
            "quality_inspection": "품질검사",
            "work_weather": "작업환경기상",
            "internal_docs": "사내문서",
        }

        target_cats: List[str] = []
        if isinstance(categories, str):
            target_cats = [cat_mapping.get(categories, categories)]
        elif isinstance(categories, list):
            target_cats = [cat_mapping.get(c, c) for c in categories if c]

        # 단일 카테고리 검색
        if len(target_cats) == 1:
            results = store.search(
                query=query,
                category=target_cats[0],
                top_k=top_k,
                allowed_access_levels=["INTERNAL", "PUBLIC"],
            )
        else:
            # 카테고리가 없거나 여러 개인 경우 전체 검색 후 인메모리 필터링 또는 통합 검색
            results = store.search(
                query=query,
                category=None,
                top_k=top_k * (len(target_cats) if target_cats else 1),
                allowed_access_levels=["INTERNAL", "PUBLIC"],
            )
            if target_cats:
                results = [r for r in results if r.get("category") in target_cats][:top_k]

        return results or []
    except Exception as e:
        print(f"[KnowledgeStore 검색 오류]: {e}")
        return []


# 기존 함수 호환성 유지용 래퍼
def search_knowledge_by_category(query: str, category: Optional[str] = None, top_k: int = 4):
    return search_knowledge_by_categories(query, categories=category, top_k=top_k)


# ------------------------------------------------------------------------------
# 기존 호환용 @tool 함수들 (그대로 유지)
# ------------------------------------------------------------------------------
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