"""modules/agent/tools/retriever_tools.py
AI-Agent와 Flask 백엔드가 공통으로 사용하는 다중 RAG 검색 도구 + 다중 RDB(ERP) 자원 조회/배정 도구.
"""
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from modules.knowledge.retriever import KnowledgeStore

store = KnowledgeStore()


def _format_search_results(items: List[dict[str, Any]]) -> str:
    if not items:
        return "관련된 규격 및 지침 문서를 찾지 못했습니다."
    formatted = []
    for item in items:
        doc_name = item.get("document_name") or "규격 문서"
        revision = item.get("revision") or "UNKNOWN"
        sim = item.get("similarity_score", 0.0)
        page = item.get("page_start", 1)
        rag_type = item.get("rag_type", "RAG")
        text = str(item.get("chunk_text", "")).strip()
        header = f"[{rag_type} | {doc_name} / Rev.{revision} (p.{page}, 유사도: {sim:.2f})]"
        formatted.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(formatted)


def search_knowledge(query: str, top_k: int = 4, rag_mode: str = "ALL") -> List[dict[str, Any]]:
    """다중 RAG 통합 검색 (공통규격 + 사내내규)"""
    try:
        return store.search(query=query, top_k=top_k, rag_mode=rag_mode)
    except Exception as e:
        print(f"[KnowledgeStore 검색 오류]: {e}")
        return []


# 기존 에이전트 서브그래프(search.py 등) 호환 래퍼
def search_knowledge_by_categories(query: str, categories: Any = None, top_k: int = 4) -> List[dict[str, Any]]:
    return search_knowledge(query=query, top_k=top_k, rag_mode="ALL")


def search_knowledge_by_category(query: str, category: Any = None, top_k: int = 4) -> List[dict[str, Any]]:
    return search_knowledge(query=query, top_k=top_k, rag_mode="ALL")


@tool
def search_wps_standard(query: str) -> str:
    """용접절차시방서(WPS) 및 사내 기술 규격을 검색합니다."""
    return _format_search_results(search_knowledge(query, top_k=4))


@tool
def search_safety_rules(query: str) -> str:
    """조선소 안전보건 지침 및 사내 보안 내규를 검색합니다."""
    return _format_search_results(search_knowledge(query, top_k=3))


wps_retriever_tool = search_wps_standard
safety_retriever_tool = search_safety_rules


# ==============================================================================
# [ERP 다중 DB 연동 도구] ship_erp 스키마에서 지부별 로봇/인력/자재를 실시간 조회
# ==============================================================================
def get_available_resources(branch_name: str = "울산 1야드") -> Dict[str, Any]:
    """특정 지부(야드/도크)의 가용 용접 로봇, 대기 기술자, 실시간 자재 재고를 ship_erp DB에서 조회"""
    from app.models.schema import ErpBranch, ErpCompany, ErpWorker, ErpWeldingRobot, ErpBranchInventory

    branch = ErpBranch.query.filter(ErpBranch.branch_name.like(f"%{branch_name[:2]}%")).first()
    if not branch:
        branch = ErpBranch.query.first()

    target_b_id = branch.branch_id if branch else 1
    target_b_name = branch.branch_name if branch else branch_name

    robots = ErpWeldingRobot.query.filter_by(branch_id=target_b_id).all()
    workers = ErpWorker.query.filter_by(branch_id=target_b_id).all()
    inventories = ErpBranchInventory.query.filter_by(branch_name=target_b_name).all()
    companies = {c.company_id: c.company_name for c in ErpCompany.query.all()}

    return {
        "branch_id": target_b_id,
        "branch_name": target_b_name,
        "available_robots": [
            {
                "robot_id": r.robot_id,
                "serial": r.robot_serial,
                "model": r.model_name,
                "payload_kg": r.payload_kg,
                "status": r.status,
            }
            for r in robots
        ],
        "available_workers": [
            {
                "worker_id": w.worker_id,
                "name": w.worker_name,
                "code": w.worker_code,
                "company": companies.get(w.company_id, "협력사"),
                "skill": w.skill_grade,
                "status": w.status,
            }
            for w in workers
        ],
        "inventory": [
            {
                "inventory_id": i.inventory_id,
                "code": i.material_code,
                "category": i.material_category,
                "name": i.material_name,
                "qty": i.current_qty,
                "unit": i.unit,
                "safety_qty": i.min_safety_qty,
                "is_low": i.current_qty < i.min_safety_qty,
            }
            for i in inventories
        ],
    }
