"""
[파일 역할] AI-Agent 담당 팀원이 복잡한 SQL 쿼리나 ChromaDB 구조를 몰라도
           함수 호출 한 번으로 다중 RAG 지식과 ERP DB(로봇·인력·자재·협력사 현황)를 조회하고
           자원 배정까지 수행할 수 있도록 제공하는 통합 도구(Tool) 모듈입니다.
"""
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from modules.knowledge.retriever import KnowledgeStore

# 싱글톤 벡터 스토어 인스턴스
store = KnowledgeStore()


def _format_search_results(items: List[dict[str, Any]]) -> str:
    """ChromaDB 검색 결과를 프롬프트에 주입하기 좋은 텍스트 포맷으로 직렬화"""
    if not items:
        return "관련된 규격 및 지침 문서를 찾지 못했습니다."

    formatted = []
    for item in items:
        doc_name = item.get("document_name") or "규격 문서"
        revision = item.get("revision") or "UNKNOWN"
        sim = item.get("similarity_score", 0.0)
        page = item.get("page_start", 1)
        text = str(item.get("chunk_text", "")).strip()

        header = f"[{doc_name} / Rev.{revision} (p.{page}, 신뢰도: {sim:.2f})]"
        formatted.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(formatted)


def search_knowledge(
    query: str,
    top_k: int = 4,
    company_type: Optional[str] = "MAIN"
) -> List[dict[str, Any]]:
    """
    ChromaDB 단일 고속 검색.
    - search_multi_rag 메서드가 retriever.py에 있으면 원청/협력사 멀티 RAG 실행
    - 없으면 기본 search 메서드로 안전하게 폴백
    """
    try:
        if hasattr(store, "search_multi_rag"):
            return store.search_multi_rag(query=query, top_k=top_k, company_type=company_type)
        return store.search(query=query, top_k=top_k)
    except Exception as e:
        print(f"[KnowledgeStore 검색 오류]: {e}")
        return []


# ==============================================================================
# 1. 다중 RAG 지식 검색 함수 (카테고리 필터 없이 전체 고속 검색 + 권한 연동)
# ==============================================================================
def search_knowledge_by_categories(
    query: str,
    categories: Any = None,
    top_k: int = 4
) -> List[dict[str, Any]]:
    """search.py 및 __init__.py의 ImportError를 해결하는 호환 함수"""
    return search_knowledge(query=query, top_k=top_k)


def search_knowledge_by_category(
    query: str,
    category: Any = None,
    top_k: int = 4
) -> List[dict[str, Any]]:
    """단일 카테고리 인자 호출 호환 함수"""
    return search_knowledge(query=query, top_k=top_k)


@tool
def search_wps_standard(query: str) -> str:
    """용접절차시방서(WPS), 모재 두께별 예열 조건, 선급 법규 및 품질 검사 기준을 통합 검색합니다."""
    results = search_knowledge_by_category(query, top_k=4)
    return _format_search_results(results)


@tool
def search_safety_rules(query: str) -> str:
    """선박 제조 현장 안전 지침, KOSHA 규격, 중량물 인양 및 사내 보안/작업 내규를 통합 검색합니다."""
    results = search_knowledge_by_category(query, top_k=3)
    return _format_search_results(results)


# ==============================================================================
# 2. ERP 다중 DB (`ship_erp`) 조회 및 자원 배정 도구 (AI-Agent & Flask 공용)
# ==============================================================================
def query_branch_erp_resources(branch_name: str = "울산 1야드", company_type: str = "MAIN") -> Dict[str, Any]:
    """
    특정 지부(예: '울산 1야드', '거제 1도크')에서 현재 즉시 투입 가능한
    자동화 용접 로봇, 대기 중인 용접 기술자(협력업체 포함), 실시간 자재 재고를 `ship_erp` DB에서 일괄 조회하는 함수입니다.
    """
    try:
        from app.models.schema import (
            ERPCompany, ERPBranch, ERPWorker, ERPWeldingRobot, ERPBranchInventory
        )

        # 1) 지부 기본 정보
        branch_obj = ERPBranch.query.filter_by(branch_name=branch_name).first()
        city = branch_obj.location_city if branch_obj else ("거제" if "거제" in branch_name else "울산")

        # 2) 해당 지부의 용접 로봇 목록
        robots = ERPWeldingRobot.query.filter_by(branch_name=branch_name).all()
        robot_list = [
            {
                "robot_id": r.robot_id,
                "robot_serial": r.robot_serial,
                "robot_name": r.robot_name,
                "model_spec": r.model_spec,
                "status": r.robot_status,
                "current_block": r.current_block,
                "health": r.battery_or_health,
            }
            for r in robots
        ]

        # 3) 해당 지부의 기술자 목록 (원청 + 협력사 조인)
        workers_q = (
            ERPWorker.query
            .join(ERPCompany, ERPWorker.company_id == ERPCompany.company_id)
            .add_columns(ERPCompany.company_name, ERPCompany.company_type)
            .filter(ERPWorker.branch_name == branch_name)
            .all()
        )
        worker_list = [
            {
                "worker_id": w[0].worker_id,
                "emp_no": w[0].emp_no,
                "name": w[0].worker_name,
                "company_name": w[1],
                "company_type": w[2],
                "role_title": w[0].role_title,
                "license": w[0].welding_license,
                "status": w[0].work_status,
                "current_block": w[0].current_block,
            }
            for w in workers_q
        ]

        # 4) 해당 지부의 자재/용접봉/보호가스 재고 목록
        inventories = ERPBranchInventory.query.filter_by(branch_name=branch_name).all()
        inventory_list = [
            {
                "inventory_id": inv.inventory_id,
                "material_code": inv.material_code,
                "category": inv.material_category,
                "material_name": inv.material_name,
                "spec": inv.specification,
                "current_qty": inv.current_qty,
                "unit": inv.unit,
                "min_safety_qty": inv.min_safety_qty,
                "is_low_stock": inv.current_qty <= inv.min_safety_qty,
            }
            for inv in inventories
        ]

        return {
            "branch_name": branch_name,
            "location_city": city,
            "available_robots": [r for r in robot_list if r["status"] == "IDLE"],
            "all_robots": robot_list,
            "available_workers": [w for w in worker_list if w["status"] == "AVAILABLE"],
            "all_workers": worker_list,
            "inventory": inventory_list,
        }
    except Exception as exc:
        print(f"[ERP DB 조회 예외]: {exc}")
        return {
            "branch_name": branch_name,
            "available_robots": [],
            "available_workers": [],
            "inventory": [],
            "error": str(exc),
        }


@tool
def get_branch_erp_resources_tool(branch_name: str = "울산 1야드") -> Dict[str, Any]:
    """
    AI-Agent가 작업지시서를 작성할 때 특정 조선소 지부(예: '울산 1야드', '거제 1도크')의
    가용 용접 로봇, 대기 중인 용접 기술자, 용접봉/강재/가스 실시간 재고 수량을 확인하는 도구입니다.
    """
    return query_branch_erp_resources(branch_name=branch_name)


def reserve_erp_resources_transaction(
    *,
    robot_serial: Optional[str] = None,
    worker_emp_no: Optional[str] = None,
    material_code: Optional[str] = None,
    use_qty: float = 1.0,
    target_block: str = "B-101",
) -> Dict[str, Any]:
    """
    작업지시서가 최종 승인(HITL)되었을 때, `ship_erp` DB에서 해당 로봇을 'RUNNING'으로,
    작업자를 'WORKING'으로 변경하고 사용된 자재 재고를 차감하는 단일 원자적(Atomic) 트랜잭션 함수입니다.
    """
    from app.models.schema import db, ERPWorker, ERPWeldingRobot, ERPBranchInventory

    actions = []
    try:
        if robot_serial:
            robot = ERPWeldingRobot.query.filter_by(robot_serial=robot_serial).first()
            if robot:
                if robot.robot_status != "IDLE":
                    return {"status": "error", "message": f"로봇 [{robot_serial}]은 현재 대기 상태가 아닙니다 ({robot.robot_status})."}
                robot.robot_status = "RUNNING"
                robot.current_block = target_block
                actions.append(f"로봇 {robot.robot_name}({robot_serial}) ➔ {target_block} 배정 완료")

        if worker_emp_no:
            worker = ERPWorker.query.filter_by(emp_no=worker_emp_no).first()
            if worker:
                if worker.work_status != "AVAILABLE":
                    return {"status": "error", "message": f"기술자 [{worker.worker_name}]님은 현재 대기 상태가 아닙니다."}
                worker.work_status = "WORKING"
                worker.current_block = target_block
                actions.append(f"기술자 {worker.worker_name}({worker_emp_no}) ➔ {target_block} 투입 완료")

        if material_code and use_qty > 0:
            inv = ERPBranchInventory.query.filter_by(material_code=material_code).first()
            if inv:
                if inv.current_qty < use_qty:
                    return {"status": "error", "message": f"자재 [{inv.material_name}] 재고가 부족합니다 (현재: {inv.current_qty})."}
                inv.current_qty = round(inv.current_qty - float(use_qty), 2)
                actions.append(f"자재 {inv.material_name} {use_qty}{inv.unit} 불출 완료 (잔여: {inv.current_qty}{inv.unit})")

        db.session.commit()
        return {"status": "success", "actions": actions}
    except Exception as exc:
        db.session.rollback()
        return {"status": "error", "message": f"ERP 자원 배정 트랜잭션 실패: {exc}"}


# 기존 코드 호환용 별칭 유지
wps_retriever_tool = search_wps_standard
safety_retriever_tool = search_safety_rules