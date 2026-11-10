"""
modules/agent/subgraphs/analysis.py
Team 1: USER 권한(RBAC) 검증, 요구사항 분석, SHIP_ERP 1차 진단 & 구조화 의도 추출 서브그래프
"""

import json
from typing import Any, Dict, List, Literal, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from modules.agent.state import DocCategory, ShipAgentState
from modules.agent.tools.erp_tools import check_hq_erp_resources

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class UniversalQueryAnalysis(BaseModel):
    task_type: Literal[
        "chat",
        "erp_resource_qa",
        "hr_safety_qa",
        "hr_worker_qa",
        "scm_partner_qa",
        "knowledge_qa",
        "doc_generation",
    ] = Field(description="질의의 핵심 목적을 분류")

    is_action_requested: bool = Field(
        default=False,
        description="실제 외주 배차 확정, 발주 진행, 원장 커밋 등 실행을 요구하면 True, 단순 조회는 False",
    )
    context_action: Literal["KEEP_MERGE", "RESET_NEW"] = Field(
        default="KEEP_MERGE",
        description="새로운 주제의 문서 작성이면 'RESET_NEW', 이전 대화 조건을 이어받으면 'KEEP_MERGE'",
    )
    target_categories: List[DocCategory] = Field(
        default_factory=list,
        description="필요한 RAG 7대 카테고리 목록",
    )
    need_weather: bool = Field(
        default=False,
        description="실시간 기상 조회 필요 여부. 날씨 질의나 'doc_generation'인 경우 True",
    )
    block_id: Optional[str] = Field(None, description="언급된 선박 블록 번호 (예: 'B-104', 'BLK-E102')")
    material: Optional[str] = Field(None, description="언급된 모재 강종 (예: 'SS400', 'AH36')")
    thickness: Optional[str] = Field(None, description="언급된 판 두께 (예: '10T', '20T')")
    location: Optional[str] = Field(None, description="언급된 작업 도시 또는 지역명 (예: '울산', '거제'. 베이나 구역 코드는 제외)")
    process: Optional[str] = Field(None, description="용접/작업 공정 (예: 'FCAW', 'SAW')")
    weld_position: Optional[str] = Field(None, description="용접 자세 코드 (예: '1G', '2F')")
    worker_name: Optional[str] = Field(
        None, 
        description="투입 인력의 순수 성명 (문맥에서 직책/수식어/조사를 제외한 이름 본체만 추출. 예: '김철수 반장' -> '김철수', '조객원 작업자' -> '조객원')"
    )
    equipment: Optional[str] = Field(None, description="투입 장비명")
    additional_specs: Dict[str, Any] = Field(
        default_factory=dict,
        description="사용자가 직접 언급한 기타 엔지니어링 조건",
    )
    search_queries: List[str] = Field(
        default_factory=list,
        description="ChromaDB 벡터 검색을 위한 공학 표준 검색어",
    )


structured_analyzer = llm.with_structured_output(
    UniversalQueryAnalysis,
    method="function_calling",
)

ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 스마트 조선소 엔지니어링 & ERP 총괄 분석 에이전트(Team 1)입니다.\n"
        "이전 대화 기록과 사용자의 최신 질의를 분석하여 의도(task_type), 액션 요청 여부(is_action_requested), 검색 쿼리, 제원을 추출하세요.\n\n"
        "[핵심 분류 원칙]\n"
        "1. 기술 기준, 선박 건조 과정, 작업지시서 구성 항목 설명 요청은 지식 질의이므로 'knowledge_qa'로 분류하세요.\n"
        "2. 사내 강판/와이어 재고 수량이나 대기 차량/장비 가용성 문의는 'erp_resource_qa'로 분류하세요.\n"
        "3. 사내/외주 작업자 전체 명단이나 인력 목록 조회 요청은 'hr_worker_qa'로 분류하세요.\n"
        "4. 특정 작업자 1인의 선급 자격증 유효기간이나 적격 여부 문의는 'hr_safety_qa'로 분류하세요.\n"
        "5. 사외 협력사 현황 조회 및 외주 발주/배차 요청은 'scm_partner_qa'로 분류하세요.\n"
        "6. 공식 '작업지시서' 또는 '작업계획서'의 작성/발행을 요청한 경우에만 'doc_generation'으로 분류하세요.\n"
        "7. 실제 외주 실행, 배차 확정, 발주 승인 등 원장 커밋을 요구하면 is_action_requested=True로 설정하십시오.",
    ),
    MessagesPlaceholder(variable_name="messages", optional=True),
    (
        "human",
        "[직전 턴 역질문 대기 여부]: {is_waiting}\n"
        "[로그인 사용자 기본 작업구역]: {user_workspace} (야드코드: {branch_code})\n"
        "[이전 누적 작업 제원]: {accumulated_params}\n"
        "[최신 사용자 질의]: {query}",
    ),
])


def analyze_and_check_erp_node(state: ShipAgentState) -> Dict[str, Any]:
    query = state.get("user_query", "").strip()
    messages = state.get("messages", [])
    existing_params = dict(state.get("extracted_params") or {})
    is_waiting = state.get("is_param_complete") == "no"

    can_access_erp = state.get("can_access_erp", "Y")
    default_workspace = state.get("default_workspace")
    branch_code = state.get("branch_code")

    res: UniversalQueryAnalysis = (ANALYSIS_PROMPT | structured_analyzer).invoke({
        "messages": messages[-6:] if messages else [],
        "is_waiting": "예 (제원 보충 대기 중)" if is_waiting else "아니오",
        "user_workspace": default_workspace or "미지정",
        "branch_code": branch_code or "미지정",
        "accumulated_params": json.dumps(existing_params, ensure_ascii=False),
        "query": query,
    })

    # 1) 컨텍스트 병합을 먼저 수행하여 merged_params 초기화 보장 (UnboundLocalError 원천 차단)
    old_block = existing_params.get("block_id")
    new_block = res.block_id

    if new_block and old_block and new_block != old_block:
        merged_params: Dict[str, Any] = {}
    elif res.context_action == "RESET_NEW" and not is_waiting:
        merged_params = {}
    else:
        merged_params = dict(existing_params)

    new_fields = {
        "block_id": res.block_id,
        "material": res.material,
        "thickness": res.thickness,
        "location": res.location,
        "process": res.process,
        "weld_position": res.weld_position,
        "worker_name": res.worker_name,
        "equipment": res.equipment,
    }
    for k, v in new_fields.items():
        if v is not None and str(v).strip() != "":
            merged_params[k] = str(v).strip()
    if res.additional_specs:
        merged_params.update(res.additional_specs)

    # USER.DEFAULT_WORKSPACE 및 BRANCH_CODE 기반 작업 위치 자동 바인딩
    if default_workspace and not merged_params.get("work_zone"):
        merged_params["work_zone"] = default_workspace
    if branch_code and not merged_params.get("branch_code"):
        merged_params["branch_code"] = branch_code
        
    # location은 반드시 실제 지자체 도시명('울산', '거제')이어야 기상 조회가 가능함
    if not merged_params.get("location") or "ZONE" in str(merged_params.get("location")).upper():
        if branch_code and "GEOJE" in branch_code.upper():
            merged_params["location"] = "거제"
        elif branch_code and "ULSAN" in branch_code.upper():
            merged_params["location"] = "울산"

    # 2) USER.CAN_ACCESS_ERP 권한 인터록 검증
    if can_access_erp == "N" and res.task_type in ("erp_resource_qa", "hr_worker_qa", "scm_partner_qa"):
        blocked_msg = (
            "🔒 **[접근 권한 제한]** 현재 로그인하신 계정은 전사 ERP 자원 및 인사/공급망 원장 조회 권한(`CAN_ACCESS_ERP=N`)이 없습니다. "
            "시스템 관리자(ADMIN) 또는 공정 관리자(MANAGER)에게 권한 승인을 요청해 주세요."
        )
        return {
            "task_type": "chat",
            "extracted_params": merged_params,
            "is_param_complete": "yes",
            "missing_params": [],
            "search_status": "completed",
            "final_answer": blocked_msg,
        }

    # 3) 일상 대화는 즉시 종료 유도
    if res.task_type == "chat":
        return {
            "task_type": "chat",
            "extracted_params": merged_params,
            "is_param_complete": "yes",
            "missing_params": [],
            "search_status": "completed",
            "final_answer": "안녕하세요! 스마트 조선소 공정 관제 및 작업지시 지원 에이전트입니다. 작업지시서 발행, WPS 규격 조회, ERP 자재/자격증/비전 상태 확인 등 무엇을 도와드릴까요?",
        }

    # 4) SHIP_ERP 1차 조회 (ERP_BLOCK_MASTER 제원 자동 동기화 + 재고/자격증/대기차량 검증)
    erp_status = None
    is_outsourcing = False
    if res.task_type in ("doc_generation", "erp_resource_qa", "hr_safety_qa", "scm_partner_qa"):
        target_block = merged_params.get("block_id")

        erp_status = check_hq_erp_resources.invoke({
            "block_id": target_block,
            "material": merged_params.get("material"),
            "worker_name": merged_params.get("worker_name"),
            "equipment": merged_params.get("equipment"),
            "branch_code": merged_params.get("branch_code"),
        })

        bm = erp_status.get("block_master")
        if bm:
            for bk in ("ship_code", "block_name", "material", "thickness", "joint_type", "process", "branch_code", "weight_ton"):
                if bm.get(bk) is not None:
                    merged_params[bk] = bm[bk]

            if not res.location and bm.get("location"):
                merged_params["location"] = bm["location"]

            erp_status = check_hq_erp_resources.invoke({
                "block_id": target_block,
                "material": merged_params.get("material"),
                "worker_name": merged_params.get("worker_name"),
                "equipment": merged_params.get("equipment"),
                "branch_code": merged_params.get("branch_code"),
            })

        is_outsourcing = bool(erp_status.get("require_outsourcing", False))

    if res.task_type == "scm_partner_qa":
        is_outsourcing = True

    # 5) 필수 제원(material, thickness) 누락 검사
    missing: List[str] = []
    is_complete = "yes"
    clarification_answer = None

    if res.task_type == "doc_generation":
        required_slots = {
            "material": "모재 강종",
            "thickness": "판 두께",
        }
        missing = [k for k in required_slots if not merged_params.get(k)]
        if missing:
            is_complete = "no"
            missing_kor = ", ".join(required_slots[m] for m in missing)
            block_notice = (
                f" (현재 확인된 조건: {json.dumps(merged_params, ensure_ascii=False)})"
                if merged_params else ""
            )
            clarification_answer = (
                f"작업지시서 발행을 위해 필수 제원이 추가로 필요합니다.{block_notice}\n"
                f"누락된 항목(**{missing_kor}**) 또는 대상 **블록 번호(`BLOCK_ID`)**를 입력해 주시면 "
                f"SHIP_ERP 마스터 원장과 WPS 규격을 대조하여 지시서를 작성해 드리겠습니다."
            )

    # 6) 검색 쿼리 보정
    queries = list(res.search_queries)
    if not queries and res.task_type in ("knowledge_qa", "doc_generation"):
        mat = merged_params.get("material", "")
        thk = merged_params.get("thickness", "")
        proc = merged_params.get("process", "용접")
        if res.task_type == "doc_generation" and (mat or thk):
            queries = [f"{mat} {thk} {proc} WPS 작업 표준 규격".strip()]
        else:
            queries = [query]

    need_weather_flag = True if res.task_type == "doc_generation" else res.need_weather

    return {
        "task_type": res.task_type,
        "is_action_requested": res.is_action_requested,
        "target_categories": res.target_categories,
        "need_weather": need_weather_flag,
        "extracted_params": merged_params,
        "erp_resource_status": erp_status,
        "is_outsourcing_required": is_outsourcing,
        "missing_params": missing,
        "is_param_complete": is_complete,
        "search_queries": queries,
        "retry_count": 0,
        "search_status": "idle",
        "final_answer": clarification_answer,
    }


builder = StateGraph(ShipAgentState)
builder.add_node("analyze_and_check_erp", analyze_and_check_erp_node)
builder.add_edge(START, "analyze_and_check_erp")
builder.add_edge("analyze_and_check_erp", END)

team1_subgraph = builder.compile()