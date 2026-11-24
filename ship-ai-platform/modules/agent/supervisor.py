"""
modules/agent/supervisor.py
Top-Level Supervisor & Self-Correction Loop Multi-Turn Graph (Team 1~4 통합)
"""

from typing import Any, Dict, Literal
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from modules.agent.state import ShipAgentState
from modules.agent.subgraphs.analysis import team1_subgraph
from modules.agent.subgraphs.docgen import team3_subgraph
from modules.agent.subgraphs.planning import team4_subgraph
from modules.agent.subgraphs.search import team2_subgraph


def top_supervisor_init_node(state: ShipAgentState) -> Dict[str, Any]:
    """
    새로운 턴 시작 시 상태를 두 영역으로 명확히 분리하여 초기화합니다.
    1) 누적 유지 데이터: 기발행 블록·작업자 이력, 직전 역질문 대기 플래그(prev_turn_waiting_clarification),
       직전 미완료 문서의 누적 extracted_params
    2) 일회성 실행 데이터: hitl_approved, is_param_complete, missing_params, is_action_requested 등 매 턴 초기화
    """
    prev_commit = state.get("erp_commit_result") or {}
    prev_commit_status = str(prev_commit.get("erp_transaction_status") or "").upper()
    was_already_committed = prev_commit_status == "COMMITTED"

    drafted_blocks = list(state.get("drafted_blocks_history") or [])
    drafted_workers = list(state.get("drafted_workers_history") or [])
    cleaned_params = dict(state.get("extracted_params") or {})

    prev_doc = state.get("generated_document") or {}
    prev_doc_status = str(prev_doc.get("status") or "").upper()
    prev_doc_id = str(prev_doc.get("doc_id") or prev_doc.get("work_no") or "").strip()
    prev_doc_ver = str(prev_doc.get("version") or "1").strip()

    # 직전 턴 역질문 대기 여부를 별도 필드로 이관한 뒤 현재 턴 상태는 초기화
    prev_turn_waiting = state.get("is_param_complete") == "no"

    # 일회성 HITL 승인 검증:
    # 기본값은 매 턴 False이며, 명시적 승인 대상 문서 ID·버전이 직전 미커밋 DRAFT와 일치할 때만 1회 허용
    req_doc_id = str(state.get("hitl_target_doc_id") or "").strip()
    req_doc_ver = str(state.get("hitl_target_doc_version") or "1").strip()
    valid_hitl_approval = (
        bool(state.get("hitl_approved"))
        and not was_already_committed
        and prev_commit_status not in ("FAILED", "ROLLED_BACK", "ERROR")
        and prev_doc_status == "DRAFT"
        and bool(prev_doc_id)
        and req_doc_id == prev_doc_id
        and req_doc_ver == prev_doc_ver
    )

    if prev_doc_status in ("DRAFT", "APPROVED") and not valid_hitl_approval:
        prev_params = prev_doc.get("parameters") or {}
        p_blk = str(prev_params.get("block_id") or "").strip().upper()
        p_wrk = str(prev_params.get("worker_name") or "").strip()
        if p_blk and p_blk not in drafted_blocks:
            drafted_blocks.append(p_blk)
        if p_wrk and p_wrk not in drafted_workers:
            drafted_workers.append(p_wrk)

        for doc_key in (
            "block_id",
            "block_name",
            "ship_code",
            "material",
            "thickness",
            "joint_type",
            "process",
            "weight_ton",
            "worker_name",
            "equipment",
            "doc_type",
        ):
            cleaned_params.pop(doc_key, None)

    return {
        # [1] 누적 대화 컨텍스트 (유지 영역)
        "extracted_params": cleaned_params,
        "excluded_blocks": list(drafted_blocks),
        "excluded_workers": list(drafted_workers),
        "drafted_blocks_history": drafted_blocks,
        "drafted_workers_history": drafted_workers,
        "prev_turn_waiting_clarification": prev_turn_waiting,

        # [2] 일회성 실행 상태 (매 턴 초기화 영역)
        "hitl_approved": valid_hitl_approval,
        "hitl_target_doc_id": None,
        "hitl_target_doc_version": None,
        "is_param_complete": "yes",
        "missing_params": [],
        "is_action_requested": False,
        "is_outsourcing_required": False,
        "need_weather": False,
        "search_queries": [],
        "retry_count": 0,
        "team2_plan": None,
        "next": None,
        "retrieved_docs": [],
        "doc_grade": None,
        "vision_data": None,
        "weather_data": None,
        "generated_document": prev_doc if valid_hitl_approval else None,
        "cad_staging_result": None,
        "erp_commit_result": None,
        "erp_resource_status": None,
        "partner_erp_plan": None,
        "final_answer": None,
        "audit_saved": False,
        "auto_assign_mode": False,
        "replan_count": 0,
        "replan_needed": False,
        "replan_feedback": None,
        "agent_reasoning_summary": None,
        "tool_trace": [],
    }


def route_after_team1(
    state: ShipAgentState,
) -> Literal["team2_search", "team3_docgen", "team4_planning", "__end__"]:
    task_type = state.get("task_type")
    is_complete = state.get("is_param_complete", "yes")
    is_outsourcing = state.get("is_outsourcing_required", False)
    auto_assign = state.get("auto_assign_mode", False)

    print("\n" + "=" * 60)
    print(
        f"🧭 [Top Supervisor Routing] task='{task_type}', "
        f"complete='{is_complete}', auto_assign={auto_assign}, outsourcing={is_outsourcing}"
    )

    if task_type == "chat" or (task_type == "doc_generation" and is_complete == "no"):
        return "__end__"

    if task_type == "cad_planning":
        return "team4_planning"

    # 순수 사내 단건/원장 조회는 빠르게 Team 3 브리핑으로 직행
    if task_type in ("erp_resource_qa", "hr_safety_qa", "hr_worker_qa") and not is_outsourcing:
        return "team3_docgen"

    return "team2_search"


def route_after_team3(
    state: ShipAgentState,
) -> Literal["team2_search", "__end__"]:
    """
    Team 3 거버넌스 게이트키퍼가 복구 가능한 결격을 감지한 경우
    Team 2 ReAct 에이전트로 제어권을 되돌려(Back-edge) 자율 재편성을 수행합니다.
    """
    if state.get("replan_needed") and int(state.get("replan_count", 0)) <= 2:
        print(
            f"🔁 [Top Supervisor Self-Correction Loop] "
            f"Team 3 ➔ Team 2 재편성 진입 (시도 #{state.get('replan_count')})"
        )
        return "team2_search"
    return "__end__"


builder = StateGraph(ShipAgentState)

builder.add_node("top_supervisor_init", top_supervisor_init_node)
builder.add_node("team1_analysis", team1_subgraph)
builder.add_node("team2_search", team2_subgraph)
builder.add_node("team3_docgen", team3_subgraph)
builder.add_node("team4_planning", team4_subgraph)

builder.add_edge(START, "top_supervisor_init")
builder.add_edge("top_supervisor_init", "team1_analysis")
builder.add_conditional_edges(
    "team1_analysis",
    route_after_team1,
    {
        "team2_search": "team2_search",
        "team3_docgen": "team3_docgen",
        "team4_planning": "team4_planning",
        "__end__": END,
    },
)
builder.add_edge("team2_search", "team3_docgen")
builder.add_edge("team4_planning", END)
builder.add_conditional_edges(
    "team3_docgen",
    route_after_team3,
    {
        "team2_search": "team2_search",
        "__end__": END,
    },
)

memory = MemorySaver()
ship_agent_app = builder.compile(checkpointer=memory)