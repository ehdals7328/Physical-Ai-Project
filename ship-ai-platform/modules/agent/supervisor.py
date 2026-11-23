"""
modules/agent/supervisor.py
Top-Level Supervisor & Self-Correction Loop Multi-Turn Graph
"""

from typing import Any, Dict, Literal
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from modules.agent.state import ShipAgentState
from modules.agent.subgraphs.analysis import team1_subgraph
from modules.agent.subgraphs.docgen import team3_subgraph
from modules.agent.subgraphs.search import team2_subgraph


def top_supervisor_init_node(state: ShipAgentState) -> Dict[str, Any]:
    """
    새로운 턴 시작 시 직전 턴의 일회성 상태를 초기화합니다.
    직전 턴에 문서(DRAFT/APPROVED)가 이미 생성된 경우, 해당 블록·작업자를 기발행 이력에 기록하고
    extracted_params의 이전 문서 전용 슬롯을 비워 다음 턴에서 동일 문서가 중복 발행되지 않도록 합니다.
    """
    prev_commit = state.get("erp_commit_result") or {}
    was_already_committed = prev_commit.get("erp_transaction_status") == "COMMITTED"

    drafted_blocks = list(state.get("drafted_blocks_history") or [])
    drafted_workers = list(state.get("drafted_workers_history") or [])
    cleaned_params = dict(state.get("extracted_params") or {})

    prev_doc = state.get("generated_document") or {}
    if prev_doc.get("status") in ("DRAFT", "APPROVED"):
        prev_params = prev_doc.get("parameters") or {}
        p_blk = str(prev_params.get("block_id") or "").strip().upper()
        p_wrk = str(prev_params.get("worker_name") or "").strip()
        if p_blk and p_blk not in drafted_blocks:
            drafted_blocks.append(p_blk)
        if p_wrk and p_wrk not in drafted_workers:
            drafted_workers.append(p_wrk)

        # 이미 초안 발행이 끝난 직전 문서의 블록·작업자·공정 슬롯 비우기 (야드·구역 컨텍스트만 유지)
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

    init_updates: Dict[str, Any] = {
        "team2_plan": None,
        "next": None,
        "retrieved_docs": [],
        "doc_grade": None,
        "vision_data": None,
        "weather_data": None,
        "generated_document": None,
        "erp_commit_result": None,
        "erp_resource_status": None,
        "partner_erp_plan": None,
        "final_answer": None,
        "audit_saved": False,
        "extracted_params": cleaned_params,
        "auto_assign_mode": False,
        "replan_count": 0,
        "replan_needed": False,
        "replan_feedback": None,
        "excluded_blocks": list(drafted_blocks),
        "excluded_workers": list(drafted_workers),
        "drafted_blocks_history": drafted_blocks,
        "drafted_workers_history": drafted_workers,
        "agent_reasoning_summary": None,
        "tool_trace": [],
    }
    if was_already_committed:
        init_updates["hitl_approved"] = False

    return init_updates


def route_after_team1(
    state: ShipAgentState,
) -> Literal["team2_search", "team3_docgen", "__end__"]:
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

builder.add_edge(START, "top_supervisor_init")
builder.add_edge("top_supervisor_init", "team1_analysis")
builder.add_conditional_edges(
    "team1_analysis",
    route_after_team1,
    {
        "team2_search": "team2_search",
        "team3_docgen": "team3_docgen",
        "__end__": END,
    },
)
builder.add_edge("team2_search", "team3_docgen")
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