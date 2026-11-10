"""
modules/agent/supervisor.py
Top-Level Supervisor & Multi-Turn Memory Graph
"""

from typing import Dict, Any, Literal
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from modules.agent.state import ShipAgentState
from modules.agent.subgraphs.analysis import team1_subgraph
from modules.agent.subgraphs.docgen import team3_subgraph
from modules.agent.subgraphs.search import team2_subgraph


def top_supervisor_init_node(state: ShipAgentState) -> Dict[str, Any]:
    """새로운 턴 시작 시 하위 서브그래프의 동적 플랜 큐를 초기화합니다."""
    return {"team2_plan": None, "next": None}


def route_after_team1(
    state: ShipAgentState,
) -> Literal["team2_search", "team3_docgen", "__end__"]:
    task_type = state.get("task_type")
    is_complete = state.get("is_param_complete", "yes")
    missing = state.get("missing_params")
    is_outsourcing = state.get("is_outsourcing_required", False)

    print("\n" + "=" * 60)
    print(f"🧭 [Top Supervisor Routing] task_type='{task_type}', is_param_complete='{is_complete}'")
    print(f"   - 누락 제원: {missing}")
    print(f"   - 사내 ERP 자원 진단: {state.get('erp_resource_status') is not None}")
    print(f"   - 외주 연계 필요 여부: {is_outsourcing}")

    if task_type == "chat" or (task_type == "doc_generation" and is_complete == "no"):
        return "__end__"

    # 순수 사내(IN_HOUSE) 조회일 때만 Team 2를 건너뛰고 Team 3으로 직행!
    # 외주 연계(is_outsourcing=True)가 필요한 경우에는 반드시 Team 2(B2B Collector)를 경유함
    if task_type in ("erp_resource_qa", "hr_safety_qa", "hr_worker_qa") and not is_outsourcing:
        return "team3_docgen"

    return "team2_search"


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
builder.add_edge("team3_docgen", END)

memory = MemorySaver()
ship_agent_app = builder.compile(checkpointer=memory)