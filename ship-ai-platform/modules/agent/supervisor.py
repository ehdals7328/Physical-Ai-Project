"""
modules/agent/supervisor.py
Top-Level Supervisor & Multi-Turn Memory Graph
"""

from typing import Literal
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from modules.agent.state import ShipAgentState
from modules.agent.subgraphs.analysis import team1_subgraph
from modules.agent.subgraphs.docgen import team3_subgraph
from modules.agent.subgraphs.search import team2_subgraph


def route_after_team1(
    state: ShipAgentState,
) -> Literal["team2_search", "team3_docgen", "__end__"]:
    task_type = state.get("task_type")
    is_complete = state.get("is_param_complete", "yes")
    missing = state.get("missing_params")
    
    print("\n" + "=" * 60)
    print(f"🧭 [Supervisor Routing] task_type='{task_type}', is_param_complete='{is_complete}'")
    print(f"   - 누락 제원: {missing}")
    print(f"   - 사내 ERP 자원 진단: {state.get('erp_resource_status') is not None}")
    print(f"   - 외주 연계 필요 여부: {state.get('is_outsourcing_required')}")
    # 1. 일상 대화이거나, 필수 제원 누락으로 역질문(final_answer)이 이미 만들어진 경우 즉시 종료
    if task_type == "chat" or (task_type == "doc_generation" and is_complete == "no"):
        return "__end__"

    # 2. 사내 재고 단순 문의(erp_resource_qa), 인사 자격 문의(hr_safety_qa), 작업자 명단 조회(hr_worker_qa)는
    #    Team 2(RAG/기상)를 건너뛰고 Team 3 브리핑으로 직행!
    if task_type in ("erp_resource_qa", "hr_safety_qa", "hr_worker_qa"):
        return "team3_docgen"

    # 3. 그 외(knowledge_qa, scm_partner_qa, doc_generation)는 Team 2로 이동
    return "team2_search"


builder = StateGraph(ShipAgentState)

builder.add_node("team1_analysis", team1_subgraph)
builder.add_node("team2_search", team2_subgraph)
builder.add_node("team3_docgen", team3_subgraph)

builder.add_edge(START, "team1_analysis")
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