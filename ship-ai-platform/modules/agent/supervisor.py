"""Ship Welding Autonomous Platform: 메인 오케스트레이터 (supervisor.py)

Supervisor가 State를 검사하여 Team 1(분석) -> Team 2(수집) -> Team 3(생성)의 유기적 흐름을
제어합니다.
"""

import os
from typing import Literal
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from modules.agent.state import ShipAgentState
from modules.agent.subgraphs.analysis import team1_subgraph
from modules.agent.subgraphs.docgen import team3_subgraph
from modules.agent.subgraphs.search import team2_subgraph


# ==============================================================================
# 1. Supervisor 라우팅 판정 함수
# ==============================================================================
def supervisor_router(
    state: ShipAgentState,
) -> Literal["team1_analysis", "team2_search", "team3_docgen", "__end__"]:
  task_type = state.get("task_type")
  search_status = state.get("search_status", "idle")
  final_answer = state.get("final_answer")

  # 1. Team 3에서 최종 답변(또는 역질문)이 생성되었다면 즉시 파이프라인 완전 종료 (무한루프 방지 ⭐)
  if final_answer:
    return "__end__"

  # 2. 작업 의도 분류가 아직 안 되었다면 Team 1(분석팀)으로
  if not task_type:
    return "team1_analysis"

  # 3. 단순 일상 대화(chat)는 RAG/기상 검색을 건너뛰고 바로 Team 3(생성팀)으로
  if task_type == "chat":
    return "team3_docgen"

  # 4. 전문 지식 질의(knowledge_qa) 또는 문서 발행(doc_generation):
  #    검색 미실행(idle) 상태라면 무조건 Team 2(검색팀)로 이동
  if search_status == "idle":
    return "team2_search"

  # 5. 검색이 실행 완료(completed, empty, error)되었으면 Team 3(생성팀)으로 이동
  return "team3_docgen"


# Supervisor 디버깅용 노드
def supervisor_node(state: ShipAgentState):
  task = state.get("task_type")
  status = state.get("search_status", "idle")
  has_ans = bool(state.get("final_answer"))
  print(
      f"[Supervisor] task_type={task}, search_status={status},"
      f" has_final_answer={has_ans}"
  )
  return {}


# ==============================================================================
# 2. 메인 오케스트레이터 그래프 빌드
# ==============================================================================
workflow = StateGraph(ShipAgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("team1_analysis", team1_subgraph)
workflow.add_node("team2_search", team2_subgraph)
workflow.add_node("team3_docgen", team3_subgraph)

# 진입점
workflow.add_edge(START, "supervisor")

# Supervisor 조건부 분기
workflow.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "team1_analysis": "team1_analysis",
        "team2_search": "team2_search",
        "team3_docgen": "team3_docgen",
        "__end__": END,
    },
)

# 각 팀 완료 후 supervisor로 복귀
workflow.add_edge("team1_analysis", "supervisor")
workflow.add_edge("team2_search", "supervisor")
workflow.add_edge("team3_docgen", "supervisor")

# 세션 메모리 체크포인터 장착
checkpointer = MemorySaver()
ship_agent_app = workflow.compile(checkpointer=checkpointer)