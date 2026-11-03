# agent_system/main_graph.py
import sys
from pathlib import Path
from typing import List, Literal
from pydantic import BaseModel, Field

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import ShipAgentState
from .subgraphs import team1_subgraph, team2_subgraph, team3_subgraph

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def make_top_supervisor(llm, members: List[str]):
    options = tuple(["FINISH", *members])

    class Router(BaseModel):
        next: Literal[*options] = Field(
            ..., description="다음에 호출할 하위 팀 이름 또는 작업 종료(FINISH)"
        )

    def supervisor_node(state: ShipAgentState):
        task_type = state.get("task_type")
        is_complete = state.get("is_param_complete")
        has_search = bool(state.get("retrieved_docs"))
        has_output = bool(state.get("generated_document") or state.get("final_answer"))

        system_prompt = (
            "당신은 조선소 스마트 공정 멀티에이전트 시스템의 총괄 오케스트레이터(Supervisor)입니다.\n"
            f"지휘 가능한 하위 팀 목록: {members}\n\n"
            "현재 상태를 보고 다음에 실행할 팀을 자율적으로 판단하여 선택하세요:\n"
            "1. 현재 턴의 사용자 입력에 대해 아직 분석(task_type)이 수행되지 않았다면 'team1_analysis'를 선택하세요.\n"
            "2. 'team1_analysis' 분석 결과, 일상 대화(chitchat)이거나 사용자에게 추가 정보 확인/최종 승인 질문을 던져야 하는 상태(is_param_complete == 'no' 및 final_answer 생성 완료)라면, 검색이나 문서 생성으로 넘어가지 말고 즉시 'FINISH'를 선택해 사용자와 대화하세요.\n"
            "3. 사용자의 최종 승인이 완료(is_param_complete == 'yes')되었으나 아직 사내 규격/기상 검색(retrieved_docs)이 수행되지 않았다면 'team2_search'를 선택하세요.\n"
            "4. 규격/기상 검색까지 완료되었으나 최종 산출물(generated_document 또는 final_answer)이 생성되지 않았다면 'team3_docgen'을 선택하세요.\n"
            "5. 모든 요청 처리가 완료되어 최종 답변이나 문서가 준비되었다면 'FINISH'를 선택하세요.\n\n"
            f"[현재 상태 요약]\n"
            f"- 사용자 입력: {state.get('user_query')}\n"
            f"- 분석된 태스크: {task_type}\n"
            f"- 조건 충족 및 사용자 승인 완료 여부(is_param_complete): {is_complete}\n"
            f"- 규격/기상 검색 완료 여부: {has_search}\n"
            f"- 에이전트 응답 또는 문서 준비 여부: {has_output}\n"
        )

        decision = llm.with_structured_output(
            Router, method="function_calling"
        ).invoke(system_prompt)

        return {"next": decision.next}

    return supervisor_node


members = ["team1_analysis", "team2_search", "team3_docgen"]
top_supervisor = make_top_supervisor(llm, members)

main_builder = StateGraph(ShipAgentState)

main_builder.add_node("supervisor", top_supervisor)
main_builder.add_node("team1_analysis", team1_subgraph)
main_builder.add_node("team2_search", team2_subgraph)
main_builder.add_node("team3_docgen", team3_subgraph)

main_builder.add_edge(START, "supervisor")

main_builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "team1_analysis": "team1_analysis",
        "team2_search": "team2_search",
        "team3_docgen": "team3_docgen",
        "FINISH": END,
    },
)

main_builder.add_edge("team1_analysis", "supervisor")
main_builder.add_edge("team2_search", "supervisor")
main_builder.add_edge("team3_docgen", "supervisor")

memory = MemorySaver()
ship_agent_app = main_builder.compile(checkpointer=memory)