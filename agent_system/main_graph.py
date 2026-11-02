# agent_system/main_graph.py
from typing import List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from state import ShipAgentState
from subgraphs import team1_subgraph, team2_subgraph, team3_subgraph

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def make_top_supervisor(llm, members: List[str]):
    options = tuple(["FINISH", *members])
    
    class Router(BaseModel):
        next: Literal[*options] = Field(
            ..., description="다음에 실행할 팀 이름 또는 FINISH"
        )

    def supervisor_node(state: ShipAgentState):
        user_query = state.get("user_query", "")
        task_type = state.get("task_type")
        is_param_complete = state.get("is_param_complete")
        has_search = bool(state.get("retrieved_docs"))
        has_doc = bool(state.get("generated_document") or state.get("final_answer"))

        system_prompt = (
            "당신은 조선소 스마트 공정의 총괄 관리자(Supervisor)입니다.\n"
            f"당신이 지휘할 수 있는 하위 전문 팀 목록: {members}\n\n"
            "작업자의 최신 질의와 현재 공정 단계 진행 상황을 검토하여, 다음에 작업을 수행할 팀을 반드시 하나만 선택하세요.\n\n"
            "[공정 순서 및 호출 규칙]:\n"
            "0. ★ 최우선 규칙: Team 1에서 필수 정보 누락으로 사용자 역질문이 생성되었거나 정보가 불완전하다면(is_param_complete=='no') ➔ 추가 검색하지 말고 즉시 'FINISH'를 선택하여 작업자에게 되물어보세요.\n"
            "1. 의도 분류(task_type)나 파라미터 분석이 완료되지 않았다면 ➔ 무조건 'team1_analysis'를 선택하세요.\n"
            "2. 요구 분석(task_type)은 끝났으나 사내 규격(retrieved_docs)이나 기상 검색이 안 되어 있다면 ➔ 무조건 'team2_search'를 선택하세요.\n"
            "3. 규격/기상 검색은 완료되었으나 최종 표준 문서나 답변이 완성되지 않았다면 ➔ 'team3_docgen'을 선택하세요.\n"
            "4. 현재 질문에 대한 최종 문서나 답변 생성이 모두 끝났다면 ➔ 'FINISH'를 선택하세요.\n\n"
            f"[현재 공정 상태 점검]:\n"
            f"- 작업자 최신 질의: \"{user_query}\"\n"
            f"- 1단계 (요구 분석 완료 여부): {bool(task_type)} (태스크: {task_type})\n"
            f"- 필수 파라미터 충족 여부: {is_param_complete}\n"
            f"- 2단계 (규격/기상 검색 여부): {has_search}\n"
            f"- 3단계 (최종 문서/답변 여부): {has_doc}\n"
        )

        decision = llm.with_structured_output(
            Router, method="function_calling"
        ).invoke(system_prompt)

        return {"next": decision.next}

    return supervisor_node


members = ["team1_analysis", "team2_search", "team3_docgen"]
top_supervisor = make_top_supervisor(llm, members)

main_builder = StateGraph(ShipAgentState)

# 최상위 노드 등록
main_builder.add_node("supervisor", top_supervisor)
main_builder.add_node("team1_analysis", team1_subgraph)
main_builder.add_node("team2_search", team2_subgraph)
main_builder.add_node("team3_docgen", team3_subgraph)

# 시작 노드 지정
main_builder.add_edge(START, "supervisor")

# Supervisor의 LLM 판단에 따른 조건부 분기
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

# 각 하위 팀 작업 완료 후 Supervisor로 복귀
main_builder.add_edge("team1_analysis", "supervisor")
main_builder.add_edge("team2_search", "supervisor")
main_builder.add_edge("team3_docgen", "supervisor")

# MemorySaver 체크포인터 장착 (멀티턴 세션 기억)
memory = MemorySaver()
ship_agent_app = main_builder.compile(checkpointer=memory)