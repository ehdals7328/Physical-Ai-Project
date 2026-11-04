"""Team 3: 문서 및 최종 응답 생성 서브그래프 (docgen.py) RAG 인출 문서의 서식 기준을 대조하여, 누락된

제원이 있을 경우 객관적 근거에 기반해 역질문합니다.
"""

import json
from typing import Any, Dict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from modules.agent.state import ShipAgentState

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

DOCGEN_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 선박 제조 엔지니어링 및 공정 총괄 책임 AI입니다.\n"
        "오직 제공된 [참조 지식 문서(RAG)]와 [현장 데이터], [확정된 작업"
        " 조건]만을 근거로 판단하세요.\n\n"
        "[작업 지침]\n"
        "1. task_type이 'chat'인 경우: 시스템 안내 및 친절한 대화 응답.\n"
        "2. task_type이 'knowledge_qa'인 경우: 참조 지식 문서에 기반해 정확한"
        " 수치와 출처를 제시하며 답변.\n"
        "3. task_type이 'doc_generation'(작업지시서/계획서 작성 요청)인 경우"
        " (⭐ 핵심 규칙):\n"
        "   - 먼저 [참조 지식 문서]에 명시된 해당 공정의 필수 기재 항목(예:"
        " 모재 강종, 판 두께, 용접 방법 등)을 확인하세요.\n"
        "   - 현재 [확정된 작업 조건]과 대조하여, 규격서 작성에 필수적인 정보가"
        " 결여되어 있다면 억지로 문서를 발행하지 마십시오.\n"
        "   - 대신 [참조 지식 문서]의 규정에 따라 지시서 완성을 위해 정확히"
        " 어떤 제원들이 추가로 확인되어야 하는지 사내 규격 근거를 들어 작업자에게"
        " 정중히 되물으세요.\n"
        "   - 모든 필수 조건이 갖추어졌다면 표준 양식에 부합하는 정식 지시서를"
        " 상세히 작성하세요.",
    ),
    (
        "human",
        "[작업 의도]: {task_type}\n"
        "[현재 확정된 작업 제원]: {extracted_params}\n"
        "[현장 실시간 기상 데이터]: {weather_data}\n"
        "[참조 규격 및 지침 문서 (RAG 인출 결과)]:\n"
        "{retrieved_docs_text}\n\n"
        "[사용자 질의 원문]: {query}",
    ),
])


def format_retrieved_docs(docs: List[Dict[str, Any]]) -> str:
  if not docs:
    return "인출된 사내 규격 문서 없음 (empty)"
  formatted = []
  for idx, d in enumerate(docs, 1):
    cat = d.get("category", "규격")
    content = d.get("content", "")
    formatted.append(f"[{idx}] (카테고리: {cat})\n{content}")
  return "\n\n".join(formatted)


def generate_response_node(state: ShipAgentState) -> Dict[str, Any]:
  task_type = state.get("task_type", "knowledge_qa")
  query = state.get("user_query", "")
  params = state.get("extracted_params", {}) or {}
  weather = state.get("weather_data")
  docs = state.get("retrieved_docs", [])

  chain = DOCGEN_PROMPT | llm
  response = chain.invoke({
      "task_type": task_type,
      "extracted_params": (
          json.dumps(params, ensure_ascii=False) if params else "없음"
      ),
      "weather_data": (
          json.dumps(weather, ensure_ascii=False) if weather else "없음"
      ),
      "retrieved_docs_text": format_retrieved_docs(docs),
      "query": query,
  })

  answer_text = response.content

  # 문서 작성이 완결된 경우만 generated_document에 담음
  is_complete_doc = (
      task_type == "doc_generation"
      and "추가" not in answer_text
      and "누락" not in answer_text
  )
  generated_doc = None
  if is_complete_doc:
    generated_doc = {
        "title": "공식 표준 작업지시서",
        "parameters": params,
        "content": answer_text,
    }

  return {
      "final_answer": answer_text,
      "generated_document": generated_doc,
      "is_param_complete": "yes" if is_complete_doc else "no",
  }


builder = StateGraph(ShipAgentState)
builder.add_node("generate_response", generate_response_node)
builder.add_edge(START, "generate_response")
builder.add_edge("generate_response", END)
team3_subgraph = builder.compile()