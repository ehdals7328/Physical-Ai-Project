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
        "제공된 [참조 지식 문서(RAG)], [현장 비전 검사 데이터], [환경 기상 조건], [작업 조건]만을 근거로 작업지시서를 작성하세요.\n"
        "- 비전 데이터에 결함이나 간섭물(CLAMP_OBSTACLE 등)이 탐지된 경우 안전 주의사항에 명시하십시오.\n"
    ),
    (
        "human",
        "[작업 의도]: {task_type}\n"
        "[확정 작업 제원]: {extracted_params}\n"
        "[비전 객체 탐지 데이터]: {vision_data}\n"
        "[현장 기상 데이터]: {weather_data}\n"
        "[참조 규격 및 지침 문서 (RAG)]:\n{retrieved_docs_text}\n\n"
        "[사용자 질의]: {query}",
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
  vision = state.get("vision_data")
  docs = state.get("retrieved_docs", [])

  chain = DOCGEN_PROMPT | llm
  response = chain.invoke({
      "task_type": task_type,
      "extracted_params": (json.dumps(params, ensure_ascii=False) if params else "없음"),
      "vision_data": json.dumps(vision, ensure_ascii=False) if vision else "비전 데이터 없음",
      "weather_data": (json.dumps(weather, ensure_ascii=False) if weather else "없음"),
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