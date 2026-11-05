"""
Team 3: 맞춤형 문서 생성 및 Function Calling 배포 서브그래프 (docgen.py)
- Node 1 (work_order_writer): 정상 및 조건부 승인(PASS / CONDITIONAL_PASS) 시 표준 작업지시서 작성
- Node 2 (troubleshooter_agent): 강풍(STOP), 필수 제원 누락 시 현장 대책/역질문 생성
- Node 3 (domain_qa_writer): 일반 기술/규격/날씨 질의 응답 전담
- Node 4 (output_validator): Function Calling(bind_tools)으로 마크다운 파일 저장 및 감사 로그 도구 실행
"""

import json
from typing import Any, Dict, List, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from modules.agent.state import ShipAgentState
from modules.agent.tools.document_tools import export_work_order_file, log_inspection_history

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
tool_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def format_retrieved_docs(docs: List[Dict[str, Any]]) -> str:
    if not docs:
        return "참조된 사내 규격 문서 없음"
    formatted = []
    for idx, d in enumerate(docs, 1):
        meta = d.get("metadata", {}) or {}
        doc_name = meta.get("document_name") or d.get("category", "규격")
        page = meta.get("page") or 1
        content = d.get("content", "")
        formatted.append(f"[{idx}] 출처: {doc_name} (p.{page})\n{content}")
    return "\n\n".join(formatted)


def route_generation_task(
    state: ShipAgentState,
) -> Literal["troubleshooter_agent", "work_order_writer", "domain_qa_writer"]:
    task_type = state.get("task_type", "knowledge_qa")
    
    # 1. 일반 질문은 즉시 전문 QA로 분기
    if task_type != "doc_generation":
        print(f"\n👉 [Team 3 라우팅]: domain_qa_writer (사유: 일반 기술/기상 질의, task_type='{task_type}')")
        return "domain_qa_writer"

    # 2. 작업지시서 생성 시 환경 및 안전 조건 검증
    weather = state.get("weather_data") or {}
    safety_eval = weather.get("safety_evaluation") or {}
    vision = state.get("vision_data") or {}
    missing = state.get("missing_params", [])
    
    weather_status = weather.get("status")
    safety_status = safety_eval.get("status")

    # 비전이 연결되지 않았을 때는 항상 안전(정상)으로 통과 처리
    detections = vision.get("detections") or []
    has_vision_obstacle = any(
        det.get("is_hazard") is True or 
        "OBSTACLE" in str(det.get("class_name", "")).upper()
        for det in detections
    ) if detections else False

    # 안전 승인 조건 (CONDITIONAL_PASS도 표준 지시서 작성 경로로 정상 수용)
    is_weather_ok = (weather_status == "SUCCESS")
    is_safety_ok = (safety_status in ("PASS", "CONDITIONAL_PASS"))
    is_params_ok = (not missing)
    is_vision_ok = (not has_vision_obstacle)

    is_fully_approved = (
        is_weather_ok and
        is_safety_ok and
        is_params_ok and
        is_vision_ok
    )

    print("\n" + "=" * 65)
    print("🔍 [Team 3 라우터 화이트리스트 검사 현황]")
    print(f"   1. 날씨 상태 (SUCCESS 여부) : {is_weather_ok} (현재: '{weather_status}')")
    print(f"   2. 안전 판정 (승인 범위)    : {is_safety_ok} (현재: '{safety_status}')")
    print(f"   3. 필수 제원 (누락 없음 여부): {is_params_ok} (누락목록: {missing})")
    print(f"   4. 비전 검사 (장애물 없음)   : {is_vision_ok} (현재 비전 미연동/정상)")
    
    if is_fully_approved:
        print("👉 [결정]: work_order_writer (작업지시서 작성 노드 진입)")
        print("=" * 65 + "\n")
        return "work_order_writer"
    
    fail_reasons = []
    if not is_weather_ok: fail_reasons.append(f"날씨미달({weather_status})")
    if not is_safety_ok: fail_reasons.append(f"안전평가미달({safety_status})")
    if not is_params_ok: fail_reasons.append(f"제원누락({missing})")
    if not is_vision_ok: fail_reasons.append("비전장애물감지")

    print(f"👉 [결정]: troubleshooter_agent (사유: {', '.join(fail_reasons)})")
    print("=" * 65 + "\n")

    return "troubleshooter_agent"


# ==============================================================================
# Node 1: 표준 작업지시서 작성 Agent (work_order_writer)
# ==============================================================================
WORK_ORDER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 조선소 공정 총괄 엔지니어 AI입니다.\n"
     "제공된 [참조 규격 문서(RAG)], [확정 작업 제원], [현장 기상 및 안전 판정]을 근거로 "
     "현장 작업자용 표준 작업지시서를 구조화된 마크다운으로 작성하세요.\n\n"
     "- 현장 기상 및 안전 판정(safety_evaluation)의 조건부 수칙(방풍막 설치, 예열 여부 등)을 작업 지침에 반드시 포함하십시오.\n"
     "- 사내 규격서에서 인용한 권장 수치(전류/전압 범위 등)와 출처 문서명을 명시하십시오."),
    ("human",
     "[확정 작업 제원]: {params}\n"
     "[현장 기상 및 안전 판정]: {weather}\n"
     "[참조 규격 문서(RAG)]:\n{docs}\n\n"
     "[사용자 요청]: {query}")
])


def work_order_writer_node(state: ShipAgentState) -> Dict[str, Any]:
    print("🚀 >>> [실행 노드]: work_order_writer_node 진입 (표준 지시서 작성)")
    params = state.get("extracted_params", {}) or {}
    weather = state.get("weather_data") or {}
    docs = state.get("retrieved_docs", [])
    query = state.get("user_query", "")

    res = (WORK_ORDER_PROMPT | llm).invoke({
        "params": json.dumps(params, ensure_ascii=False),
        "weather": json.dumps(weather, ensure_ascii=False) if weather else "기상 정보 없음",
        "docs": format_retrieved_docs(docs),
        "query": query
    })

    answer = res.content
    
    # 🌟 실제 안전 평가 결과 상태(PASS 또는 CONDITIONAL_PASS)를 그대로 보존
    actual_status = (weather.get("safety_evaluation") or {}).get("status", "PASS")
    doc_title = f"{params.get('material', '표준')} {params.get('thickness', '')} 작업지시서".strip()

    generated_doc = {
        "title": doc_title,
        "status": actual_status,
        "parameters": params,
        "weather": weather,
        "content": answer
    }
    return {
        "final_answer": answer,
        "generated_document": generated_doc,
        "is_param_complete": "yes"
    }


# ==============================================================================
# Node 2: 공정 보정·트러블슈팅 및 역질문 Agent (troubleshooter_agent)
# ==============================================================================
TROUBLESHOOT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 조선소 현장 안전 감리 및 트러블슈팅 전문 AI입니다.\n"
     "현재 작업 요청에 대해 [필수 조건 누락] 또는 [기상 악화(STOP)]가 발생했습니다.\n\n"
     "[대응 원칙]\n"
     "1. [누락된 필수 파라미터]에 명시된 현장 조건(작업 위치, 대상 사양 등)만 사용자에게 되물으세요.\n"
     "   - 사내 규격(RAG)에서 참조해야 하는 공학 기준값은 사용자에게 되묻지 마십시오.\n"
     "2. 강풍(STOP) 등으로 작업이 불가한 경우: 즉시 '작업 보류(HOLD/STOP)' 판정과 보정 조치를 제시하십시오."),
    ("human",
     "[누락된 필수 파라미터]: {missing}\n"
     "[입력된 작업 제원]: {params}\n"
     "[현장 기상 및 안전 판정]: {weather}\n"
     "[참조 규격 문서(RAG)]:\n{docs}\n\n"
     "[사용자 질의]: {query}")
])


def troubleshooter_node(state: ShipAgentState) -> Dict[str, Any]:
    print("⚠️ >>> [실행 노드]: troubleshooter_node 진입 (안전 미달/조건부 보류 조치)")
    missing = state.get("missing_params", [])
    params = state.get("extracted_params", {}) or {}
    weather = state.get("weather_data") or {}
    docs = state.get("retrieved_docs", [])
    query = state.get("user_query", "")

    if weather.get("status") == "NEED_LOCATION" and state.get("task_type") != "doc_generation":
        return {
            "final_answer": weather.get("message", "기상 확인을 위해 작업 위치(조선소/지역명)를 알려주세요."),
            "generated_document": None,
            "is_param_complete": "no"
        }

    res = (TROUBLESHOOT_PROMPT | llm).invoke({
        "missing": missing,
        "params": json.dumps(params, ensure_ascii=False),
        "weather": json.dumps(weather, ensure_ascii=False),
        "docs": format_retrieved_docs(docs),
        "query": query
    })

    answer = res.content
    is_hold_doc = state.get("task_type") == "doc_generation" and not missing
    generated_doc = None
    if is_hold_doc:
        generated_doc = {
            "title": "공정 안전 보류(HOLD) 및 현장 조치 지시서",
            "status": "HOLD",
            "parameters": params,
            "weather": weather,
            "content": answer
        }

    return {
        "final_answer": answer,
        "generated_document": generated_doc,
        "is_param_complete": "no" if missing else "yes"
    }


# ==============================================================================
# Node 3: 도메인 Q&A 및 실시간 기상 응답 Agent (domain_qa_writer)
# ==============================================================================
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 선박 제조 기술 및 현장 환경 안내 전문 AI입니다.\n"
     "- [현장 기상 데이터]가 제공된 경우 현재 위치의 기온, 풍속, 습도 및 안전 판정 사항을 정확히 안내하세요.\n"
     " - 영문 지명(예: Geoje, Ulsan 등)은 자연스러운 '한글 지명(거제, 울산 등)'으로 변환하여 안내하세요.\n"
     "- [참조 규격 문서(RAG)]가 제공된 경우 문서의 수치와 기준을 인용하여 답변하고 출처 문서명을 밝히세요."),
    ("human",
     "[현장 기상 데이터]: {weather}\n"
     "[참조 규격 문서(RAG)]:\n{docs}\n\n"
     "[사용자 질의]: {query}")
])


def domain_qa_writer_node(state: ShipAgentState) -> Dict[str, Any]:
    print("💬 >>> [실행 노드]: domain_qa_writer_node 진입 (일반 기술 Q&A 작성)")
    weather = state.get("weather_data")
    docs = state.get("retrieved_docs", [])
    query = state.get("user_query", "")

    res = (QA_PROMPT | llm).invoke({
        "weather": json.dumps(weather, ensure_ascii=False) if weather else "해당 없음",
        "docs": format_retrieved_docs(docs),
        "query": query
    })

    return {
        "final_answer": res.content,
        "generated_document": None,
        "is_param_complete": "yes"
    }


# ==============================================================================
# Node 4: Function Calling 기반 파일 발행 및 감사 로깅 Agent (output_validator)
# ==============================================================================
llm_with_doc_tools = tool_llm.bind_tools([export_work_order_file, log_inspection_history])

AUDIT_DISPATCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 문서 발행 및 감사 기록 관제 AI입니다.\n"
     "1. [생성된 문서 객체]가 존재하면 `export_work_order_file` 도구를 호출하여 마크다운 파일로 저장하세요.\n"
     "2. 모든 요청에 대해 `log_inspection_history` 도구를 호출하여 감사 이력을 기록하세요."),
    ("human",
     "[사용자 질의]: {query}\n[생성된 문서 객체 존재 여부]: {has_doc}\n[판정 상태]: {status}")
])


def output_validator_node(state: ShipAgentState) -> Dict[str, Any]:
    gen_doc = state.get("generated_document")
    query = state.get("user_query", "")
    status_flag = gen_doc.get("status", "PASS") if isinstance(gen_doc, dict) else "QA_PASS"

    try:
        ai_msg = (AUDIT_DISPATCH_PROMPT | llm_with_doc_tools).invoke({
            "query": query,
            "has_doc": bool(gen_doc),
            "status": status_flag
        })

        audit_saved = False
        for tc in getattr(ai_msg, "tool_calls", []):
            if tc["name"] == "export_work_order_file" and isinstance(gen_doc, dict):
                export_work_order_file.invoke({
                    "document_data": gen_doc,
                    "file_prefix": tc["args"].get("file_prefix", "work_order")
                })
                audit_saved = True
            elif tc["name"] == "log_inspection_history":
                log_inspection_history.invoke({
                    "session_id": "AGENT_RUNTIME",
                    "query": query,
                    "status": status_flag
                })
                audit_saved = True
        return {"audit_saved": audit_saved}
    except Exception as e:
        print(f"[Team 3 Validator Warning]: 도구 실행 생략 ({e})")
        return {"audit_saved": False}


# ==============================================================================
# 서브그래프 조립
# ==============================================================================
builder = StateGraph(ShipAgentState)

builder.add_node("work_order_writer", work_order_writer_node)
builder.add_node("troubleshooter_agent", troubleshooter_node)
builder.add_node("domain_qa_writer", domain_qa_writer_node)
builder.add_node("output_validator", output_validator_node)

builder.add_conditional_edges(
    START,
    route_generation_task,
    {
        "work_order_writer": "work_order_writer",
        "troubleshooter_agent": "troubleshooter_agent",
        "domain_qa_writer": "domain_qa_writer"
    }
)

builder.add_edge("work_order_writer", "output_validator")
builder.add_edge("troubleshooter_agent", "output_validator")
builder.add_edge("domain_qa_writer", "output_validator")
builder.add_edge("output_validator", END)

team3_subgraph = builder.compile()