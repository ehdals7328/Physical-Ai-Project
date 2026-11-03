"""
Team 3: 맞춤형 문서 생성 및 서식 검증 서브그래프 (docgen.py)
작업 유형(task_type)에 따라 전문 문서를 생성하고, 환경 규칙 및 양식을 검증한 뒤 마크다운 파일 저장 및 이력을 기록합니다.
"""
from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from state import ShipAgentState
from tools import export_work_order_file, log_inspection_history

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ==============================================================================
# 1. Pydantic 정형 출력 스키마 정의
# ==============================================================================
class WeldWorkOrder(BaseModel):
    document_title: str = Field(default="표준 용접 작업지시서 (Work Order)", description="문서 제목")
    material: str = Field(description="모재 강종 (예: SS400, AH36)")
    thickness: str = Field(description="모재 두께 (예: 10T, 20T)")
    target_current: str = Field(description="용접 전류 (예: 220A)")
    target_voltage: str = Field(description="용접 전압 (예: 24V)")
    travel_speed: str = Field(description="용접 속도 (예: 5.0mm/s)")
    torch_angle: str = Field(description="토치 각도 (예: 70도)")
    weather_condition: Dict[str, Any] = Field(description="반영된 현장 기상 상태")
    safety_precautions: List[str] = Field(description="기상 및 규격 기반 필수 안전/공정 수칙 목록")

class HeavyLiftingPlan(BaseModel):
    document_title: str = Field(default="중량물 취급 작업계획서 (Lifting Plan)", description="문서 제목")
    target_equipment: str = Field(description="작업 대상 및 사용 장비 (예: 골리앗 크레인 블록 인양)")
    current_wind_speed: str = Field(description="실시간 풍속 정보")
    operation_permission: str = Field(description="작업 가능 여부 (작업 승인 / 작업 일시 중지)")
    safety_guidelines: List[str] = Field(description="풍속 제한 및 KOSHA 안전 수칙")

weld_llm = llm.with_structured_output(WeldWorkOrder, method="function_calling")
heavy_llm = llm.with_structured_output(HeavyLiftingPlan, method="function_calling")

# ==============================================================================
# 2. 전문 작성 에이전트 노드 정의
# ==============================================================================
def weld_order_agent_node(state: ShipAgentState) -> Dict[str, Any]:
    """[Agent 3-1: 용접 작업지시서 전문 작성]"""
    params = state.get("extracted_params", {})
    weather = state.get("weather_info", {})
    safety_eval = state.get("safety_evaluation", {})
    retrieved_docs = "\n".join(state.get("retrieved_docs", []))
    mandatory_rules = safety_eval.get("mandatory_rules", [])

    prompt = (
        "당신은 조선소 표준 공정 지시서 작성 전문가입니다. "
        "아래 제공된 사내 규격 문서와 실시간 현장 기상 데이터를 바탕으로 표준 용접 작업지시서를 작성하세요.\n\n"
        f"[작업 조건]\n"
        f"- 모재 재질: {params.get('material', 'SS400')}\n"
        f"- 모재 두께: {params.get('thickness', '10T')}\n"
        f"- 이음 형상: {params.get('joint_type', '맞대기')}\n\n"
        f"[사내 규격 인출 문서]\n{retrieved_docs}\n\n"
        f"[실시간 현장 기상 환경]\n"
        f"- 위치: {weather.get('location')}\n"
        f"- 기온: {weather.get('temperature')} ℃\n"
        f"- 풍속: {weather.get('wind_speed')} m/s\n\n"
        f"[환경 검증 도구가 판정한 필수 안전 준수사항]\n"
        f"{mandatory_rules}\n\n"
        "※ 지침: 도구가 판정한 필수 준수사항과 표준 보호구 착용 지침을 반드시 안전수칙에 충실히 반영하세요."
    )

    res: WeldWorkOrder = weld_llm.invoke(prompt)
    doc_dict = res.model_dump()
    doc_dict["weather_condition"] = weather
    return {"generated_document": doc_dict}

def heavy_plan_agent_node(state: ShipAgentState) -> Dict[str, Any]:
    """[Agent 3-2: 중량물 취급 안전계획서 전문 작성]"""
    weather = state.get("weather_info", {})
    safety_eval = state.get("safety_evaluation", {})
    retrieved_docs = "\n".join(state.get("retrieved_docs", []))
    mandatory_rules = safety_eval.get("mandatory_rules", [])

    prompt = (
        "당신은 조선소 산업안전 관리자입니다. KOSHA 지침과 실시간 기상 데이터를 반영해 중량물 작업계획서를 작성하세요.\n\n"
        f"[질문 및 조건]: {state.get('user_query')}\n"
        f"[현장 기상]: {weather}\n"
        f"[KOSHA 안전 지침 문서]:\n{retrieved_docs}\n\n"
        f"[환경 검증 도구가 판정한 필수 안전 준수사항]:\n{mandatory_rules}\n\n"
        "※ 지침: 실시간 풍속이 10m/s 초과인 경우 반드시 작업 일시 중지(STOP)를 명시하고, "
        "슬링벨트 체결 각도 및 신호수 배치 수칙을 포함하십시오."
    )

    res: HeavyLiftingPlan = heavy_llm.invoke(prompt)
    return {"generated_document": res.model_dump()}

def general_qa_agent_node(state: ShipAgentState) -> Dict[str, Any]:
    """[Agent 3-3: 도메인 기술 질의응답 전담 작성]"""
    retrieved_docs = "\n".join(state.get("retrieved_docs", []))
    weather = state.get("weather_info", {})

    prompt = (
        "당신은 선박 제조 엔지니어링 전문가입니다. 사내 표준 시방서 및 안전 규정을 근거로 "
        "사용자의 기술 질문에 대해 출처를 명시하여 전문적이고 친절하게 답변하세요.\n\n"
        f"질문: {state.get('user_query')}\n\n"
        f"참조 규격 문서:\n{retrieved_docs}\n\n"
        f"현장 기상 참고:\n{weather}"
    )

    ans = llm.invoke(prompt)
    return {
        "final_answer": ans.content,
        "generated_document": None
    }

# ==============================================================================
# 3. 서식 검증 및 파일 저장 노드 (하드코딩 제거 ➔ 도구 판정 결과 기반 반영)
# ==============================================================================
def output_validator_node(state: ShipAgentState) -> Dict[str, Any]:
    """[Agent 3-4: 서식 검증, 환경 룰 도구 판정결과 주입 및 Markdown 파일 저장]"""
    task_type = state.get("task_type", "general_qa")
    generated_doc = state.get("generated_document")
    safety_eval = state.get("safety_evaluation", {})
    file_export_msg = None

    if task_type == "weld_order" and generated_doc:
        precautions = generated_doc.get("safety_precautions", [])
        mandatory_rules = safety_eval.get("mandatory_rules", [])

        # 1. domain_tools 도구가 도출한 필수 법적/공정 수칙을 우선 주입
        for rule in mandatory_rules:
            if not any(rule[:10] in p for p in precautions):
                precautions.insert(0, rule)

        # 2. 도구 판정에 예열 항목이 없다면, LLM이 임의로 생성한 저온 예열 문구 제거 (환각 방어)
        has_preheat_rule = any("예열" in r for r in mandatory_rules)
        if not has_preheat_rule:
            precautions = [p for p in precautions if "예열" not in p]

        # 3. 필수 보호구 착용 수칙 방어
        if not any("보호구" in p for p in precautions):
            precautions.append("FCAW 용접 전용 보호구(차광 차면, 방진마스크) 착용 필수")

        generated_doc["safety_precautions"] = precautions

        # 마크다운 파일 저장 도구 호출
        file_export_msg = export_work_order_file.invoke({
            "document_data": generated_doc,
            "file_prefix": "weld_order"
        })

    elif task_type == "heavy_plan" and generated_doc:
        file_export_msg = export_work_order_file.invoke({
            "document_data": generated_doc,
            "file_prefix": "heavy_plan"
        })

    base_answer = state.get("final_answer") or "표준 문서 작성을 완료했습니다."
    final_text = f"{base_answer}\n[{file_export_msg}]" if file_export_msg else base_answer

    return {
        "generated_document": generated_doc,
        "final_answer": final_text
    }

def save_audit_node(state: ShipAgentState) -> dict:
    """[Agent 3-5: 작업 이력 감사 DB 기록 노드]"""
    session_id = state.get("session_id") or "test_session_id"
    last_query = state.get("user_query") or ""
    params = state.get("extracted_params") or {}
    messages = state.get("messages") or []

    # 1. messages 리스트에서 사용자가 입력했던 모든 대화(HumanMessage) 추출
    user_history = [
        m.content for m in messages 
        if m.__class__.__name__ == "HumanMessage"
    ]
    
    # 만약 messages에 아직 현재 입력이 안 들어갔거나 비어있다면 last_query 사용
    if not user_history and last_query:
        user_history = [last_query]

    dialogue_flow = " -> ".join(user_history)

    # 2. 전체 대화 흐름과 누적된 추출 조건(강종, 두께, 위치 등)을 하나의 문자열로 결합
    full_query_log = f"[추출조건: {params}] | 대화내역: {dialogue_flow}"

    try:
        res = log_inspection_history.invoke({
            "session_id": session_id,
            "query": full_query_log,  # 👈 마지막 단어 대신 전체 내역 전달!
            "status": "PASS"
        })
    except Exception as e:
        res = f"이력 기록 예외: {str(e)}"

    return {"audit_result": res}

# ==============================================================================
# 4. 서브그래프 라우팅 및 빌드 (save_audit_node 엣지 정상 결합)
# ==============================================================================
def route_by_task_type(state: ShipAgentState) -> Literal["weld_order_agent", "heavy_plan_agent", "general_qa_agent"]:
    task_type = state.get("task_type", "general_qa")
    if task_type == "weld_order":
        return "weld_order_agent"
    elif task_type == "heavy_plan":
        return "heavy_plan_agent"
    return "general_qa_agent"

builder = StateGraph(ShipAgentState)

builder.add_node("weld_order_agent", weld_order_agent_node)
builder.add_node("heavy_plan_agent", heavy_plan_agent_node)
builder.add_node("general_qa_agent", general_qa_agent_node)
builder.add_node("output_validator", output_validator_node)
builder.add_node("save_audit", save_audit_node)

builder.add_conditional_edges(
    START,
    route_by_task_type,
    {
        "weld_order_agent": "weld_order_agent",
        "heavy_plan_agent": "heavy_plan_agent",
        "general_qa_agent": "general_qa_agent"
    }
)

builder.add_edge("weld_order_agent", "output_validator")
builder.add_edge("heavy_plan_agent", "output_validator")
builder.add_edge("general_qa_agent", "output_validator")

# output_validator -> save_audit -> END 연결
builder.add_edge("output_validator", "save_audit")
builder.add_edge("save_audit", END)

team3_subgraph = builder.compile()