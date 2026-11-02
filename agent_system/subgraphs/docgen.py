from typing import Dict, List, Literal, Optional, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from state import ShipAgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --------------------------------------------------------------------------
# 1. Pydantic 스키마 정의
# --------------------------------------------------------------------------
class WeldWorkOrder(BaseModel):
    document_title: str = Field(default="표준 용접 작업지시서 (Work Order)")
    material: str = Field(..., description="모재 강종 (예: SS400, AH36)")
    thickness: str = Field(..., description="모재 판 두께 (예: 10T)")
    target_current: str = Field(..., description="권장 용접 전류 (A 단위)")
    target_voltage: str = Field(..., description="권장 용접 전압 (V 단위)")
    travel_speed: str = Field(..., description="권장 주행 속도 (mm/s 단위)")
    torch_angle: str = Field(..., description="권장 토치 각도")
    weather_condition: Optional[Dict[str, Any]] = Field(default=None, description="작업 현장 기상 데이터")
    safety_precautions: List[str] = Field(
        default_factory=list,
        description="기온, 풍속 및 현장 기본 규정에 따른 필수 안전 수칙 목록 (최소 2건 이상)"
    )

class HeavyLiftingPlan(BaseModel):
    document_title: str = Field(default="중량물 취급 및 인양 작업계획서 (Safety Plan)")
    target_object: str = Field(..., description="인양 대상 블록/설비 명칭")
    sling_angle_limit: str = Field(default="60도 이내", description="슬링벨트 결속 각도 제한")
    safety_factor: str = Field(default="5 이상", description="와이어로프 법정 안전계수")
    wind_status: str = Field(..., description="현장 풍속 및 작업 허용 여부 판정")
    weather_condition: Optional[Dict[str, Any]] = Field(default=None, description="작업 현장 기상 데이터")
    safety_precautions: List[str] = Field(
        default_factory=list,
        description="강풍 대응 및 낙하 방지 안전 조치 목록"
    )

class TechnicalQAAnswer(BaseModel):
    summary: str = Field(..., description="기술 질의에 대한 명확한 핵심 답변 요약")
    standard_reference: str = Field(..., description="근거 사내 WPS 규정 및 KOSHA 기준 조항")
    detailed_guidance: List[str] = Field(default_factory=list, description="세부 작업 수칙")

# --------------------------------------------------------------------------
# 2. 전문 작성 노드
# --------------------------------------------------------------------------
def weld_order_agent_node(state: ShipAgentState) -> Dict:
    """[Agent 1: 용접 지시서 전문 작성] WPS 규격 + 현장 기상(기온, 풍속) 결합 지시서 작성"""
    user_query = state.get("user_query", "")
    params = state.get("extracted_params", {})
    docs = state.get("retrieved_docs", [])
    weather = state.get("weather_info", {})

    system_prompt = (
        "당신은 조선소 생산기술팀 용접 시방서 전문 엔지니어입니다.\n"
        "인출된 [사내 규격 문서]와 현장 [실시간 기상 데이터]를 엄격히 분석하여 '표준 용접 작업지시서'를 작성하세요.\n\n"
        "※ 안전수칙(safety_precautions) 필수 작성 원칙:\n"
        "1. [외기 온도 판정]: 외기온도가 5℃ 이하인 경우, '모재 온도가 5℃ 이하이므로 100℃ 이상 예열 필수'를 명시하세요.\n"
        "2. [풍속 환경 판정]: 실시간 풍속이 2.0m/s 이상인 경우, 보호가스(Shielding Gas) 날림 방지를 위한 '방풍막(Wind Screen) 설치 필수' 수칙을 반드시 추가하세요.\n"
        "3. [기본 안전 수칙]: 기상 영향이 없더라도 'FCAW 용접 전용 보호구(차광 차면, 방진마스크) 착용' 등 현장 필수 수칙을 반드시 1건 이상 포함하세요.\n"
        "4. 안전수칙(safety_precautions)은 절대로 빈 배열([])로 두지 마세요.\n\n"
        f"[추출 파라미터]: {params}\n"
        f"[사내 규격 문서]:\n{chr(10).join(docs)}\n"
        f"[현장 기상]: {weather}"
    )

    order = llm.with_structured_output(
        WeldWorkOrder,
        method="function_calling"
    ).invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ])

    return {
        "generated_document": order.model_dump(),
        "final_answer": "현장 기상 및 사내 WPS 규정을 반영한 표준 용접 작업지시서 작성을 완료했습니다."
    }

def heavy_plan_agent_node(state: ShipAgentState) -> Dict:
    """[Agent 2: 중량물 계획서 작성]"""
    user_query = state.get("user_query", "")
    params = state.get("extracted_params", {})
    docs = state.get("retrieved_docs", [])
    weather = state.get("weather_info", {})

    system_prompt = (
        "당신은 조선소 산업안전보건 관리자입니다.\n"
        "인출된 [안전 지침]과 현장 [기상 데이터]를 반영해 중량물 작업계획서를 작성하세요.\n"
        "실시간 풍속이 10m/s 초과 시 인양 작업을 일시 중지하도록 판정하세요.\n\n"
        f"[추출 파라미터]: {params}\n"
        f"[사내 지침 문서]:\n{chr(10).join(docs)}\n"
        f"[현장 기상]: {weather}"
    )

    plan = llm.with_structured_output(
        HeavyLiftingPlan,
        method="function_calling"
    ).invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ])

    return {
        "generated_document": plan.model_dump(),
        "final_answer": "실시간 풍속 및 KOSHA 안전 지침을 반영한 중량물 취급계획서 작성을 완료했습니다."
    }

def general_qa_agent_node(state: ShipAgentState) -> Dict:
    """[Agent 3: 기술 Q&A 작성]"""
    user_query = state.get("user_query", "")
    docs = state.get("retrieved_docs", [])

    system_prompt = (
        "당신은 조선소 기술 상담 엔지니어입니다. 제공된 사내 규격 문서만을 근거로 답변하세요.\n\n"
        f"[사내 규정 문서]:\n{chr(10).join(docs)}"
    )

    answer = llm.with_structured_output(
        TechnicalQAAnswer,
        method="function_calling"
    ).invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ])

    return {
        "generated_document": answer.model_dump(),
        "final_answer": answer.summary
    }

def output_validator_node(state: ShipAgentState) -> Dict:
    """[Agent 4: 서식 검증 및 안전수칙 Fallback 방어 노드]"""
    generated_doc = state.get("generated_document", {})
    task_type = state.get("task_type", "general_qa")
    weather = state.get("weather_info", {})

    if task_type == "weld_order" and generated_doc:
        precautions = generated_doc.get("safety_precautions", [])
        wind_speed = float(weather.get("wind_speed", 0.0))
        temp = float(weather.get("temperature", 20.0))

        # 1. 풍속 2.0m/s 이상인데 방풍 수칙이 누락된 경우 강제 주입
        has_wind_rule = any("방풍" in p or "풍속" in p for p in precautions)
        if wind_speed >= 2.0 and not has_wind_rule:
            precautions.append(f"현재 실시간 풍속이 {wind_speed}m/s이므로, 보호가스 비산 방지용 방풍막(Wind Screen)을 설치하십시오.")[cite: 3]

        # 2. 기온 5℃ 이하 예열 수칙 누락 방어
        has_temp_rule = any("예열" in p for p in precautions)
        if temp <= 5.0 and not has_temp_rule:
            precautions.insert(0, f"현재 외기온도가 {temp}℃(5℃ 이하)이므로, 용접 전 모재 100℃ 이상 예열이 필수입니다.")[cite: 1, 3]

        # 3. 기본 수칙 보완
        if not precautions:
            precautions.append("용접 전용 보안경 및 차광 보호구 착용 철저")

        generated_doc["safety_precautions"] = precautions

    return {"generated_document": generated_doc}

# --------------------------------------------------------------------------
# 3. 서브그래프 빌드
# --------------------------------------------------------------------------
def route_doc_generator(state: ShipAgentState) -> Literal["weld_order_agent", "heavy_plan_agent", "general_qa_agent"]:
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

builder.add_conditional_edges(
    START,
    route_doc_generator,
    {
        "weld_order_agent": "weld_order_agent",
        "heavy_plan_agent": "heavy_plan_agent",
        "general_qa_agent": "general_qa_agent",
    }
)

builder.add_edge("weld_order_agent", "output_validator")
builder.add_edge("heavy_plan_agent", "output_validator")
builder.add_edge("general_qa_agent", "output_validator")
builder.add_edge("output_validator", END)

team3_subgraph = builder.compile()