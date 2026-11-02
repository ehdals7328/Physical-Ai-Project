# agent_system/subgraphs/analysis.py
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from state import ShipAgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --------------------------------------------------------------------------
# 1. Pydantic 구조화 출력 스키마 정의
# --------------------------------------------------------------------------
class IntentClassification(BaseModel):
    task_type: Literal["general_qa", "weld_order", "heavy_plan"] = Field(
        ...,
        description="사용자 질의의 핵심 의도 (general_qa: 도메인 단순질의, weld_order: 용접작업지시서, heavy_plan: 중량물인양계획서)"
    )

class QueryOptimization(BaseModel):
    extracted_params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "질문에서 추출된 조선/용접 엔티티 딕셔너리. "
            "주요 키: material(강종), thickness(두께), joint_type(이음형상), welding_process(용접공정), "
            "welding_position(자세), root_gap(루트간격), location(지역/독), wind_speed(풍속), equipment(장비명) 등. "
            "질문에 언급되지 않은 항목은 절대로 키를 생성하지 마세요."
        )
    )
    search_queries: List[str] = Field(
        default_factory=list,
        description="규격서 검색에 사용할 표준 공학 검색 키워드 목록 (1~3개)"
    )

# 태스크별 작업 수행을 위한 필수 파라미터 기준
REQUIRED_PARAMS = {
    "weld_order": ["material", "thickness"],      # 용접지시서는 모재 강종과 두께 필수
    "heavy_plan": ["equipment", "wind_speed"],    # 중량물계획서는 인양장비와 풍속 필수
    "general_qa": []                              # 단순 질의응답은 제약 없음
}

# --------------------------------------------------------------------------
# 2. 세부 에이전트 노드 정의
# --------------------------------------------------------------------------
def intent_classifier_node(state: ShipAgentState) -> Dict:
    """[Agent 1: 의도 분류] 사용자 질문의 목적을 전문 도메인 태스크로 분류"""
    user_query = state.get("user_query", "")

    system_prompt = (
        "당신은 조선소 스마트 공정의 요구사항 분석 전문가입니다.\n"
        "작업자의 질의를 바탕으로 수행해야 할 업무 유형을 하나만 분류하세요:\n"
        "- 'weld_order': 강재 용접, 맞대기/필렛 용접 조건, 용접 작업지시서 작성 요청\n"
        "- 'heavy_plan': 블록 인양, 크레인 작업, 샤클/와이어로프 중량물 취급계획서 요청\n"
        "- 'general_qa': 단순 규정 질문, 용어 설명, 결함 방지 수칙 질문"
    )

    decision = llm.with_structured_output(
        IntentClassification,
        method="function_calling"
    ).invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ])

    return {"task_type": decision.task_type}


def query_optimizer_node(state: ShipAgentState) -> Dict:
    """[Agent 2: 파라미터 추출 및 쿼리 정제] 대화 맥락과 기존 조건을 누적하여 정제"""
    user_query = state.get("user_query", "")
    task_type = state.get("task_type", "general_qa")
    prev_params = state.get("extracted_params") or {}
    
    # 이전 대화 문맥 참조
    messages = state.get("messages", [])
    history_context = ""
    if messages:
        history_context = "\n".join(
            [f"{msg.type}: {msg.content}" for msg in messages[-6:]]
        )

    system_prompt = (
        "당신은 조선·해양 생산기술 및 공정설계 수석 엔지니어입니다.\n"
        "이전 대화 맥락과 현재 작업자의 질의를 종합하여 전문 엔지니어링 파라미터를 추출하고 벡터 검색 쿼리를 도출하세요.\n\n"
        "※ 엔티티 추출 규칙:\n"
        "1. 이전 대화에서 이미 언급된 조건(강종, 두께, 위치 등)은 이번 질문에서 바뀌지 않는 한 그대로 유지(누적)하세요.\n"
        "2. 질문에서 새롭게 언급되거나 수정된 조건만 덮어쓰거나 추가하세요.\n"
        "3. 전혀 언급되지 않은 속성은 None으로 채우지 말고 키를 생성하지 마세요.\n\n"
        f"[확정된 태스크 유형]: {task_type}\n"
        f"[이전까지 누적된 파라미터]: {prev_params}\n"
        f"[최근 대화 기록]:\n{history_context}"
    )

    result = llm.with_structured_output(
        QueryOptimization,
        method="function_calling"
    ).invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ])

    # 기존 누적 파라미터에 새 파라미터 덮어쓰기 병합
    merged_params = dict(prev_params)
    for k, v in result.extracted_params.items():
        if v is not None:
            merged_params[k] = v

    return {
        "extracted_params": merged_params,
        "search_queries": result.search_queries
    }


def parameter_checker_node(state: ShipAgentState) -> Dict:
    """[Agent 3: 필수 파라미터 누락 검증 및 사용자 역질문 생성]"""
    task_type = state.get("task_type", "general_qa")
    params = state.get("extracted_params") or {}
    
    required = REQUIRED_PARAMS.get(task_type, [])
    missing = [p for p in required if not params.get(p)]

    # 필수 정보가 누락된 경우 -> 친절한 한국어 안내문 생성
    if missing:
        field_names_kor = {
            "material": "모재 강종(예: SS400, AH36)",
            "thickness": "모재 두께(예: 10T, 20T)",
            "equipment": "인양 장비명(예: 타워크레인, 지브크레인)",
            "wind_speed": "작업 현장 풍속(m/s)"
        }
        missing_desc = ", ".join([field_names_kor.get(m, m) for m in missing])
        
        clarification_text = (
            f"정확한 {task_type} 작성을 위해 필수 조건이 누락되었습니다.\n"
            f"▶ 추가 확인 필요 항목: [{missing_desc}]\n"
            f"해당 정보를 알려주시면 사내 규격을 조회하여 표준 지시서를 즉시 작성해 드리겠습니다."
        )
        return {
            "final_answer": clarification_text,
            "is_param_complete": "no"
        }

    return {"is_param_complete": "yes"}


def route_param_check(state: ShipAgentState) -> str:
    """필수 파라미터 검증 필요 여부 라우팅"""
    task_type = state.get("task_type", "general_qa")
    params = state.get("extracted_params") or {}
    required = REQUIRED_PARAMS.get(task_type, [])
    
    # 필수 파라미터가 비어있으면 checker 노드로 보내서 역질문 생성
    if any(not params.get(p) for p in required):
        return "parameter_checker"
    
    return END

# --------------------------------------------------------------------------
# 3. Team 1 서브그래프 빌드
# --------------------------------------------------------------------------
builder = StateGraph(ShipAgentState)

builder.add_node("intent_classifier", intent_classifier_node)
builder.add_node("query_optimizer", query_optimizer_node)
builder.add_node("parameter_checker", parameter_checker_node)

builder.add_edge(START, "intent_classifier")
builder.add_edge("intent_classifier", "query_optimizer")

# 파라미터 충족 시 바로 통과, 부족하면 역질문 생성 후 종료
builder.add_conditional_edges(
    "query_optimizer",
    route_param_check,
    {
        "parameter_checker": "parameter_checker",
        END: END
    }
)
builder.add_edge("parameter_checker", END)

team1_subgraph = builder.compile()