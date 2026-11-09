"""
modules/agent/subgraphs/analysis.py
Team 1: 요구사항 분석, 사내 ERP 자원 1차 진단 & 구조화 의도 추출 서브그래프
"""

import json
from typing import Any, Dict, List, Literal, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from modules.agent.state import DocCategory, ShipAgentState, TaskType
from modules.agent.tools.erp_tools import check_hq_erp_resources

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ==============================================================================
# 1. Agentic 의도 및 파라미터 추출 스키마
# ==============================================================================
class UniversalQueryAnalysis(BaseModel):
    task_type: Literal[
        "chat",                 # 단순 일상 인사, 잡담
        "erp_resource_qa",      # 사내 자재(강판, 와이어), 창고 재고 잔여량 단순 문의
        "hr_safety_qa",         # 특정 작업자 선급 자격증 단건 유효성 문의
        "hr_worker_qa",         # 🌟 추가: 사내 작업자 전체 명단/인력 현황 실시간 DB 조회
        "scm_partner_qa",       # 외주 협력사((주)대한용재, (주)한국운송) 가용성 및 외주 실행 문의
        "knowledge_qa",         # WPS 규격, KOSHA 지침, 지시서 구성 항목 문의 등
        "doc_generation"        # 공식 '작업지시서' 작성/발행 요청
    ] = Field(description="질의의 핵심 목적을 분류")
    
    # 🌟 추가: 단순 조회 vs 실제 물리적 원장 커밋(배차, 발주, 발행) 요구 구분
    is_action_requested: bool = Field(
        default=False,
        description="실제 외주 배차 확정, 발주 진행, 구매 등 물리적 원장 실행을 요구하면 True, 단순 현황 조회의 경우 False"
    )

    context_action: Literal["KEEP_MERGE", "RESET_NEW"] = Field(
        default="KEEP_MERGE",
        description="새로운 주제의 문서 작성이면 'RESET_NEW', 이전 대화 조건을 이어받으면 'KEEP_MERGE'",
    )
    target_categories: List[DocCategory] = Field(
        default_factory=list,
        description="필요한 RAG 7대 카테고리 목록",
    )
    need_weather: bool = Field(
        default=False,
        description="실시간 기상 조회 필요 여부. 날씨 질의나 'doc_generation'인 경우 True",
    )
    block_id: Optional[str] = Field(None, description="언급된 선박 블록 번호 (예: 'B-104', 'B-202', 'B-305')")
    material: Optional[str] = Field(None, description="언급된 모재 강종 (예: 'SS400', 'AH36', 'EH36')")
    thickness: Optional[str] = Field(None, description="언급된 판 두께 (예: '10T', '16mm', '25T')")
    location: Optional[str] = Field(None, description="언급된 작업 지역/독 (예: '울산', '거제')")
    process: Optional[str] = Field(None, description="용접/작업 공정 (예: 'FCAW', 'SAW', '맞대기')")
    worker_name: Optional[str] = Field(None, description="투입 작업자 성함 (예: '김철수', '이영희', '박무자격')")
    equipment: Optional[str] = Field(None, description="투입 장비명 (예: '트랜스포터', '골리앗 크레인')")
    additional_specs: Dict[str, Any] = Field(
        default_factory=dict,
        description="사용자가 직접 언급한 기타 엔지니어링 조건",
    )
    search_queries: List[str] = Field(
        default_factory=list,
        description="ChromaDB 벡터 검색을 위한 공학 표준 검색어",
    )


# 🌟 400 에러(Invalid schema) 방지를 위해 method="function_calling" 명시
structured_analyzer = llm.with_structured_output(
    UniversalQueryAnalysis,
    method="function_calling"
)

ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 스마트 조선소 엔지니어링 & ERP 총괄 분석 에이전트(Team 1)입니다.\n"
        "이전 대화 기록과 사용자의 최신 질의를 분석하여 의도(task_type), 액션 요청 여부(is_action_requested), 검색 쿼리, 제원을 추출하세요.\n\n"
        "[핵심 분류 원칙]\n"
        "1. '선박 건조 과정 알려줘', '용접 전류 기준이 어떻게 돼?', '작업지시서에 들어가는 항목이 뭐야?'는 지식 질의이므로 무조건 'knowledge_qa'입니다.\n"
        "2. '창고에 와이어 재고 남았어?'는 'erp_resource_qa'입니다.\n"
        "3. '작업자 누구 있어?', '사내 용접사 명단 보여줘'처럼 인력 목록을 물으면 'hr_worker_qa'입니다.\n"
        "4. '김철수 반장 자격증 유효해?'처럼 특정 작업자의 자격증 만료를 물으면 'hr_safety_qa'입니다.\n"
        "5. '대한용재나 한국운송 외주 현황 알려줘', '외주 배차 1대 확정해줘'는 'scm_partner_qa'입니다.\n"
        "6. 명시적으로 '작업지시서 작성해줘/발행해줘'라고 요청한 경우에만 'doc_generation'으로 분류하세요.\n"
        "7. '외주 실행하자', '배차 확정해줘', '발주 진행해줘'처럼 실제 원장 승인을 요구하면 is_action_requested=True로 설정하십시오.",
    ),
    MessagesPlaceholder(variable_name="messages", optional=True),
    (
        "human",
        "[직전 턴 역질문 대기 여부]: {is_waiting}\n"
        "[이전 누적 작업 제원]: {accumulated_params}\n"
        "[최신 사용자 질의]: {query}",
    ),
])


# ==============================================================================
# 2. Team 1 메인 분석 및 사내 ERP 1차 진단 노드
# ==============================================================================
def analyze_and_check_erp_node(state: ShipAgentState) -> Dict[str, Any]:
    query = state.get("user_query", "").strip()
    messages = state.get("messages", [])
    existing_params = dict(state.get("extracted_params") or {})
    is_waiting = state.get("is_param_complete") == "no"

    res: UniversalQueryAnalysis = (ANALYSIS_PROMPT | structured_analyzer).invoke({
        "messages": messages[-6:] if messages else [],
        "is_waiting": "예 (제원 보충 대기 중)" if is_waiting else "아니오",
        "accumulated_params": json.dumps(existing_params, ensure_ascii=False),
        "query": query,
    })

    # 1) 컨텍스트 병합 또는 블록 교체 시 초기화
    old_block = existing_params.get("block_id")
    new_block = res.block_id

    if new_block and old_block and new_block != old_block:
        merged_params = {}
    elif res.context_action == "RESET_NEW" and not is_waiting:
        merged_params = {}
    else:
        merged_params = dict(existing_params)

    new_fields = {
        "block_id": res.block_id,
        "material": res.material,
        "thickness": res.thickness,
        "location": res.location,
        "process": res.process,
        "worker_name": res.worker_name,
        "equipment": res.equipment,
    }
    for k, v in new_fields.items():
        if v is not None and str(v).strip() != "":
            merged_params[k] = str(v).strip()
    if res.additional_specs:
        merged_params.update(res.additional_specs)

    # 2) 일상 대화는 즉시 종료 유도
    if res.task_type == "chat":
        return {
            "task_type": "chat",
            "extracted_params": merged_params,
            "is_param_complete": "yes",
            "missing_params": [],
            "search_status": "completed",
            "final_answer": "안녕하세요! 스마트 조선소 공정 관제 및 작업지시 지원 에이전트입니다. 작업지시서 발행, WPS 규격 조회, ERP 자재/자격증 확인 등 무엇을 도와드릴까요?",
        }

    # 3) 본사 ERP 1차 조회 (블록 마스터 제원 강제 동기화 + 재고/자격증 검증)
    erp_status = None
    is_outsourcing = False
    if res.task_type in ("doc_generation", "erp_resource_qa", "hr_safety_qa", "scm_partner_qa"):
        target_block = merged_params.get("block_id")

        erp_status = check_hq_erp_resources.invoke({
            "block_id": target_block,
            "material": merged_params.get("material"),
            "worker_name": merged_params.get("worker_name"),
            "equipment": merged_params.get("equipment"),
        })

        # 블록 마스터에 등록된 블록이면 ERP 공식 설계 데이터로 무조건 동기화
        bm = erp_status.get("block_master")
        if bm:
            for bk in ("material", "thickness", "joint_type", "process"):
                if bm.get(bk):
                    merged_params[bk] = bm[bk]

            if not res.location and bm.get("location"):
                merged_params["location"] = bm["location"]

            erp_status = check_hq_erp_resources.invoke({
                "block_id": target_block,
                "material": merged_params.get("material"),
                "worker_name": merged_params.get("worker_name"),
                "equipment": merged_params.get("equipment"),
            })

        is_outsourcing = bool(erp_status.get("require_outsourcing", False))

    if res.task_type == "scm_partner_qa":
        is_outsourcing = True

    # 4) 필수 제원(material, thickness) 누락 검사
    missing: List[str] = []
    is_complete = "yes"
    clarification_answer = None

    if res.task_type == "doc_generation":
        # 불필요한 location 필수 요구를 제거하고 엔지니어링 핵심 2대 제원만 검사
        required_slots = {
            "material": "모재 강종(예: SS400, AH36, EH36)",
            "thickness": "판 두께(예: 10T, 16T, 25T)",
        }
        missing = [k for k in required_slots if not merged_params.get(k)]
        if missing:
            is_complete = "no"
            missing_kor = ", ".join(required_slots[m] for m in missing)
            block_notice = (
                f" (확인된 조건: {json.dumps(merged_params, ensure_ascii=False)})"
                if merged_params else ""
            )
            clarification_answer = (
                f"작업지시서 발행을 위해 필수 제원이 추가로 필요합니다.{block_notice}\n"
                f"누락된 항목(**{missing_kor}**)이나 투입할 **블록 번호(예: B-104, B-305)**를 알려주시면 "
                f"ERP 자재 현황과 WPS 규격을 대조하여 지시서를 작성해 드리겠습니다."
            )

    # 5) 검색 쿼리 보정
    queries = list(res.search_queries)
    if not queries and res.task_type in ("knowledge_qa", "doc_generation"):
        mat = merged_params.get("material", "")
        thk = merged_params.get("thickness", "")
        proc = merged_params.get("process", "용접")
        if res.task_type == "doc_generation" and (mat or thk):
            queries = [f"{mat} {thk} {proc} WPS 작업 표준 규격".strip()]
        else:
            queries = [query]

    need_weather_flag = True if res.task_type == "doc_generation" else res.need_weather

    return {
        "task_type": res.task_type,
        "is_action_requested": res.is_action_requested,  # 🌟 액션 플래그 반환
        "target_categories": res.target_categories,
        "need_weather": need_weather_flag,
        "extracted_params": merged_params,
        "erp_resource_status": erp_status,
        "is_outsourcing_required": is_outsourcing,
        "missing_params": missing,
        "is_param_complete": is_complete,
        "search_queries": queries,
        "retry_count": 0,
        "search_status": "idle",
        "final_answer": clarification_answer,
    }


builder = StateGraph(ShipAgentState)
builder.add_node("analyze_and_check_erp", analyze_and_check_erp_node)
builder.add_edge(START, "analyze_and_check_erp")
builder.add_edge("analyze_and_check_erp", END)

team1_subgraph = builder.compile()