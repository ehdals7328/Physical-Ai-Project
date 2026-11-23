"""
modules/agent/subgraphs/search.py
Team 2: LLM bind_tools() 기반 ReAct(Reason-Act-Observe) 자율 탐색·편성 서브그래프
- LLM이 스스로 필요한 도구를 반복 호출하고 관찰(Observation)하며 블록/작업자/자재/장비/규정/기상/비전/A2A 외주를 자율 조율
- Team 3 거버넌스 게이트키퍼에서 차단 피드백(replan_feedback)이 돌아오면 자동으로 대안 탐색 및 재편성 수행
- 내부 ToolMessage는 react_messages에 격리하여 전역 대화 히스토리 오염 방지
"""

import json
import re
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import Field

from modules.agent.state import ShipAgentState
from modules.agent.tools.domain_tools import evaluate_environmental_safety
from modules.agent.tools.erp_tools import (
    check_hq_erp_resources,
    check_yard_materials_and_equipment,
    evaluate_heavy_lifting_capacity,
    find_eligible_workers,
    find_pending_blocks,
    get_branch_context_from_db,
    list_erp_workers,
    query_partner_erp_b2b,
)
from modules.agent.tools.partner_mcp_client import negotiate_partner_via_a2a
from modules.agent.tools.retriever_tools import search_knowledge_by_categories
from modules.agent.tools.vision_tools import (
    get_vision_inspection_status,
    get_vision_overview_for_qa,
)
from modules.agent.tools.weather_mcp_client import get_realtime_weather_from_mcp

MAX_REACT_STEPS = 4


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return super().default(obj)


def _to_json(obj: Any, max_chars: int = 3500) -> str:
    s = json.dumps(obj, ensure_ascii=False, cls=DecimalEncoder)
    return s if len(s) <= max_chars else s[:max_chars] + "...(truncated)"


# ==============================================================================
# Team 2 전용 확장 State (전역 messages와 내부 react_messages 완전 격리)
# ==============================================================================
class Team2ReActState(ShipAgentState, total=False):
    react_messages: List[BaseMessage]
    react_step: int
    react_done: bool


# ==============================================================================
# 공통 환경·비전 Fail-Closed 안전 판정 헬퍼 (기존 안전 로직 100% 보존)
# ==============================================================================
def _append_vision_clearance_warning(
    safety_res: Dict[str, Any], vision_info: Dict[str, Any] | None
) -> Dict[str, Any]:
    if not vision_info:
        return safety_res

    v_status = str(vision_info.get("status", "")).upper()
    v_insp = str(vision_info.get("inspection_status", "")).upper()
    v_clearance = str(vision_info.get("safety_clearance_yn") or "").upper()
    rules = list(safety_res.get("mandatory_rules") or [])

    if v_status == "ERROR" or v_insp == "ERROR":
        err_msg = vision_info.get("error") or vision_info.get("error_message") or "비전 DB/카메라 조회 오류"
        rules.append(f"비전 관제 시스템 오류(VISION_INSPECTIONS): {err_msg} — 안전 검증 불가로 Fail-Closed 작업 중지")
        safety_res.update({
            "mandatory_rules": rules,
            "status": "STOP",
            "safety_level": "FORBIDDEN",
            "is_safe": False,
            "action_message": "비전 관제 조회 오류로 인한 작업 즉시 보류",
        })
    elif v_clearance == "N":
        interfering = ", ".join(vision_info.get("interfering_objects") or ["간섭 장애물"])
        rules.append(f"비전 안전 인터록: 작업 구역 인근 간섭({interfering}) 감지 — 장애물 제거 전 착수 금지")
        safety_res.update({
            "mandatory_rules": rules,
            "status": "STOP",
            "safety_level": "FORBIDDEN",
            "is_safe": False,
            "action_message": "비전 안전거리 미확보(간섭물 감지)로 인한 작업 즉시 보류",
        })
    elif v_status == "NO_DATA" or v_insp == "NO_DATA":
        prereqs = list(safety_res.get("prerequisites") or [])
        prereqs.append("비전 실시간 계측 이력 미존재(계측 대기): 작업 착수 전 현장 안전거리 육안 확인 필수")
        safety_res["prerequisites"] = prereqs

    return safety_res


def _run_weather_and_vision_check(
    location: Optional[str],
    branch_code: Optional[str],
    block_id: Optional[str],
    work_id: Optional[str],
    camera_code: Optional[str],
    doc_type: Optional[str],
    process: Optional[str],
    equipment: Optional[str],
    need_vision: bool = True,
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    vision_info = None
    if need_vision:
        try:
            vision_info = get_vision_inspection_status.invoke({
                "work_id": work_id or block_id,
                "camera_code": camera_code,
            })
        except Exception as e:
            vision_info = {"status": "ERROR", "error_message": str(e)}

    resolved_loc = location
    loc_upper = str(resolved_loc or "").strip().upper()
    if not loc_upper or any(tok in loc_upper for tok in ("ZONE", "DOCK", "BAY", "BR-", "YARD")):
        lookup_br = loc_upper if loc_upper.startswith("BR-") else branch_code
        b_ctx = get_branch_context_from_db(lookup_br)
        resolved_loc = b_ctx.get("region_name") or ("거제" if "GEOJE" in str(lookup_br or "").upper() else "울산")

    try:
        weather_info = dict(get_realtime_weather_from_mcp(resolved_loc) or {})
    except Exception as e:
        weather_info = {"status": "ERROR", "error": str(e)}

    status = weather_info.get("status")
    if status != "SUCCESS":
        err_msg = weather_info.get("error") or "기상 데이터 결측"
        fail_closed = _append_vision_clearance_warning(
            {
                "status": "STOP",
                "safety_level": "FORBIDDEN",
                "is_safe": False,
                "mandatory_rules": [f"기상 데이터 수집 실패({status or 'ERROR'}: {err_msg}) — Fail-Closed 작업 중지"],
                "prerequisites": [],
                "action_message": "기상 데이터 확인 불가로 인한 작업 즉시 보류",
            },
            vision_info,
        )
        return {
            "location": resolved_loc,
            "status": status or "ERROR",
            "error": err_msg,
            "safety_evaluation": fail_closed,
        }, vision_info

    temp = float(weather_info["temperature"]) if weather_info.get("temperature") is not None else None
    wind = float(weather_info["wind_speed"]) if weather_info.get("wind_speed") is not None else None
    rain = float(weather_info["precipitation"]) if weather_info.get("precipitation") is not None else None

    try:
        if doc_type == "HEAVY_LIFTING_PLAN":
            if wind is not None and wind >= 10.0:
                safety_res = {
                    "status": "STOP",
                    "safety_level": "FORBIDDEN",
                    "is_safe": False,
                    "mandatory_rules": [f"순간풍속 {wind}m/s (중량물 양중 중지 기준 10.0m/s 초과) — 인양 즉시 중지"],
                    "prerequisites": [],
                    "action_message": "강풍으로 인한 중량물 인양 즉시 중단",
                }
            elif (wind is not None and wind >= 7.0) or (rain is not None and rain > 0):
                rules = []
                if wind is not None and wind >= 7.0:
                    rules.append(f"풍속 {wind}m/s (7~10m/s 주의): 유도 로프(Tag Line) 2개소 체결 및 저속 인양 필수")
                if rain is not None and rain > 0:
                    rules.append(f"강수({rain}mm/h) 감지: 줄걸이 슬립 방지 및 지반 사전 점검 필수")
                safety_res = {
                    "status": "CONDITIONAL_PASS",
                    "safety_level": "CAUTION",
                    "is_safe": True,
                    "mandatory_rules": rules,
                    "prerequisites": ["작업지휘자 현장 상주 및 지상 30cm 시험 인양 실시"],
                    "action_message": "중량물 양중 기상 주의 조건부 승인",
                }
            else:
                safety_res = {
                    "status": "PASS",
                    "safety_level": "SAFE",
                    "is_safe": True,
                    "mandatory_rules": [f"실측 풍속 {wind}m/s (중량물 양중 허용 기준 10.0m/s 이내 정상)"],
                    "prerequisites": ["전담 신호수 배치 및 인양 반경 하부 출입 통제"],
                    "action_message": "중량물 양중 기상 안전 통과",
                }
        else:
            safety_res = evaluate_environmental_safety.invoke({
                "current_temp": temp,
                "current_wind": wind,
                "rainfall": rain,
                "thresholds": None,
                "task_context": str(process or equipment or "야외 용접"),
                "is_outdoor": True,
            })

        safety_res = _append_vision_clearance_warning(safety_res, vision_info)
        weather_info["safety_evaluation"] = safety_res
        return weather_info, vision_info
    except Exception as e:
        fail_closed = _append_vision_clearance_warning(
            {
                "status": "STOP",
                "safety_level": "FORBIDDEN",
                "is_safe": False,
                "mandatory_rules": [f"환경 안전 평가 오류({e}) — 안전 보류"],
                "prerequisites": [],
                "action_message": "안전 평가 오류로 인한 작업 즉시 보류",
            },
            vision_info,
        )
        return {
            "location": resolved_loc,
            "status": "EVALUATION_ERROR",
            "error": str(e),
            "safety_evaluation": fail_closed,
        }, vision_info


# ==============================================================================
# ReAct LLM에 바인딩할 도구 스키마 정의 (실제 실행 + State 동기화는 executor가 담당)
# ==============================================================================
@tool
def tool_find_pending_blocks(
    branch_code: Optional[str] = Field(
        default=None,
        description="조회할 특정 야드 코드(예: 'BR-ULSAN-01', 'BR-GEOJE-01'). 전체 야드를 조회할 때는 'ALL'이나 한글을 넣지 말고 반드시 null(None)을 전달하세요.",
    ),
    process: Optional[str] = Field(
        default=None,
        description="특정 용접 공정 코드(FCAW, SAW, GMAW, SMAW, GTAW 중 하나). 전체 공정을 조회할 때는 반드시 null(None)을 전달하세요.",
    ),
) -> str:
    """대상 야드와 공정에서 작업 편성 가능한 블록 후보 목록을 탐색합니다."""
    return ""


@tool
def tool_find_eligible_workers(
    branch_code: Optional[str] = Field(
        default=None,
        description="조회할 특정 야드 코드(예: 'BR-ULSAN-01', 'BR-GEOJE-01'). 전체 야드를 조회할 때는 'ALL'이나 한글을 넣지 말고 반드시 null(None)을 전달하세요.",
    ),
    process: Optional[str] = Field(
        default=None,
        description="특정 용접 공정 코드(FCAW, SAW, GMAW, SMAW, GTAW 중 하나). 전체 공정을 조회할 때는 반드시 null(None)을 전달하세요.",
    ),
) -> str:
    """대상 야드 권역·용접 공정과 일치하고 자격이 유효하며 현재 대기 중(IDLE)인 적격 작업자 후보를 탐색합니다."""
    return ""


@tool
def tool_check_materials_and_equipment(
    branch_code: Optional[str] = None,
    process: Optional[str] = None,
    weight_ton: Optional[float] = None,
    requested_equipment: Optional[str] = None,
) -> str:
    """야드의 강판/와이어 가용 재고와 대기 중인 크레인/트랜스포터 및 중량물 정격하중 안전율을 점검합니다."""
    return ""


@tool
def tool_sync_erp_plan(
    block_id: Optional[str] = None,
    worker_name: Optional[str] = None,
    equipment: Optional[str] = None,
    branch_code: Optional[str] = None,
    process: Optional[str] = None,
    preferred_execution_mode: Optional[Literal["AUTO", "IN_HOUSE", "OUTSOURCED_B2B"]] = None,
    reasoning: Optional[str] = None,
) -> str:
    """선정한 블록·작업자·장비·수행모드를 본사 ERP 원장과 최종 정밀 대조하고 작업 계획 상태(State)에 확정 반영합니다."""
    return ""


@tool
def tool_search_knowledge_rag(
    query: str,
    categories: Optional[List[str]] = None,
) -> str:
    """사내 WPS 용접 절차서, 와이어/플럭스 보관 지침, NDT 비파괴검사, 중량물 안전 규정을 ChromaDB에서 검색합니다. 결과가 부족하면 키워드를 바꿔 재호출하세요."""
    return ""


@tool
def tool_check_weather_and_vision(
    location: Optional[str] = None,
    block_id: Optional[str] = None,
    branch_code: Optional[str] = None,
) -> str:
    """현장 실시간 기상(FastMCP)과 YOLOv11 비전 안전거리·결함 상태를 조회하고 환경 안전 판정을 수행합니다."""
    return ""


@tool
def tool_negotiate_b2b_partner(
    outsourcing_domain: Literal["ALL_PARTNERS", "WELDING_MATERIAL", "HEAVY_TRANSPORT", "OUTSOURCED_WORKER"],
    branch_code: Optional[str] = None,
    material: Optional[str] = None,
    process: Optional[str] = None,
    worker_name: Optional[str] = None,
    requested_qty: Optional[float] = None,
    weight_ton: Optional[float] = None,
    urgency: Literal["NORMAL", "URGENT"] = "NORMAL",
) -> str:
    """외부 협력사 FastMCP 카탈로그를 조회하고, 외주 조달/인력/배차가 필요할 때 협력사 A2A 에이전트와 납기·단가·대체자원을 자율 협의합니다."""
    return ""


@tool
def tool_get_vision_overview(
    block_id: Optional[str] = None,
    camera_code: Optional[str] = None,
) -> str:
    """현장 비전(YOLOv11) 카메라별 탐지 현황, 용접 대기 구역, 간섭물 및 로봇 관제 상태를 조회합니다."""
    return ""


@tool
def tool_list_all_workers(branch_code: Optional[str] = None) -> str:
    """사내외 전체 작업자 인사 원장(ERP_WORKER_CERT)을 조회합니다."""
    return ""


REACT_TOOLS = [
    tool_find_pending_blocks,
    tool_find_eligible_workers,
    tool_check_materials_and_equipment,
    tool_sync_erp_plan,
    tool_search_knowledge_rag,
    tool_check_weather_and_vision,
    tool_negotiate_b2b_partner,
    tool_get_vision_overview,
    tool_list_all_workers,
]

react_llm = ChatOpenAI(model="gpt-5-mini", reasoning_effort="low").bind_tools(REACT_TOOLS)

REACT_SYSTEM_PROMPT = """당신은 스마트 조선소 자율 공정 편성 및 데이터 감리 ReAct 에이전트(Team 2)입니다.
주어진 [사용자 목표], [작업 유형], [현재 확보된 제원], [거버넌스 재편성 피드백]을 바탕으로 도구를 **최소 턴(병렬 호출 적극 활용)**으로 호출하여 문제를 완전히 해결하세요.

[속도 및 정확도 핵심 규칙]
1. **작업자 식별자 규칙**: `tool_sync_erp_plan`의 `worker_name` 인자에는 사번(`EMP-1002`)이 아니라 **반드시 한글 성명(`최선임`, `강숙련` 등)**을 전달하세요.
2. **병렬(Parallel) 일괄 호출 필수**: 대상 블록과 작업자(또는 장비)가 정해지면, 다음 턴에 **`tool_sync_erp_plan`, `tool_check_weather_and_vision`, `tool_search_knowledge_rag` 3개를 한 번에 동시에 호출**하세요. 절대 1개씩 순차 호출하지 마세요.
3. **사내/외주 인력 즉시 분기 규칙**:
   - `tool_find_eligible_workers`에서 `eligible_count >= 1`이면 바로 그 인원을 선택하고 `tool_negotiate_b2b_partner(outsourcing_domain='OUTSOURCED_WORKER')`를 호출하지 마세요.
   - 반대로 `eligible_count == 0`이면 사내/상주 가용 인력이 없다는 뜻이므로 `tool_list_all_workers`를 다시 뒤지지 말고 **즉시 `tool_negotiate_b2b_partner(outsourcing_domain='OUTSOURCED_WORKER')`를 호출**해 외주 용접사를 확보하세요.

[작업 유형별 자율 행동 원칙]
1. **용접 작업지시서 생성 (`doc_type == 'WELDING_WORK_ORDER'`)**:
   - **복수 블록('둘 다', '모두') 요청 시**: 현재 결재 시스템은 1회 1건씩 초안을 발행하므로, 가장 먼저 착수할 1순위 블록 1건을 먼저 선정해 작업지시서 초안을 완성하고, 요약 문장에 "나머지 블록(예: BLK-E102)도 바로 이어서 발행할까요?"라고 안내하세요.
   - **1단계 (자원 탐색 — 필요 시 병렬 호출)**: `block_id`나 `worker_name`이 비어 있으면 `tool_find_pending_blocks`와 `tool_find_eligible_workers`를 즉시 호출해 후보를 확보하세요.
   - **2단계 (동기화·안전·규격 일괄 검증)**: 후보가 정해지면 `tool_sync_erp_plan` + `tool_check_weather_and_vision` + `tool_search_knowledge_rag`를 **한 턴에 병렬 호출**하세요.
   - **3단계 (예외 시에만 외주 A2A 연계)**: 와이어 재고 부족(`require_outsourcing=True`) 시 `WELDING_MATERIAL`, 사내 적격 인력 0명 시 `OUTSOURCED_WORKER`로 `tool_negotiate_b2b_partner`를 1회 호출하세요.
   - **자율 편성 모드(`auto_assign_mode=True`)**에서 선정한 블록의 기상/비전이 `STOP`이면 다른 대기 블록 후보로 전환하세요.
1-1. **중량물 취급 작업계획서 생성 (`doc_type == 'HEAVY_LIFTING_PLAN'`)**:
   - **용접사 탐색 금지**: 중량물 인양/운반 계획서에는 용접사가 필요 없으므로 `tool_find_eligible_workers`와 `tool_negotiate_b2b_partner(outsourcing_domain='OUTSOURCED_WORKER')`를 **절대 호출하지 마세요**.
   - **1단계 (동기화·기상·규격 병렬 호출)**: `block_id`가 있으면 즉시 `tool_sync_erp_plan` + `tool_check_weather_and_vision`(location은 반드시 `'울산'` 또는 `'거제'` 같은 도시명) + `tool_search_knowledge_rag`를 동시에 호출하세요.
   - **2단계 (사내 장비 정격하중 초과 시 외주 중량물 A2A 배차)**: `tool_sync_erp_plan` 결과에서 사내 장비가 `OVERLOAD`(`need_outsourcing=True`)이고 사용자가 과하중 장비를 강제 지정(`user_explicit_equipment=True`)한 것이 아니라면, 즉시 `tool_negotiate_b2b_partner(outsourcing_domain='HEAVY_TRANSPORT', weight_ton=블록중량, branch_code=야드코드)`를 호출해 협력사 대형 트랜스포터/크레인 배차를 자율 협의하세요.
2. **거버넌스 게이트키퍼 차단 복구 (`replan_feedback` 존재 시)**:
   - 피드백에 명시된 차단 사유를 해소할 대체 작업자·외주·장비·블록을 찾아 `tool_sync_erp_plan`으로 즉시 교체하세요.
3. **비전 관제 질의 (`vision_inspection_qa`)**: `tool_get_vision_overview` 1회 호출 후 즉시 종료하세요.
4. **협력사 외주 문의 (`scm_partner_qa`)**: `tool_negotiate_b2b_partner` 호출 후 종료하세요.
5. **기술 규격 / 기상 질의 (`knowledge_qa`)**: 규격 질문은 `tool_search_knowledge_rag`, 기상 확인은 `tool_check_weather_and_vision`을(둘 다 필요 시 병렬) 호출하세요.

[종료 조건]
- 필요한 근거와 확정 편성안이 확보되면 즉시 도구 호출을 멈추고, **무엇을 탐색해 어떤 기준으로 편성했는지 1~2줄 한국어 요약**만 출력하고 종료하세요.
"""


# ==============================================================================
# Node 1: ReAct 루프 초기화 (최초 진입 & Team 3 복구 진입 공통)
# ==============================================================================
import re


def _extract_pure_user_query(raw_query: str) -> str:
    """웹 라우트에서 덧붙인 화면 컨텍스트([현재 화면...], [화면 요약...])를 제거하고 순수 사용자 발화만 추출"""
    q = str(raw_query or "").strip()
    for marker in ("\n[현재 화면", "\n[화면 ", "\n[선택된 ", "\n---"):
        if marker in q:
            q = q.split(marker, 1)[0].strip()
    if q.startswith("[") and "\n" in q:
        lines = [ln.strip() for ln in q.splitlines() if ln.strip() and not ln.strip().startswith("[")]
        if lines:
            q = lines[-1]
    return q


def _collect_session_drafted_from_messages(
    messages: List[BaseMessage],
    target_doc_type: Optional[str] = None,
) -> tuple[List[str], List[str]]:
    """현재 세션 대화 기록에서 동일 문서 유형(용접 vs 중량물)으로 이미 발행된 블록 ID와 배정 작업자를 분리 수집"""
    blocks: List[str] = []
    workers: List[str] = []
    is_lifting = (target_doc_type == "HEAVY_LIFTING_PLAN")
    header_keyword = "중량물 취급 작업계획서 발행 브리핑" if is_lifting else "공식 작업지시서 발행 브리핑"

    for m in (messages or []):
        if getattr(m, "type", "") != "ai":
            continue
        txt = str(getattr(m, "content", ""))
        if header_keyword not in txt:
            continue
        for bm in re.findall(r"대상 블록\s*/\s*(?:제원|총 인양하중)\*\*:\s*`([^`]+)`", txt):
            b_clean = bm.strip().upper()
            if b_clean and b_clean not in blocks:
                blocks.append(b_clean)
        if not is_lifting:
            for wm in re.findall(r"투입 인력\s*/\s*자재\*\*:\s*([^\s(`]+)", txt):
                w_clean = wm.strip()
                if w_clean and w_clean != "-" and w_clean not in workers:
                    workers.append(w_clean)
    return blocks, workers


def react_init_node(state: Team2ReActState) -> Dict[str, Any]:
    task_type = state.get("task_type", "knowledge_qa")
    params = dict(state.get("extracted_params") or {})
    replan_feedback = state.get("replan_feedback")
    replan_count = state.get("replan_count", 0)
    auto_assign = bool(state.get("auto_assign_mode", False))

    target_doc_type = params.get("doc_type")
    msg_blks, msg_wrks = _collect_session_drafted_from_messages(
        state.get("messages") or [],
        target_doc_type=target_doc_type,
    )
    # replan 중 추가된 제외 목록은 유지하되, 새 턴(replan==0)에서는 현재 문서 유형(용접 vs 중량물) 기준 이력만 적용
    base_excl_blks = list(state.get("excluded_blocks") or []) if replan_count > 0 else []
    base_excl_wrks = list(state.get("excluded_workers") or []) if replan_count > 0 else []
    pure_user_q = _extract_pure_user_query(state.get("user_query", ""))
    pure_user_q_upper = pure_user_q.upper()

    excluded_blocks = list(dict.fromkeys([str(b).strip().upper() for b in base_excl_blks if b] + msg_blks))
    excluded_workers = list(dict.fromkeys([str(w).strip() for w in base_excl_wrks if w] + msg_wrks))
    # 새 턴에서 사용자가 명시적으로 블록을 지정한 경우(예: 장비 변경 테스트)에는 제외 목록에서 해제
    if replan_count == 0:
        excluded_blocks = [b for b in excluded_blocks if b not in pure_user_q_upper]

    if replan_feedback:
        mission = (
            f"🔄 **[자율 복구 재편성 미션 #{replan_count}]**\n"
            f"- 직전 계획이 Team 3 거버넌스 게이트키퍼에서 차단되었습니다.\n"
            f"- **차단 피드백**: {replan_feedback}\n"
            f"- 제외할 블록: {excluded_blocks} / 제외할 작업자: {excluded_workers}\n"
            f"- 현재 제원: {_to_json(params, 1200)}\n"
            f"- 자율 편성 모드 여부: {auto_assign}\n"
            f"도구를 호출하여 차단 사유를 해소할 적격 대안(대체 작업자, 외주 연계, 대체 장비 또는 대체 블록)을 찾고 "
            f"`tool_sync_erp_plan`으로 최종 반영하세요."
        )
    else:
        # 사용자가 순수 발화에서 명시 재요청한 게 아니라면 이미 발행된 블록·작업자는 params에서 확실히 제거
        cur_blk = str(params.get("block_id") or "").strip().upper()
        if cur_blk and cur_blk in excluded_blocks and cur_blk not in pure_user_q_upper:
            for k in (
                "block_id", "block_name", "ship_code", "material",
                "thickness", "joint_type", "process", "weight_ton",
            ):
                params.pop(k, None)

        cur_wrk = str(params.get("worker_name") or "").strip()
        if cur_wrk and cur_wrk in excluded_workers and cur_wrk not in pure_user_q:
            params.pop("worker_name", None)

        recent_ai_msgs = [
            str(getattr(m, "content", ""))[:500]
            for m in (state.get("messages") or [])
            if getattr(m, "type", "") == "ai"
        ][-2:]
        last_ai_context = "\n---\n".join(recent_ai_msgs)

        mission = (
            f"🎯 **[사용자 순수 요청]**: {pure_user_q}\n"
            f"- 최근 대화 맥락:\n{last_ai_context or '없음'}\n"
            f"- 작업 유형(`task_type`): {task_type}\n"
            f"- 문서 유형(`doc_type`): {params.get('doc_type') or 'N/A'}\n"
            f"- 자율 편성 모드(`auto_assign_mode`): {auto_assign}\n"
            f"- 선호 수행 모드: {state.get('preferred_execution_mode', 'AUTO')}\n"
            f"- 외주 도메인 힌트: {state.get('outsourcing_domain', 'NOT_APPLICABLE')}\n"
            f"- 기상 조회 필요 여부: {state.get('need_weather', False)}\n"
            f"- 기본 야드 코드: {params.get('branch_code') or state.get('branch_code')}\n"
            f"- 현재 추출된 제원: {_to_json(params, 1200)}\n"
            f"- [절대 준수] 이미 초안 발행 완료된 블록(재선택·재발행 금지): {excluded_blocks}\n"
            f"- [절대 준수] 이미 초안 배정된 작업자(중복 배정 금지): {excluded_workers}\n"
            + (
                "- 위 제외 목록에 해당하는 블록/작업자는 절대 `tool_sync_erp_plan`에 넣지 말고, "
                "반드시 `tool_find_pending_blocks`와 `tool_find_eligible_workers`를 먼저 호출해 남은 다음 블록과 다른 적격 작업자를 선정하세요."
                if (excluded_blocks or excluded_workers)
                else ""
            )
        )

    print("\n" + "-" * 60)
    print(
        f"🤖 [Team 2 ReAct Init] task={task_type}, auto_assign={auto_assign}, "
        f"replan=#{replan_count}, excluded_blocks={excluded_blocks}, excluded_workers={excluded_workers}"
    )
    if replan_feedback:
        print(f"   ↩️ 복구 피드백: {replan_feedback}")

    return {
        "react_messages": [
            SystemMessage(content=REACT_SYSTEM_PROMPT),
            HumanMessage(content=mission),
        ],
        "extracted_params": params,
        "excluded_blocks": excluded_blocks,
        "excluded_workers": excluded_workers,
        "drafted_blocks_history": excluded_blocks,
        "drafted_workers_history": excluded_workers,
        "react_step": 0,
        "react_done": False,
        "replan_needed": False,
    }

# ==============================================================================
# Node 2: ReAct 추론 및 도구 선택 (LLM bind_tools)
# ==============================================================================
def react_agent_node(state: Team2ReActState) -> Dict[str, Any]:
    step = state.get("react_step", 0)
    r_msgs = list(state.get("react_messages") or [])

    if step >= MAX_REACT_STEPS:
        print(f"   ⏹️ [ReAct] 최대 추론 스텝({MAX_REACT_STEPS}) 도달 ➔ 종료")
        return {"react_done": True}

    try:
        ai_msg: AIMessage = react_llm.invoke(r_msgs)
    except Exception as e:
        print(f"⚠️ [Team 2 ReAct LLM 오류]: {e}")
        return {"react_done": True}

    r_msgs.append(ai_msg)
    tool_calls = getattr(ai_msg, "tool_calls", None) or []

    if tool_calls:
        names = [tc.get("name") for tc in tool_calls]
        print(f"   🧠 [ReAct Step #{step + 1}] LLM 자율 도구 선택: {names}")
        return {
            "react_messages": r_msgs,
            "react_step": step + 1,
            "react_done": False,
        }

    summary = (ai_msg.content or "").strip()
    print(f"   ✅ [ReAct Step #{step + 1}] 자율 탐색 완료: {summary[:120]}")
    return {
        "react_messages": r_msgs,
        "react_step": step + 1,
        "react_done": True,
        "agent_reasoning_summary": summary or state.get("agent_reasoning_summary"),
    }


def route_after_react_agent(
    state: Team2ReActState,
) -> Literal["react_tool_executor", "react_finalize"]:
    if state.get("react_done"):
        return "react_finalize"
    r_msgs = state.get("react_messages") or []
    if r_msgs and getattr(r_msgs[-1], "tool_calls", None):
        return "react_tool_executor"
    return "react_finalize"


# ==============================================================================
# Node 3: 도구 실행 + 관찰(Observation) 반환 + 표준 State 실시간 동기화
# ==============================================================================
def react_tool_executor_node(state: Team2ReActState) -> Dict[str, Any]:
    r_msgs = list(state.get("react_messages") or [])
    last_ai = r_msgs[-1]
    tool_calls = getattr(last_ai, "tool_calls", None) or []

    params = dict(state.get("extracted_params") or {})
    erp_status = dict(state.get("erp_resource_status") or {})
    retrieved_docs = list(state.get("retrieved_docs") or [])
    seen_doc_ids = {
        (d.get("metadata") or {}).get("doc_id") for d in retrieved_docs if d.get("metadata")
    }
    weather_data = state.get("weather_data")
    vision_data = state.get("vision_data")
    partner_plan = state.get("partner_erp_plan")
    is_outsourcing = bool(state.get("is_outsourcing_required", False))
    pref_mode = state.get("preferred_execution_mode", "AUTO")
    outsourcing_domain = state.get("outsourcing_domain") or "NOT_APPLICABLE"
    doc_grade = state.get("doc_grade")
    search_status = state.get("search_status", "idle")
    reasoning_summary = state.get("agent_reasoning_summary")
    tool_trace = list(state.get("tool_trace") or [])
    excluded_blocks = list(state.get("excluded_blocks") or [])
    excluded_workers = list(state.get("excluded_workers") or [])
    can_access_internal = state.get("can_access_internal_rag", "Y") != "N"

    for tc in tool_calls:
        t_name = tc.get("name", "")
        t_args = dict(tc.get("args") or {})
        t_id = tc.get("id", f"call_{t_name}")
        obs: Any = {}

        try:
            if t_name == "tool_find_pending_blocks":
                obs = find_pending_blocks.invoke({
                    "branch_code": t_args.get("branch_code") or params.get("branch_code"),
                    "process": t_args.get("process") or params.get("process"),
                    "exclude_blocks": excluded_blocks,
                })
                if obs.get("blocks"):
                    erp_status["available_blocks"] = obs["blocks"]

            elif t_name == "tool_find_eligible_workers":
                obs = find_eligible_workers.invoke({
                    "branch_code": t_args.get("branch_code") or params.get("branch_code"),
                    "process": t_args.get("process") or params.get("process"),
                    "exclude_workers": excluded_workers,
                })

            elif t_name == "tool_check_materials_and_equipment":
                obs = check_yard_materials_and_equipment.invoke({
                    "branch_code": t_args.get("branch_code") or params.get("branch_code"),
                    "process": t_args.get("process") or params.get("process"),
                    "weight_ton": t_args.get("weight_ton") or params.get("weight_ton"),
                    "requested_equipment": t_args.get("requested_equipment") or params.get("equipment"),
                })

            elif t_name == "tool_sync_erp_plan":
                req_blk = str(t_args.get("block_id") or params.get("block_id") or "").strip().upper()
                req_wrk = str(t_args.get("worker_name") or params.get("worker_name") or "").strip()
                pure_q = _extract_pure_user_query(state.get("user_query", ""))

                if req_blk and req_blk in excluded_blocks and req_blk not in pure_q.upper():
                    obs = {
                        "status": "REJECTED_DUPLICATE",
                        "error_message": (
                            f"'{req_blk}' 블록은 이미 이번 세션에서 초안 발행이 완료되었습니다. "
                            "반드시 `tool_find_pending_blocks`를 호출해 아직 발행되지 않은 다른 블록(타 야드 포함 시 branch_code=null)을 선택하세요."
                        ),
                    }
                    tool_trace.append({"step": state.get("react_step", 0), "tool": t_name, "args": t_args, "status": "REJECTED_DUPLICATE"})
                    r_msgs.append(ToolMessage(tool_call_id=t_id, name=t_name, content=_to_json(obs)))
                    continue

                if req_wrk and req_wrk in excluded_workers and req_wrk not in pure_q:
                    obs = {
                        "status": "REJECTED_DUPLICATE",
                        "error_message": (
                            f"'{req_wrk}' 작업자는 이미 다른 초안에 배정되어 중복 투입할 수 없습니다. "
                            "반드시 `tool_find_eligible_workers`를 호출해 다른 대기 작업자를 선택하세요."
                        ),
                    }
                    tool_trace.append({"step": state.get("react_step", 0), "tool": t_name, "args": t_args, "status": "REJECTED_DUPLICATE"})
                    r_msgs.append(ToolMessage(tool_call_id=t_id, name=t_name, content=_to_json(obs)))
                    continue

                if req_blk:
                    params["block_id"] = req_blk
                if req_wrk and params.get("doc_type") != "HEAVY_LIFTING_PLAN":
                    params["worker_name"] = req_wrk
                if t_args.get("equipment") and not params.get("user_explicit_equipment"):
                    params["equipment"] = str(t_args["equipment"]).strip()
                if t_args.get("branch_code"):
                    params["branch_code"] = str(t_args["branch_code"]).strip().upper()
                valid_procs = {"FCAW", "SAW", "GMAW", "SMAW", "GTAW", "TIG", "MIG"}
                if t_args.get("process"):
                    raw_p = str(t_args["process"]).strip().upper()
                    if raw_p in valid_procs:
                        params["process"] = raw_p
                if str(params.get("process") or "").upper() not in valid_procs:
                    params.pop("process", None)
                if t_args.get("preferred_execution_mode"):
                    pref_mode = t_args["preferred_execution_mode"]
                if t_args.get("reasoning"):
                    reasoning_summary = t_args["reasoning"]

                erp_status = check_hq_erp_resources.invoke({
                    "block_id": params.get("block_id"),
                    "material": params.get("material"),
                    "worker_name": params.get("worker_name"),
                    "equipment": params.get("equipment"),
                    "branch_code": params.get("branch_code"),
                    "process": params.get("process"),
                })

                # [보강 1] LLM이 사번(EMP-1002)으로 넘겼더라도 DB에서 조회된 실제 한글 성명으로 정규화
                wq = erp_status.get("worker_qualification") or {}
                if wq.get("worker_name"):
                    params["worker_name"] = wq["worker_name"]

                bm = erp_status.get("block_master")
                if bm:
                    for bk in (
                        "ship_code", "block_name", "material", "thickness",
                        "joint_type", "process", "branch_code", "weight_ton",
                    ):
                        if bm.get(bk) is not None:
                            params[bk] = bm[bk]
                    if bm.get("work_zone"):
                        params["work_zone"] = bm["work_zone"]

                b_info = erp_status.get("branch_info") or {}
                if b_info.get("region_name"):
                    params["location"] = b_info["region_name"]
                if not params.get("work_zone") and b_info.get("default_zone"):
                    params["work_zone"] = b_info["default_zone"]

                if params.get("doc_type") == "HEAVY_LIFTING_PLAN" and params.get("weight_ton"):
                    lifting_eval = evaluate_heavy_lifting_capacity(
                        block_weight_ton=params.get("weight_ton"),
                        vehicles_equipment=erp_status.get("vehicles_equipment") or [],
                        target_branch=params.get("branch_code"),
                        requested_equipment=params.get("equipment"),
                    )
                    erp_status["lifting_capacity_eval"] = lifting_eval
                    if lifting_eval.get("selected_equipment"):
                        erp_status["assigned_vehicle_code"] = lifting_eval["selected_equipment"].get("asset_code")
                    if lifting_eval.get("need_outsourcing"):
                        erp_status["require_outsourcing"] = True
                        erp_status["outsourcing_reason"] = f"HEAVY_TRANSPORT: {lifting_eval.get('reason')}"
                        outsourcing_domain = "HEAVY_TRANSPORT"

                if pref_mode == "OUTSOURCED_B2B" or erp_status.get("require_outsourcing"):
                    is_outsourcing = True
                elif pref_mode == "IN_HOUSE":
                    is_outsourcing = False

                obs = {
                    "synced_params": params,
                    "block_master": bm,
                    "worker_qualification": erp_status.get("worker_qualification"),
                    "worker_qualified": erp_status.get("worker_qualified"),
                    "stock_available": erp_status.get("stock_available"),
                    "selected_wire_code": erp_status.get("selected_wire_code"),
                    "assigned_vehicle_code": erp_status.get("assigned_vehicle_code"),
                    "lifting_capacity_eval": erp_status.get("lifting_capacity_eval"),
                    "require_outsourcing": erp_status.get("require_outsourcing"),
                    "outsourcing_reason": erp_status.get("outsourcing_reason"),
                }

            elif t_name == "tool_search_knowledge_rag":
                q = str(t_args.get("query") or state.get("user_query") or "").strip()
                cats = t_args.get("categories") or None
                if not can_access_internal and cats:
                    cats = [c for c in cats if c != "internal_docs"]
                raw_items = search_knowledge_by_categories(query=q, categories=cats, top_k=4)
                added = []
                for item in raw_items:
                    col_type = str(item.get("collection_type") or item.get("access_level") or "PUBLIC").upper()
                    if not can_access_internal and col_type in ("INTERNAL", "CONFIDENTIAL"):
                        continue
                    cid = item.get("chunk_id") or str(hash(item.get("chunk_text", "")))
                    if cid not in seen_doc_ids:
                        seen_doc_ids.add(cid)
                        doc_obj = {
                            "category": item.get("category", "규격"),
                            "content": item.get("chunk_text", "") or str(item),
                            "metadata": {
                                "document_name": item.get("document_name"),
                                "document_code": item.get("document_code"),
                                "collection_type": col_type,
                                "revision": item.get("revision"),
                                "page": item.get("page_start"),
                                "similarity": item.get("similarity_score"),
                                "source": item.get("source"),
                                "doc_id": cid,
                            },
                        }
                        retrieved_docs.append(doc_obj)
                        added.append({
                            "doc": doc_obj["metadata"].get("document_name"),
                            "page": doc_obj["metadata"].get("page"),
                            "snippet": doc_obj["content"][:280],
                        })
                search_status = "completed" if retrieved_docs else "empty"
                doc_grade = "yes" if retrieved_docs else "no"
                obs = {
                    "query": q,
                    "newly_found": len(added),
                    "total_docs": len(retrieved_docs),
                    "previews": added,
                    "hint": "검색 결과가 없으면 다른 공학 키워드로 재검색하세요." if not added else "충분한 규격 문서를 확보했습니다.",
                }

            elif t_name == "tool_check_weather_and_vision":
                target_blk = t_args.get("block_id") or params.get("block_id")
                target_br = t_args.get("branch_code") or params.get("branch_code") or state.get("branch_code")
                target_loc = t_args.get("location") or params.get("location")
                need_vis = state.get("task_type") == "doc_generation"
                weather_data, vision_data = _run_weather_and_vision_check(
                    location=target_loc,
                    branch_code=target_br,
                    block_id=target_blk,
                    work_id=state.get("work_id"),
                    camera_code=params.get("camera_code"),
                    doc_type=params.get("doc_type"),
                    process=params.get("process"),
                    equipment=params.get("equipment"),
                    need_vision=need_vis,
                )
                obs = {
                    "weather": weather_data,
                    "vision_summary": {
                        "status": (vision_data or {}).get("status"),
                        "welding_state": (vision_data or {}).get("welding_state"),
                        "inspection_status": (vision_data or {}).get("inspection_status"),
                        "safety_clearance_yn": (vision_data or {}).get("safety_clearance_yn"),
                    } if vision_data else None,
                }

            elif t_name == "tool_negotiate_b2b_partner":
                dom = t_args.get("outsourcing_domain") or outsourcing_domain or "WELDING_MATERIAL"
                if dom == "NOT_APPLICABLE":
                    dom = "WELDING_MATERIAL"
                outsourcing_domain = dom
                is_outsourcing = True

                p_plan = query_partner_erp_b2b.invoke({
                    "outsourcing_domain": dom,
                    "material": t_args.get("material") or params.get("material") or "",
                    "process": t_args.get("process") or params.get("process") or "",
                    "worker_name": t_args.get("worker_name") or params.get("worker_name") or "",
                })
                if dom in ("WELDING_MATERIAL", "HEAVY_TRANSPORT", "OUTSOURCED_WORKER"):
                    a2a_res = negotiate_partner_via_a2a(
                        outsourcing_domain=dom,
                        target_branch=str(t_args.get("branch_code") or params.get("branch_code") or state.get("branch_code") or "BR-ULSAN-01"),
                        requested_qty=t_args.get("requested_qty") or params.get("quantity"),
                        weight_ton=t_args.get("weight_ton") or params.get("weight_ton"),
                        material=str(t_args.get("material") or params.get("material") or ""),
                        process=str(t_args.get("process") or params.get("process") or ""),
                        worker_name=str(t_args.get("worker_name") or params.get("worker_name") or ""),
                        requested_item_name=str(params.get("requested_item_name") or ""),
                        urgency=str(t_args.get("urgency") or params.get("urgency") or "NORMAL"),
                    )
                    p_plan["a2a_negotiation"] = a2a_res
                    a2a_prop = a2a_res.get("proposal") or {}
                    if a2a_res.get("status") == "SUCCESS" and a2a_prop:
                        sel = dict(p_plan.get("selected_partner") or {})
                        if a2a_prop.get("partner_id"):
                            sel["partner_id"] = a2a_prop["partner_id"]
                            sel["partner_name"] = a2a_prop.get("partner_name") or sel.get("partner_name")
                        if a2a_prop.get("assigned_vehicle_code"):
                            sel["assigned_vehicle_code"] = a2a_prop["assigned_vehicle_code"]
                            sel["max_load_ton"] = a2a_prop.get("max_load_ton")
                            sel["business_type"] = "HEAVY_TRANSPORT"
                        if a2a_prop.get("agreed_unit_price_krw"):
                            sel["unit_price_krw"] = a2a_prop["agreed_unit_price_krw"]
                        # [보강 2] 외주 추천 작업자 성명 동기화
                        if a2a_prop.get("recommended_worker"):
                            sel.update(a2a_prop["recommended_worker"])
                            if sel.get("worker_name"):
                                params["worker_name"] = sel["worker_name"]
                        p_plan["selected_partner"] = sel

                partner_plan = p_plan
                obs = p_plan

            elif t_name == "tool_get_vision_overview":
                vision_data = get_vision_overview_for_qa.invoke({
                    "block_id": t_args.get("block_id") or params.get("block_id"),
                    "camera_code": t_args.get("camera_code") or params.get("camera_code"),
                })
                obs = vision_data

            elif t_name == "tool_list_all_workers":
                obs = list_erp_workers.invoke({"branch_code": t_args.get("branch_code")})
                if isinstance(obs, dict) and obs.get("workers"):
                    erp_status["available_workers"] = obs["workers"]

            else:
                obs = {"status": "ERROR", "message": f"알 수 없는 도구: {t_name}"}

        except Exception as e:
            obs = {"status": "ERROR", "error": str(e)}

        tool_trace.append({
            "step": state.get("react_step", 0),
            "tool": t_name,
            "args": t_args,
            "status": obs.get("status", "OK") if isinstance(obs, dict) else "OK",
        })
        r_msgs.append(
            ToolMessage(
                tool_call_id=t_id,
                name=t_name,
                content=_to_json(obs),
            )
        )

    return {
        "react_messages": r_msgs,
        "extracted_params": params,
        "erp_resource_status": erp_status,
        "retrieved_docs": retrieved_docs,
        "search_status": search_status,
        "doc_grade": doc_grade,
        "weather_data": weather_data,
        "vision_data": vision_data,
        "partner_erp_plan": partner_plan,
        "is_outsourcing_required": is_outsourcing,
        "preferred_execution_mode": pref_mode,
        "outsourcing_domain": outsourcing_domain,
        "agent_reasoning_summary": reasoning_summary,
        "tool_trace": tool_trace,
    }


# ==============================================================================
# Node 4: 결정론적 안전망 (ReAct가 필수 안전 데이터 호출을 빠뜨렸을 때만 자동 보완)
# ==============================================================================
def react_finalize_node(state: Team2ReActState) -> Dict[str, Any]:
    task_type = state.get("task_type", "knowledge_qa")
    params = dict(state.get("extracted_params") or {})
    erp_status = dict(state.get("erp_resource_status") or {})
    weather_data = state.get("weather_data")
    vision_data = state.get("vision_data")
    retrieved_docs = list(state.get("retrieved_docs") or [])
    partner_plan = state.get("partner_erp_plan")
    is_outsourcing = bool(state.get("is_outsourcing_required", False))
    doc_grade = state.get("doc_grade")
    search_status = state.get("search_status", "idle")

    if task_type == "doc_generation":
        # 1. 확정된 블록/작업자가 있는데 ERP 최종 동기화가 안 맞으면 1회 보정
        bm = erp_status.get("block_master") or {}
        wq = erp_status.get("worker_qualification") or {}
        if params.get("block_id") and (
            bm.get("block_id") != params.get("block_id")
            or (params.get("worker_name") and wq.get("worker_name") != params.get("worker_name"))
        ):
            erp_status = check_hq_erp_resources.invoke({
                "block_id": params.get("block_id"),
                "material": params.get("material"),
                "worker_name": params.get("worker_name"),
                "equipment": params.get("equipment"),
                "branch_code": params.get("branch_code"),
                "process": params.get("process"),
            })
            if erp_status.get("block_master"):
                for bk in ("ship_code", "block_name", "material", "thickness", "joint_type", "process", "branch_code", "weight_ton"):
                    if erp_status["block_master"].get(bk) is not None:
                        params[bk] = erp_status["block_master"][bk]
            if params.get("doc_type") == "HEAVY_LIFTING_PLAN" and params.get("weight_ton"):
                erp_status["lifting_capacity_eval"] = evaluate_heavy_lifting_capacity(
                    block_weight_ton=params.get("weight_ton"),
                    vehicles_equipment=erp_status.get("vehicles_equipment") or [],
                    target_branch=params.get("branch_code"),
                    requested_equipment=params.get("equipment"),
                )

        # 2. 기상·비전 안전 데이터 누락 시 필수 수집
        if not weather_data or not vision_data:
            weather_data, vision_data = _run_weather_and_vision_check(
                location=params.get("location"),
                branch_code=params.get("branch_code") or state.get("branch_code"),
                block_id=params.get("block_id"),
                work_id=state.get("work_id"),
                camera_code=params.get("camera_code"),
                doc_type=params.get("doc_type"),
                process=params.get("process"),
                equipment=params.get("equipment"),
                need_vision=True,
            )

        # 3. RAG 근거 문서 누락 시 1회 자동 검색
        if not retrieved_docs:
            q = (
                "중량물 취급 작업계획서 크레인 양중 줄걸이 안전수칙"
                if params.get("doc_type") == "HEAVY_LIFTING_PLAN"
                else f"{params.get('process', '')} {params.get('material', '')} {params.get('thickness', '')} WPS 시공 조건 건조 보관 NDT".strip()
            )
            raw_items = search_knowledge_by_categories(query=q, categories=None, top_k=4)
            for item in raw_items:
                retrieved_docs.append({
                    "category": item.get("category", "규격"),
                    "content": item.get("chunk_text", "") or str(item),
                    "metadata": {
                        "document_name": item.get("document_name"),
                        "document_code": item.get("document_code"),
                        "page": item.get("page_start"),
                        "similarity": item.get("similarity_score"),
                        "source": item.get("source"),
                    },
                })
            if retrieved_docs:
                search_status = "completed"
                doc_grade = "yes"

        # 4. 외주 필요 판정 시 협력사 협의(A2A 포함) 누락 보완
        is_lifting_need_a2a = (
            params.get("doc_type") == "HEAVY_LIFTING_PLAN"
            and bool((erp_status.get("lifting_capacity_eval") or {}).get("need_outsourcing"))
            and not bool(params.get("user_explicit_equipment"))
            and not (((partner_plan or {}).get("a2a_negotiation") or {}).get("proposal") or {}).get("assigned_vehicle_code")
        )
        if is_lifting_need_a2a:
            p_plan = query_partner_erp_b2b.invoke({
                "outsourcing_domain": "HEAVY_TRANSPORT",
                "material": params.get("material") or "",
                "process": "",
                "worker_name": "",
            })
            a2a_res = negotiate_partner_via_a2a(
                outsourcing_domain="HEAVY_TRANSPORT",
                target_branch=str(params.get("branch_code") or state.get("branch_code") or "BR-ULSAN-01"),
                weight_ton=params.get("weight_ton"),
                urgency=str(params.get("urgency") or "NORMAL"),
            )
            p_plan["a2a_negotiation"] = a2a_res
            a2a_prop = a2a_res.get("proposal") or {}
            if a2a_res.get("status") == "SUCCESS" and a2a_prop.get("assigned_vehicle_code"):
                sel = dict(p_plan.get("selected_partner") or {})
                sel.update({
                    "partner_id": a2a_prop.get("partner_id") or sel.get("partner_id"),
                    "partner_name": a2a_prop.get("partner_name") or sel.get("partner_name"),
                    "assigned_vehicle_code": a2a_prop["assigned_vehicle_code"],
                    "max_load_ton": a2a_prop.get("max_load_ton"),
                    "unit_price_krw": a2a_prop.get("agreed_unit_price_krw"),
                    "business_type": "HEAVY_TRANSPORT",
                })
                p_plan["selected_partner"] = sel
            partner_plan = p_plan
            is_outsourcing = True
        elif (is_outsourcing or erp_status.get("require_outsourcing")) and not partner_plan:
            dom = state.get("outsourcing_domain") or "WELDING_MATERIAL"
            if dom == "NOT_APPLICABLE":
                dom = "HEAVY_TRANSPORT" if params.get("doc_type") == "HEAVY_LIFTING_PLAN" else "WELDING_MATERIAL"
            partner_plan = query_partner_erp_b2b.invoke({
                "outsourcing_domain": dom,
                "material": params.get("material") or "",
                "process": params.get("process") or "",
                "worker_name": params.get("worker_name") or "",
            })

    return {
        "extracted_params": params,
        "erp_resource_status": erp_status,
        "weather_data": weather_data,
        "vision_data": vision_data,
        "retrieved_docs": retrieved_docs,
        "search_status": search_status,
        "doc_grade": doc_grade,
        "partner_erp_plan": partner_plan,
        "is_outsourcing_required": is_outsourcing or bool(erp_status.get("require_outsourcing")),
    }


# ==============================================================================
# Team 2 ReAct SubGraph 조립
# ==============================================================================
builder = StateGraph(Team2ReActState)

builder.add_node("react_init", react_init_node)
builder.add_node("react_agent", react_agent_node)
builder.add_node("react_tool_executor", react_tool_executor_node)
builder.add_node("react_finalize", react_finalize_node)

builder.add_edge(START, "react_init")
builder.add_edge("react_init", "react_agent")
builder.add_conditional_edges(
    "react_agent",
    route_after_react_agent,
    {
        "react_tool_executor": "react_tool_executor",
        "react_finalize": "react_finalize",
    },
)
builder.add_edge("react_tool_executor", "react_agent")
builder.add_edge("react_finalize", END)

team2_subgraph = builder.compile()