# modules/agent/subgraphs/planning.py
"""
Team 4: 멀티모달 AI 도면 해석 & 자율 블록 분할(Block Division) -> ERP 자원 제약 APS 스케줄링 서브그래프
- 고정 블록 개수/고정 명칭 없이 AI가 도면 이미지와 공간 좌표 클러스터를 판독해 블록을 분할한다.
- 기존 erp_tools(check_hq_erp_resources, find_eligible_workers)를 재사용해 야드 자원과 결합한다.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from datetime import timedelta

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing import Literal

from modules.agent.state import ShipAgentState
from modules.agent.tools.cad_tools import (
    extract_cad_spatial_facts,
    get_daily_work_briefing,
)
from modules.agent.tools.erp_tools import (
    check_hq_erp_resources,
    find_eligible_workers,
)
from modules.agent.tools.vision_tools import user_engine

# 도면 이미지 + 공간 좌표 판독 및 자율 블록 분할용 멀티모달 정밀 모델
planner_llm = ChatOpenAI(model="gpt-5.1")


class SubWeldTask(BaseModel):
    task_name: str = Field(description="세부 용접 작업명 (예: 종·횡 거더 맞대기 용접, 늑골 필렛 보강 용접)")
    joint_type: str = Field(description="BUTT 또는 FILLET")
    process_code: str = Field(description="SAW 또는 FCAW")
    est_weld_length_mm: float = Field(description="해당 세부 작업의 추정 용접장(mm)")


class DividedBlockPlan(BaseModel):
    priority_rank: int = Field(description="선박 건조 순서(하부 Tanktop/Double Bottom 우선 -> 상부 Deck 후행)에 따른 착수 순위 (1부터 순차)")
    block_code: str = Field(description="도면의 데크·프레임 구역을 반영한 고유 블록 코드 (예: GUN-TT-FR00-15, GUN-DK-FR00-27)")
    block_name: str = Field(description="도면의 실제 데크명·프레임 구간·구조 특징을 반영한 블록 명칭")
    deck_level: str = Field(description="도면에서 식별된 데크 층 (예: TANKTOP AND SINGLE BOTTOM, 1-DECK 4200 ABOVE BL)")
    frame_range: str = Field(description="해당 블록이 포함하는 프레임 범위 (예: Fr.0~15)")
    material_grade: str = Field(description="AH36 또는 SS400")
    thickness_mm: float = Field(description="대표 판두께(mm)")
    process_code: Literal["SAW", "FCAW"] = Field(
        description="블록의 대표 주 용접 공정 (반드시 'SAW' 또는 'FCAW' 중 단일 값 선택, '+' 결합 금지)"
    )
    joint_type: Literal["BUTT", "FILLET"] = Field(
        description="블록의 대표 주 이음 형식 (반드시 'BUTT' 또는 'FILLET' 중 단일 값 선택)"
    )
    wire_mat_code: Literal["WIRE-SAW-L60", "WIRE-CSF71T"] = Field(
        description="대표 주 공정에 대응하는 단일 와이어 품목코드 (SAW면 'WIRE-SAW-L60', FCAW면 'WIRE-CSF71T')"
    )
    est_weld_length_mm: float = Field(description="블록 총 주용접장(mm)")
    est_weight_ton: float = Field(description="부재 규모와 두께를 고려한 추정 블록 중량(톤)")
    est_wire_kg: float = Field(description="예상 용접 와이어 소요량(kg)")
    est_man_hours: float = Field(description="취부·용접·정리를 포함한 예상 총 공수(Man-Hours)")
    est_duration_days: int = Field(description="야드 가용 인원 기준 예상 작업 소요일수 (1~4일)")
    planned_start_date: str = Field(description="예상 착수일 (YYYY-MM-DD)")
    planned_end_date: str = Field(description="예상 완료일 (YYYY-MM-DD)")
    need_material_po: bool = Field(description="해당 블록 착수 전 와이어 자재 부족으로 선제 발주(PO)가 필요하면 True")
    need_external_lifting: bool = Field(description="추정 중량이 야드 크레인 정격하중 안전 한계(정격×90%)를 초과해 외주 트랜스포터가 필요하면 True")
    sub_tasks: List[SubWeldTask] = Field(description="이 블록을 완성하기 위한 세부 용접 작업 2~3개")
    schedule_rationale: str = Field(description="블록 분할 기준, 건조 순서, 자재·설비 제약을 반영한 편성 근거 (1~2문장)")


class AiCadAnalysisAndSchedule(BaseModel):
    drawing_interpretation: str = Field(
        description="도면 전체 구조 해석 요약: 식별된 갑판 층수, 프레임 범위, 주요 부재 규격, 알루미늄 등 제외 구역 설명 (3문장 이내)"
    )
    block_division_criterion: str = Field(
        description="블록 분할 기준 요약: 수밀격벽/기관실 구획 및 야드 크레인 정격하중 안전 한계를 어떻게 반영해 몇 개 블록으로 나눴는지 설명"
    )
    divided_blocks: List[DividedBlockPlan] = Field(
        min_length=1,
        description="도면의 구조 구획·프레임 범위·크레인 정격하중 한계에 따라 1개 이상 유동적으로 분할된 제작 블록 목록 (단일 소규모 도면은 억지 분할 금지)",
    )


structured_cad_planner = planner_llm.with_structured_output(AiCadAnalysisAndSchedule)


# ==============================================================================
# Node 1: CAD 공간 좌표·이미지 추출 & 본사 ERP 자원 통합 조회 (Pure Tools)
# ==============================================================================
def cad_metric_collector_node(state: ShipAgentState) -> Dict[str, Any]:
    params = state.get("extracted_params") or {}
    branch_code = params.get("branch_code") or state.get("branch_code") or "BR-ULSAN-01"
    work_zone = params.get("work_zone") or state.get("default_workspace") or "ZONE-A-BAY1"
    dxf_path = str(state.get("cad_dxf_path") or "").strip()
    pdf_path = str(state.get("cad_pdf_path") or "").strip()

    if not dxf_path:
        briefing = get_daily_work_briefing(user_engine=user_engine, branch_code=branch_code)
        return {"cad_staging_result": {"status": "BRIEFING_ONLY", "briefing": briefing}}

    extracted = extract_cad_spatial_facts(
        dxf_path_str=dxf_path,
        pdf_path_str=pdf_path or None,
        branch_code=branch_code,
        work_zone=work_zone,
        user_engine=user_engine,
    )
    if extracted.get("status") != "EXTRACTED":
        return {"cad_staging_result": extracted}

    # 기존 erp_tools.py 함수 재사용으로 야드 자재·크레인·가용 작업자·기존 진행 블록 수집
    erp_res = check_hq_erp_resources.invoke({"branch_code": branch_code})
    workers_res = find_eligible_workers.invoke({"branch_code": branch_code})

    crane_assets = [
        eq for eq in (erp_res.get("vehicles_equipment") or [])
        if str(eq.get("branch_code") or "").upper() == branch_code.upper()
        and any(k in str(eq.get("asset_category") or "").upper() for k in ("CRANE", "TRANSPORTER", "FORKLIFT"))
    ]
    wire_inventory = {
        code: {
            "item_name": info.get("item_name"),
            "available_qty": info.get("available_qty")
            if str(info.get("branch_code") or "").upper() == branch_code.upper()
            else 0.0,
            "unit": info.get("unit"),
        }
        for code, info in (erp_res.get("inventory") or {}).items()
        if str(info.get("material_category") or "").upper() == "FILLER_WIRE"
    }
    eligible_workers_summary = [
        {
            "worker_id": w.get("worker_id"),
            "worker_name": w.get("worker_name"),
            "weld_process": w.get("weld_process"),
            "company_code": w.get("company_code"),
        }
        for w in (workers_res.get("workers") or [])
    ]

    extracted["yard_context"] = {
        "branch_info": erp_res.get("branch_info") or {},
        "base_start_date": datetime.now().strftime("%Y-%m-%d"),
        "wire_inventory": wire_inventory,
        "lifting_equipment": crane_assets,
        "eligible_workers": eligible_workers_summary,
        "existing_blocks": [
            {
                "block_id": b.get("block_id"),
                "block_name": b.get("block_name"),
                "is_active_work": b.get("is_active_work"),
            }
            for b in (erp_res.get("available_blocks") or [])
            if str(b.get("branch_code") or "").upper() == branch_code.upper()
            and not str(b.get("block_id") or "").upper().startswith("GUN-")
        ],
    }
    return {"cad_staging_result": extracted}


# ==============================================================================
# Node 2: 멀티모달 AI 도면 해석 & 자율 블록 분할 -> 자원 제약 APS 일정 편성
# ==============================================================================
def llm_aps_scheduler_node(state: ShipAgentState) -> Dict[str, Any]:
    staging = dict(state.get("cad_staging_result") or {})
    status = staging.get("status")

    if status == "REJECTED":
        msg = f"⛔ **CAD 도면 업로드 차단 ({staging.get('gate')})**\n- 사유: {staging.get('reason', '도면 검증 실패')}"
        return {
            "cad_staging_result": staging,
            "cad_dxf_path": None,
            "cad_pdf_path": None,
            "final_answer": msg,
            "messages": [AIMessage(content=msg)],
        }

    if status == "BRIEFING_ONLY":
        briefing = staging.get("briefing") or {}
        carry = briefing.get("carry_over_tasks") or []
        pending = briefing.get("pending_tasks") or []
        p1 = briefing.get("next_p1_task")

        lines = [f"### 📋 [{briefing.get('branch_code')}] 금일 공정 계획 브리핑"]
        if carry:
            lines.append("\n**1. 진행 중인 작업 (`IN_PROGRESS`)**")
            for c in carry:
                lines.append(
                    f"- `{c['block_code']}` {c['block_name']} | 진척률 **{c['progress_pct']}%** (지시서: `{c['assigned_wo_number']}`)"
                )
        if pending:
            lines.append("\n**2. 신규 착수 대기 작업 큐 (`PENDING`)**")
            for p in pending:
                lines.append(
                    f"- **[P{p['priority_rank']}] `{p['block_code']}`** {p['block_name']} "
                    f"({p['process_code']} · {p['material_grade']} {p['thickness_mm']}T · "
                    f"일정: {p.get('planned_start_date') or '-'} ~ {p.get('planned_end_date') or '-'})"
                )
        if p1:
            lines.append(
                f"\n👉 오늘 신규 착수 1순위는 **`{p1['block_code']}`**입니다. 작업지시서와 중량물 계획서를 바로 편성할까요?"
            )
        elif not carry:
            lines.append("\n현재 등록된 작업 계획이 없습니다. CAD 구조 도면(`.dxf`)을 업로드해 공정 계획을 생성해 주세요.")

        reply_text = "\n".join(lines)
        return {
            "cad_staging_result": staging,
            "final_answer": reply_text,
            "messages": [AIMessage(content=reply_text)],
        }

    dwg_meta: Dict[str, Any] = staging.get("drawing_meta") or {}
    spatial_views: List[Dict[str, Any]] = staging.get("spatial_views") or []
    yard_ctx: Dict[str, Any] = staging.get("yard_context") or {}
    image_b64: Optional[str] = staging.pop("drawing_image_b64", None)

    prompt_text = (
        "당신은 조선소 선체 생산설계(Block Division) 및 공정계획(APS) 수석 엔지니어입니다.\n"
        "제공된 [선체 구조 도면 청사진 이미지]와 [DXF 공간 좌표 기반 뷰·프레임 추출 데이터], 그리고 [대상 야드 설비·인력·재고 현황]을 정밀 분석하여\n"
        "1) 도면 전체의 구조(갑판 층수, 프레임 구간, 기관실/SEA CHEST 등 특이 구획)를 해석하고,\n"
        "2) 실제 조선소 조립·인양 가능한 **블록 단위(N개)**로 자율 분할한 뒤,\n"
        "3) 선박 건조 순서와 자원 제약을 반영한 공정 계획 및 대략적 일정을 수립하십시오.\n\n"
        "[필수 엔지니어링 규칙]\n"
        "1. **고정 개수·억지 분할 금지**: 도면에 실제로 존재하는 뷰(`TANKTOP AND SINGLE BOTTOM`, `1-DECK 4200 ABOVE BL`, `DOUBLE BOTTOM` 등)와 "
        "실제 프레임 번호(`Fr.0~15`, `Fr.15~27` 등)를 기준으로 분할하세요. 크레인 정격하중 안전 한계(예: 50톤 크레인의 90% = 45톤)와 실제 구조 구획에 따라 "
        "단일 소규모 블록이면 1개로 유지하고, 대형 데크나 이중저만 기관실·중앙·선수 구역 등으로 필요한 개수만큼 합리적으로 나누십시오.\n"
        "2. **선박 건조 순서 엄수**: 반드시 하부 구조(`TANKTOP`, `DOUBLE BOTTOM`, `SINGLE BOTTOM`, 기관실 받침대 구역)를 상부 갑판(`1-DECK`, `A-DECK`)보다 먼저(P1, P2...) 배치하십시오.\n"
        "3. **세부 용접 작업(sub_tasks) 도출**: 각 블록마다 도면의 부재 규격(`HP80x6`, `HP140x8`, `W200x8`, `FL150x10` 등)을 반영해 맞대기(`BUTT`/SAW)와 필렛(`FILLET`/FCAW) 세부 작업을 2~3개 도출하고, "
        "취부·용접·정리를 포함한 현실적인 공수(`est_man_hours`)와 가용 작업자 수 기준 소요일수(`est_duration_days`)를 계산하십시오.\n"
        "4. **자재·양중 제약 반영**: 야드 가용 와이어 재고가 부족한 공정의 블록은 `need_material_po=True`로 표시하고 조달 리드타임(1일)을 반영해 착수일을 잡으십시오.\n\n"
        f"- 선박명: {dwg_meta.get('vessel_name')} | 도면명: {dwg_meta.get('drawing_title')}\n"
        f"- 대상 야드: {dwg_meta.get('branch_code')} / 작업구역: {dwg_meta.get('work_zone')}\n"
        f"- 제외된 비강재(알루미늄) 구역: {json.dumps(dwg_meta.get('excluded_non_steel_views'), ensure_ascii=False)}\n"
        f"- DXF 공간 좌표 기반 추출 뷰 및 프레임 데이터:\n{json.dumps(spatial_views, ensure_ascii=False, indent=2)}\n"
        f"- 대상 야드 실시간 자원(크레인·가용 작업자·와이어 재고·기존 블록):\n{json.dumps(yard_ctx, ensure_ascii=False, indent=2)}"
    )

    msg_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    if image_b64:
        msg_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
        })

    analysis_out: AiCadAnalysisAndSchedule = structured_cad_planner.invoke([
        HumanMessage(content=msg_content)
    ])

    wire_inv = yard_ctx.get("wire_inventory") or {}
    remaining_wire: Dict[str, float] = {
        code: float((info or {}).get("available_qty") or 0.0)
        for code, info in wire_inv.items()
    }
    crane_caps = [
        float(eq.get("max_load_ton") or 0.0)
        for eq in (yard_ctx.get("lifting_equipment") or [])
        if eq.get("max_load_ton") is not None
    ]
    max_safe_crane_ton = round(max(crane_caps) * 0.9, 2) if crane_caps else None

    base_dt_str = yard_ctx.get("base_start_date") or datetime.now().strftime("%Y-%m-%d")
    base_dt = datetime.strptime(base_dt_str, "%Y-%m-%d")
    material_ready_dt = base_dt + timedelta(days=1)

    lower_keywords = ("TANKTOP", "DOUBLE BOTTOM", "SINGLE BOTTOM", "BOTTOM", "ENGINE")
    upper_keywords = ("1-DECK", "2-DECK", "A-DECK", "MAIN DECK", "UPPER DECK")
    lower_latest_end: Optional[datetime] = None

    scheduled_packages: List[Dict[str, Any]] = []
    for blk in sorted(analysis_out.divided_blocks, key=lambda x: x.priority_rank):
        pkg = blk.model_dump()
        pkg["plan_state"] = "STAGING"
        pkg["estimation_status"] = "AI_ESTIMATED_REVIEW_REQUIRED"

        duration_days = max(1, int(pkg.get("est_duration_days") or 1))
        try:
            s_dt = datetime.strptime(str(pkg.get("planned_start_date") or base_dt_str), "%Y-%m-%d")
        except ValueError:
            s_dt = base_dt
        try:
            e_dt = datetime.strptime(str(pkg.get("planned_end_date") or ""), "%Y-%m-%d")
            span_days = max(1, (e_dt - s_dt).days)
        except ValueError:
            span_days = duration_days
            e_dt = s_dt + timedelta(days=span_days)

        # 1) 와이어 품목별 누적 재고 차감 및 해당 블록 단독 리드타임 반영
        w_code = pkg["wire_mat_code"]
        req_wire = float(pkg.get("est_wire_kg") or 0.0)
        avail_before = remaining_wire.get(w_code, 0.0)

        if avail_before < req_wire:
            pkg["need_material_po"] = True
            if s_dt < material_ready_dt:
                s_dt = material_ready_dt
                e_dt = s_dt + timedelta(days=span_days)
                if "리드타임" not in pkg.get("schedule_rationale", ""):
                    pkg["schedule_rationale"] = (
                        f"{pkg.get('schedule_rationale', '').rstrip()} "
                        f"({w_code} 가용 {round(avail_before, 1)}kg < 소요 {round(req_wire, 1)}kg로 조달 리드타임 1일 반영)"
                    ).strip()
        else:
            pkg["need_material_po"] = False

        remaining_wire[w_code] = max(0.0, avail_before - req_wire)

        # 2) 선후행 구조 제약: 하부 구조 일정이 밀렸을 때 상부 갑판이 하부 완료일보다 먼저 시작되지 않도록 보정
        deck_upper = str(pkg.get("deck_level") or pkg.get("block_name") or "").upper()
        is_lower = any(k in deck_upper for k in lower_keywords)
        is_upper = (not is_lower) and any(k in deck_upper for k in upper_keywords)

        if is_upper and lower_latest_end is not None and s_dt < lower_latest_end:
            s_dt = lower_latest_end
            e_dt = s_dt + timedelta(days=span_days)
            if "하부 구조 완료" not in pkg.get("schedule_rationale", ""):
                pkg["schedule_rationale"] = (
                    f"{pkg.get('schedule_rationale', '').rstrip()} "
                    f"(하부 구조 완료일 {lower_latest_end.strftime('%Y-%m-%d')} 이후 착수로 선후행 보정)"
                ).strip()

        if is_lower:
            lower_latest_end = e_dt if (lower_latest_end is None or e_dt > lower_latest_end) else lower_latest_end

        # 3) 야드 크레인 안전 한계(정격 90%) 결정론적 재검증
        if max_safe_crane_ton is not None:
            pkg["need_external_lifting"] = float(pkg.get("est_weight_ton") or 0.0) > max_safe_crane_ton

        pkg["planned_start_date"] = s_dt.strftime("%Y-%m-%d")
        pkg["planned_end_date"] = e_dt.strftime("%Y-%m-%d")
        scheduled_packages.append(pkg)

    staging["status"] = "STAGING_SUCCESS"
    staging["drawing_interpretation"] = analysis_out.drawing_interpretation
    staging["block_division_criterion"] = analysis_out.block_division_criterion
    staging["work_packages"] = scheduled_packages

    summary_lines = [
        f"### 📐 AI CAD 구조 해석 및 자율 블록 분할 완료 (`{dwg_meta.get('drawing_title')}`)",
        f"- **도면 구조 해석:** {analysis_out.drawing_interpretation}",
        f"- **블록 분할 기준:** {analysis_out.block_division_criterion}\n",
    ]
    for p in scheduled_packages:
        flags = []
        if p.get("need_material_po"):
            flags.append("⚠️ 자재 선제발주 필요")
        if p.get("need_external_lifting"):
            flags.append("🏗️ 외주 트랜스포터 필요")
        flag_str = f" | {' · '.join(flags)}" if flags else ""
        summary_lines.append(
            f"- **[P{p['priority_rank']}] `{p['block_code']}`** — {p['block_name']} ({p['frame_range']})\n"
            f"  · 제원: {p['material_grade']} {p['thickness_mm']}T · {p['process_code']}/{p['joint_type']} · "
            f"용접장 {p['est_weld_length_mm']}mm · 추정중량 **{p['est_weight_ton']}톤** · 공수 **{p['est_man_hours']} M/H** ({p['est_duration_days']}일)\n"
            f"  · 일정: **{p['planned_start_date']} ~ {p['planned_end_date']}**{flag_str}\n"
            f"  · 근거: {p['schedule_rationale']}"
        )

    reply_text = "\n".join(summary_lines)
    return {
        "cad_staging_result": staging,
        "cad_dxf_path": None,
        "cad_pdf_path": None,
        "final_answer": reply_text,
        "messages": [AIMessage(content=reply_text)],
    }


builder = StateGraph(ShipAgentState)
builder.add_node("cad_metric_collector", cad_metric_collector_node)
builder.add_node("llm_aps_scheduler", llm_aps_scheduler_node)

builder.add_edge(START, "cad_metric_collector")
builder.add_edge("cad_metric_collector", "llm_aps_scheduler")
builder.add_edge("llm_aps_scheduler", END)

team4_subgraph = builder.compile()