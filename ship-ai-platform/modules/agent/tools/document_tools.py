"""
modules/agent/tools/document_tools.py
에이전트가 발행한 구조화 문서를 표준 DOCX/PDF 템플릿 치환, 외주 발주 주문서(PO) PDF 생성 및 Audit 로그로 저장하는 표준 도구 모음
"""
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from docxtpl import DocxTemplate
from langchain_core.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "generated_docs"
TEMPLATE_PATH = OUTPUT_DIR / "templates" / "work_order_template.docx"
DAILY_OUTPUT_ROOT = OUTPUT_DIR / "outputs"


def _prepare_korean_font_css() -> tuple[str, str]:
    """xhtml2pdf용 한글 폰트 로컬 복사 및 @font-face CSS 반환"""
    try:
        from reportlab.pdfbase import pdfmetrics  # type: ignore
        from reportlab.pdfbase.ttfonts import TTFont  # type: ignore
        from reportlab.lib.fonts import addMapping  # type: ignore
    except ImportError:
        return "sans-serif", ""

    local_font_dir = OUTPUT_DIR / "fonts"
    local_font_dir.mkdir(parents=True, exist_ok=True)

    font_pairs = [
        (
            "KorFont",
            str(local_font_dir / "malgun.ttf"),
            str(local_font_dir / "malgunbd.ttf"),
        ),
        (
            "KorFont",
            r"C:\Windows\Fonts\malgun.ttf",
            r"C:\Windows\Fonts\malgunbd.ttf",
        ),
        (
            "KorFont",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        ),
    ]

    for f_name, sys_reg, sys_bold in font_pairs:
        if os.path.exists(sys_reg):
            try:
                local_reg = local_font_dir / Path(sys_reg).name
                if not local_reg.exists():
                    shutil.copy2(sys_reg, local_reg)

                local_bold = local_font_dir / Path(sys_bold).name
                if os.path.exists(sys_bold) and not local_bold.exists():
                    shutil.copy2(sys_bold, local_bold)
                elif not local_bold.exists():
                    local_bold = local_reg

                reg_str = str(local_reg.resolve()).replace("\\", "/")
                bold_str = str(local_bold.resolve()).replace("\\", "/")

                pdfmetrics.registerFont(TTFont(f_name, reg_str))
                pdfmetrics.registerFont(TTFont(f"{f_name}-Bold", bold_str))
                addMapping(f_name, 0, 0, f_name)
                addMapping(f_name, 0, 1, f_name)
                addMapping(f_name, 1, 0, f"{f_name}-Bold")
                addMapping(f_name, 1, 1, f"{f_name}-Bold")

                font_face_css = f"""
                @font-face {{
                    font-family: '{f_name}';
                    src: url('{reg_str}');
                    font-weight: normal;
                }}
                @font-face {{
                    font-family: '{f_name}';
                    src: url('{bold_str}');
                    font-weight: bold;
                }}
                """
                return f"'{f_name}'", font_face_css
            except Exception as e:
                print(f"⚠️ [한글 폰트 로컬 복사/등록 경고]: {e}")

    return "sans-serif", ""


def _render_html_to_pdf_fallback(context: dict, pdf_path: Path) -> bool:
    """3순위 폴백: 프로젝트 내부 폴더로 폰트를 복사해 xhtml2pdf 보안 정책(Resource Policy) 완벽 우회"""
    try:
        from xhtml2pdf import pisa  # type: ignore
    except ImportError:
        print("⚠️ [PDF 3순위 폴백 실패]: xhtml2pdf 미설치 (pip install xhtml2pdf 필요)")
        return False

    font_family, font_face_css = _prepare_korean_font_css()

    weather_snapshot = (
        f"관측 야드: {context.get('location', '-')} | "
        f"기온: {context.get('temp', '-')}℃ | "
        f"풍속: {context.get('wind', '-')}m/s | "
        f"강수량: {context.get('rain', '-')}mm/h | "
        f"안전 판정: {context.get('safety_status', '-')}"
    )

    checklist_html = str(context.get("safety_checklist_str", "-")).replace(
        "\n", "<br/>"
    )

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<style>
    {font_face_css}
    @page {{ size: A4 portrait; margin: 9mm 11mm; }}
    * {{ font-family: {font_family}; }}
    body {{ font-family: {font_family}; font-size: 8.0pt; color: #111111; line-height: 1.3; }}
    .title {{ text-align: center; font-size: 14.0pt; font-weight: bold; margin-bottom: 8px; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 5px; }}
    th, td {{ border: 0.5pt solid #333333; padding: 4px 5px; vertical-align: middle; font-family: {font_family}; }}
    th {{ background-color: #E8EEF5; font-weight: bold; text-align: left; width: 18%; }}
    td {{ width: 32%; }}
    .sec-hdr {{ background-color: #1F4E79; color: #FFFFFF; font-weight: bold; font-size: 9.0pt; padding: 4px 7px; }}
</style>
</head>
<body>
    <div class="title">선 박 블 록 용 접 작 업 지 시 서 (WELDING WORK ORDER)</div>

    <table>
        <tr>
            <th>문서번호</th><td><b>{context.get('work_order_number', '-')}</b></td>
            <th>개정 / 상태</th><td><b>{context.get('revision_no', '-')} / {context.get('approval_status', '-')}</b></td>
        </tr>
        <tr>
            <th>발행일자</th><td>{context.get('created_date', '-')}</td>
            <th>결재 (작성/검토/승인)</th><td>작성: AI Agent | 검토: {context.get('reviewed_by', '-')} | 승인: {context.get('approved_by', '-')}</td>
        </tr>
        <tr>
            <th>수행구분</th><td><b>{context.get('execution_type', '-')}</b></td>
            <th>협력사명</th><td>{context.get('partner_name', '-')}</td>
        </tr>
    </table>

    <table>
        <tr><td colspan="4" class="sec-hdr">■ 1. 대상 블록 및 공정 위치 정보</td></tr>
        <tr>
            <th>대상 호선 (SHIP_CODE)</th><td>{context.get('ship_code', '-')}</td>
            <th>대상 블록 (BLOCK_CODE)</th><td><b>{context.get('block_id', '-')}</b></td>
        </tr>
        <tr>
            <th>용접 부재 / 블록명</th><td>{context.get('block_name', '-')}</td>
            <th>블록 중량 (WEIGHT_TON)</th><td>{context.get('weight_ton', '-')} 톤</td>
        </tr>
        <tr>
            <th>작업 야드 (BRANCH)</th><td>{context.get('branch_name', '-')}</td>
            <th>작업 구역 (WORK_ZONE)</th><td>{context.get('work_zone', '-')}</td>
        </tr>

        <tr><td colspan="4" class="sec-hdr">■ 2. 투입 인력 · 자재 · 장비 배정 현황</td></tr>
        <tr>
            <th>담당 작업자 / 사번</th><td><b>{context.get('worker_name', '-')} ({context.get('worker_id', '-')})</b></td>
            <th>선급 자격 / 인증공정</th><td>{context.get('cert_code', '-')} ({context.get('weld_process', '-')})</td>
        </tr>
        <tr>
            <th>배정 강재 Lot</th><td>{context.get('allocated_lot_no', '-')}</td>
            <th>용접 와이어 / 선점량</th><td>{context.get('wire_spec_text', '-')} / {context.get('reserved_wire_kg', '-')} kg</td>
        </tr>
        <tr>
            <th>배차 장비 / 차량</th><td>{context.get('assigned_vehicle_code', '-')}</td>
            <th>외주 발주(PO) 번호</th><td>{context.get('b2b_po_number', '-')}</td>
        </tr>

        <tr><td colspan="4" class="sec-hdr">■ 3. WPS 용접 시공 조건 및 현장 비전 안전 인터록</td></tr>
        <tr>
            <th>적용 절차서 (WPS NO)</th><td><b>{context.get('wps_code', '-')}</b></td>
            <th>모재 강종 / 두께</th><td><b>{context.get('material', '-')} / {context.get('thickness', '-')} mm</b></td>
        </tr>
        <tr>
            <th>용접 공정 / 이음형상</th><td><b>{context.get('process', '-')} / {context.get('joint_type', '-')} ({context.get('weld_position', '-')})</b></td>
            <th>용가재 규격 / 품목</th><td>{context.get('wire_spec_text', '-')}</td>
        </tr>
        <tr>
            <th>표준 전류 / 전압</th><td>{context.get('current_amp', '-')} A / {context.get('voltage_v', '-')} V</td>
            <th>이송 속도 / 보호가스</th><td>{context.get('travel_speed', '-')} cm/min / {context.get('shielding_gas', '-')}</td>
        </tr>
        <tr>
            <th>예열 및 층간온도</th><td>{context.get('preheat_temp', '-')}</td>
            <th>선급 품질검사 (NDT)</th><td>{context.get('ndt_requirement', '-')}</td>
        </tr>
        <tr>
            <th>용접재료 건조·관리</th><td colspan="3">{context.get('consumable_spec', '-')}</td>
        </tr>
        <tr>
            <th>YOLOv11 비전 탐지 상태</th><td>{context.get('vision_status_text', '-')}</td>
            <th>작업 구역 안전 클리어런스</th><td><b>{context.get('safety_clearance_text', '-')}</b></td>
        </tr>

        <tr><td colspan="4" class="sec-hdr">■ 4. 실시간 현장 기상 환경 및 필수 안전 준수사항 (Safety Interlock)</td></tr>
        <tr>
            <th>실시간 기상 스냅샷</th><td colspan="3">{weather_snapshot}</td>
        </tr>
        <tr>
            <th>필수 안전 체크리스트</th><td colspan="3">{checklist_html}</td>
        </tr>

        <tr><td colspan="4" class="sec-hdr">■ 5. 현장 작업 완료 확인 및 근거 규격</td></tr>
        <tr>
            <th>작업자 확인</th><td>{context.get('worker_name', '-')} (인)</td>
            <th>현장 반장 확인</th><td>____________________ (인)</td>
        </tr>
        <tr>
            <th>참조 규격 (RAG 근거)</th><td colspan="3">{context.get('source_refs', '-')}</td>
        </tr>
    </table>
</body>
</html>"""

    with open(pdf_path, "wb") as pdf_file:
        status = pisa.CreatePDF(html_content, dest=pdf_file, encoding="utf-8")
    return not status.err and pdf_path.exists()


def _convert_docx_to_pdf(
    docx_path: Path, output_dir: Path, context: dict = None
) -> str | None:
    """1순위: MS Word -> 2순위: LibreOffice -> 3순위: xhtml2pdf 자동 폴백"""
    pdf_path = output_dir / f"{docx_path.stem}.pdf"

    try:
        from docx2pdf import convert

        convert(str(docx_path), str(pdf_path))
        if pdf_path.exists():
            print("📄 [PDF 변환 완료 - 1순위 MS Word]")
            return str(pdf_path)
    except Exception:
        pass

    soffice_cmd = shutil.which("soffice") or (
        r"C:\Program Files\LibreOffice\program\soffice.exe"
        if os.path.exists(r"C:\Program Files\LibreOffice\program\soffice.exe")
        else None
    )
    if soffice_cmd:
        try:
            subprocess.run(
                [
                    soffice_cmd,
                    "--headless",
                    "--convert-to",
                    "pdf",
                    str(docx_path),
                    "--outdir",
                    str(output_dir),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if pdf_path.exists():
                print("📄 [PDF 변환 완료 - 2순위 LibreOffice]")
                return str(pdf_path)
        except Exception:
            pass

    if context and _render_html_to_pdf_fallback(context, pdf_path):
        print("📄 [PDF 생성 완료 - 3순위 xhtml2pdf 내장 렌더러]")
        return str(pdf_path)

    return None


@tool
def export_work_order_file(
    document_data: Dict[str, Any], file_prefix: str = "work_doc"
) -> str:
    """
    공식 작업지시서를 표준 DOCX 템플릿({{변수}})에 채워 날짜별 폴더(generated_docs/outputs/YYYY-MM-DD/)에 저장합니다.
    - 초안(DRAFT): 웹 검토용 .pdf 1개만 유지 (임시 .docx 정리, .md 미생성)
    - 승인(APPROVED): 공식 보관용 .docx + .pdf 2개 생성
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    daily_dir = DAILY_OUTPUT_ROOT / today_str
    daily_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wo_num = document_data.get("work_order_number") or timestamp
    base_name = f"{file_prefix}_{wo_num}"

    docx_path = daily_dir / f"{base_name}.docx"

    approval_status = document_data.get("approval_status") or "DRAFT"
    params = document_data.get("parameters") or {}
    weld_params = document_data.get("welding_params_json") or {}
    source_refs = document_data.get("source_refs_json") or []
    weather = document_data.get("weather") or {}
    vision = document_data.get("vision_inspection") or document_data.get("vision") or {}
    erp_commit = document_data.get("erp_commit") or {}
    erp_status = document_data.get("erp_resource_status") or {}
    partner_plan = document_data.get("partner_erp_plan") or {}
    selected_partner = partner_plan.get("selected_partner") or {}

    env = weather.get("environment") or weather
    temp_val = env.get("temp") if env.get("temp") is not None else weather.get("temperature")
    wind_val = env.get("wind") if env.get("wind") is not None else weather.get("wind_speed")
    rain_val = env.get("rain") if env.get("rain") is not None else weather.get("precipitation")

    worker_qual = erp_status.get("worker_qualification") or {}
    branch_info = erp_status.get("branch_info") or {}
    block_master = erp_status.get("block_master") or {}
    inventory_dict = erp_status.get("inventory") or {}

    resolved_worker_id = (
        erp_commit.get("worker_id")
        or selected_partner.get("assigned_worker_id")
        or worker_qual.get("worker_id")
        or "-"
    )
    resolved_worker_name = (
        erp_commit.get("worker_name")
        or selected_partner.get("worker_name")
        or params.get("worker_name")
        or worker_qual.get("worker_name")
        or "-"
    )
    resolved_cert_code = (
        selected_partner.get("cert_code")
        or worker_qual.get("cert_code")
        or weld_params.get("cert_code")
        or "-"
    )

    resolved_branch = (
        params.get("branch_code")
        or branch_info.get("branch_code")
        or block_master.get("branch_code")
        or "BR-ULSAN-01"
    )
    resolved_work_zone = (
        params.get("work_zone")
        or branch_info.get("default_zone")
        or block_master.get("location")
        or ("DOCK-1" if "GEOJE" in str(resolved_branch).upper() else "ZONE-A-BAY1")
    )

    # 수행구분, 수행사명 및 외주 발주(PO) 번호 동적 바인딩
    exec_type_raw = (
        erp_commit.get("execution_type")
        or document_data.get("execution_type")
        or "IN_HOUSE"
    )
    if exec_type_raw == "OUTSOURCED_B2B":
        resolved_partner_name = (
            selected_partner.get("partner_name")
            or erp_commit.get("partner_id")
            or "-"
        )
        po_no = (
            erp_commit.get("po_number")
            or erp_commit.get("b2b_order_number")
            or erp_commit.get("b2b_sales_order_no")
            or erp_commit.get("b2b_po_ack")
            or document_data.get("po_number")
        )
        b2b_po_display = po_no or f"PO-{str(wo_num).replace('WO-DRAFT-', '').replace('WO-', '')}"
    else:
        hq_comp = branch_info.get("company_code") or erp_status.get("hq_company_code") or "COMP-HQ"
        resolved_partner_name = f"본사 직영 ({hq_comp})"
        b2b_po_display = "해당없음 (사내직영)"

    resolved_vehicle_code = (
        document_data.get("assigned_vehicle_code")
        or erp_commit.get("assigned_vehicle_code")
        or erp_status.get("assigned_vehicle_code")
    )
    if not resolved_vehicle_code:
        for eq in (erp_status.get("vehicles_equipment") or []):
            if (
                str(eq.get("branch_code", "")).upper() == str(resolved_branch).upper()
                and str(eq.get("operation_status", "")).upper() == "STANDBY"
            ):
                resolved_vehicle_code = eq.get("asset_code")
                break
    resolved_vehicle_code = resolved_vehicle_code or "-"

    resolved_wire_code = (
        document_data.get("wire_item_code")
        or selected_partner.get("hq_item_code")
        or erp_status.get("selected_wire_code")
        or "-"
    )
    wire_item_info = inventory_dict.get(resolved_wire_code) or {}
    wire_item_name = wire_item_info.get("item_name") or selected_partner.get("item_name")
    wire_spec_text = (
        f"{resolved_wire_code} ({wire_item_name})"
        if wire_item_name and resolved_wire_code != "-"
        else resolved_wire_code
    )

    raw_wps = (
        document_data.get("wps_code")
        or weld_params.get("wps_code")
        or erp_commit.get("wps_code")
        or block_master.get("wps_code")
        or params.get("wps_code")
    )
    if not raw_wps or str(raw_wps).lower().endswith(".pdf"):
        proc_str = params.get("process") or block_master.get("process") or "WELD"
        mat_str = params.get("material") or block_master.get("material") or "STD"
        resolved_wps_code = f"WPS-{proc_str}-{mat_str}-001"
    else:
        resolved_wps_code = str(raw_wps)

    resolved_position = (
        params.get("weld_position")
        or weld_params.get("weld_position")
        or block_master.get("weld_position")
        or (
            "6G"
            if "6G" in str(resolved_cert_code)
            else ("2F" if str(block_master.get("joint_type")).upper() == "FILLET" else "1G")
        )
    )

    resolved_proc = params.get("process") or block_master.get("process") or "-"
    resolved_gas = weld_params.get("shielding_gas") or "-"
    if str(resolved_proc).upper() == "SAW" and (
        resolved_gas == "-" or any(g in str(resolved_gas).upper() for g in ("CO2", "AR", "차폐가스"))
    ):
        resolved_gas = "입상 플럭스 차폐 (보호가스 미사용)"

    raw_preheat = weld_params.get("preheat_temp_c") or weld_params.get("preheat_temp")
    if raw_preheat is not None and str(raw_preheat).strip() != "":
        p_text = str(raw_preheat).strip().replace("°C", "℃").replace("C", "℃")
        if "℃" not in p_text and any(ch.isdigit() for ch in p_text):
            preheat_display = f"{p_text}℃"
        else:
            preheat_display = p_text
    else:
        preheat_display = "-"

    clearance_yn = vision.get("safety_clearance_yn")
    insp_status = vision.get("inspection_status") or vision.get("status")
    vision_status_text = (
        f"용접선(weld_joint) 정상 탐지 ({insp_status})" if insp_status else "-"
    )
    if clearance_yn == "Y":
        safety_clearance_text = "확보 완료 (주변 간섭물 미감지 / Y)"
    elif clearance_yn == "N":
        safety_clearance_text = "위험 감지 (간섭물 제거 필수 / N)"
    else:
        safety_clearance_text = str(clearance_yn or "-")

    safety_eval = weather.get("safety_evaluation") or {}
    rules = safety_eval.get("mandatory_rules") or []
    prereqs = safety_eval.get("prerequisites") or []
    raw_checklist = [str(r).strip() for r in (rules + prereqs) if r]
    checklist_items = (
        [f"□ {r}" for r in raw_checklist]
        if raw_checklist
        else [
            "□ 작업 전 모재 용접부 수분·유분·녹 제거 및 표면 청정 상태 점검",
            "□ 용접기 접지 케이블 체결 상태 및 개인 보호구(보안면·안전대) 착용 확인",
        ]
    )
    checklist_str = "\n".join(checklist_items)
    docx_checklist = "   /   ".join(checklist_items)

    source_refs_str = (
        ", ".join(
            f"{ref.get('document_code') or ref.get('document_title', '-')}"
            f" (p.{ref.get('page', 1)})"
            for ref in source_refs
        )
        if source_refs
        else "-"
    )

    resolved_reviewer = (
        document_data.get("reviewed_by")
        or erp_commit.get("reviewer_name")
        or erp_status.get("foreman_name")
        or "현장반장(검토필)"
    )
    resolved_approver = (
        document_data.get("approved_by")
        or erp_commit.get("approved_by")
        or ("공정관리자(HITL)" if approval_status == "APPROVED" else "-")
    )

    context = {
        "work_order_number": wo_num,
        "revision_no": document_data.get("revision_no") or "Rev.1",
        "created_date": today_str,
        "approval_status": approval_status,
        "reviewed_by": resolved_reviewer,
        "approved_by": resolved_approver,
        "execution_type": exec_type_raw,
        "partner_name": resolved_partner_name,
        "ship_code": params.get("ship_code") or block_master.get("ship_code") or "-",
        "block_id": params.get("block_id") or block_master.get("block_id") or "-",
        "block_name": params.get("block_name") or block_master.get("block_name") or "-",
        "weight_ton": (
            params.get("weight_ton")
            if params.get("weight_ton") is not None
            else (block_master.get("weight_ton") or "-")
        ),
        "branch_name": resolved_branch,
        "work_zone": resolved_work_zone,
        "worker_name": resolved_worker_name,
        "worker_id": resolved_worker_id,
        "cert_code": resolved_cert_code,
        "weld_process": (
            params.get("process")
            or selected_partner.get("weld_process")
            or worker_qual.get("weld_process")
            or "-"
        ),
        "allocated_lot_no": (
            document_data.get("allocated_lot_no")
            or erp_commit.get("allocated_lot_no")
            or erp_status.get("assigned_lot_no")
            or "-"
        ),
        "wire_item_code": resolved_wire_code,
        "wire_spec_text": wire_spec_text,
        "reserved_wire_kg": (
            document_data.get("reserve_wire_kg")
            if document_data.get("reserve_wire_kg") is not None
            else (erp_commit.get("reserved_wire_kg") or "-")
        ),
        "assigned_vehicle_code": resolved_vehicle_code,
        "b2b_po_number": b2b_po_display,
        "wps_code": resolved_wps_code,
        "material": params.get("material") or block_master.get("material") or "-",
        "thickness": str(
            params.get("thickness") or block_master.get("thickness") or "-"
        ).replace("T", ""),
        "process": resolved_proc,
        "joint_type": params.get("joint_type") or block_master.get("joint_type") or "-",
        "weld_position": resolved_position,
        "current_amp": str(weld_params.get("current_amp") or "-").replace("A", "").replace("a", "").strip(),
        "voltage_v": str(weld_params.get("voltage_v") or "-").replace("V", "").replace("v", "").strip(),
        "travel_speed": str(weld_params.get("travel_speed_cm_min") or weld_params.get("travel_speed") or "-").replace("cm/min", "").strip(),
        "shielding_gas": resolved_gas,
        "preheat_temp": preheat_display,
        "ndt_requirement": (
            weld_params.get("ndt_requirement")
            or weld_params.get("ndt_spec")
            or "-"
        ),
        "consumable_spec": (
            weld_params.get("consumable_spec")
            or weld_params.get("consumable_storage_rule")
            or "-"
        ),
        "vision_status_text": vision_status_text,
        "safety_clearance_text": safety_clearance_text,
        "safety_clearance_yn": clearance_yn or "-",
        "location": params.get("location") or weather.get("location") or "-",
        "temp": temp_val if temp_val is not None else "-",
        "wind": wind_val if wind_val is not None else "-",
        "rain": rain_val if rain_val is not None else "-",
        "safety_status": safety_eval.get("status") or "-",
        "safety_checklist": docx_checklist,
        "safety_checklist_str": checklist_str,
        "source_refs": source_refs_str,
    }

    pdf_path_str = ""
    docx_created = False
    if TEMPLATE_PATH.exists():
        doc = DocxTemplate(str(TEMPLATE_PATH))
        doc.render(context)
        doc.save(str(docx_path))
        docx_created = True
        pdf_path_str = _convert_docx_to_pdf(docx_path, daily_dir, context=context)

        # 초안(DRAFT) 상태에서는 PDF만 유지하고 임시 .docx 파일은 삭제하여 불필요한 파일 누적 방지
        if approval_status == "DRAFT" and pdf_path_str and Path(pdf_path_str).exists():
            docx_path.unlink(missing_ok=True)
            docx_created = False
    else:
        print(
            f"⚠️ [템플릿 미발견]: {TEMPLATE_PATH.resolve()} 경로에"
            " work_order_template.docx 파일을 배치하세요."
        )

    pdf_filename = Path(pdf_path_str).name if pdf_path_str else None
    docx_filename = docx_path.name if docx_created else None

    result_payload = {
        "status": "SUCCESS",
        "date_folder": today_str,
        "docx_path": str(docx_path.resolve()) if docx_created else None,
        "pdf_path": pdf_path_str or None,
        "download_pdf_url": (
            f"/api/files/download/{today_str}/{pdf_filename}"
            if pdf_filename
            else None
        ),
        "download_docx_url": (
            f"/api/files/download/{today_str}/{docx_filename}"
            if docx_filename
            else None
        ),
    }
    return json.dumps(result_payload, ensure_ascii=False)


@tool
def export_outsourcing_po_pdf(document_data: Dict[str, Any]) -> str:
    """
    외주(OUTSOURCED_B2B) 작업 최종 승인 시 수주 협력사에 송부할 1페이지 공식 발주 주문서(Purchase Order) PDF를 생성합니다.
    """
    try:
        from xhtml2pdf import pisa  # type: ignore
    except ImportError:
        return json.dumps({"status": "ERROR", "reason": "xhtml2pdf 미설치"}, ensure_ascii=False)

    today_str = datetime.now().strftime("%Y-%m-%d")
    daily_dir = DAILY_OUTPUT_ROOT / today_str
    daily_dir.mkdir(parents=True, exist_ok=True)

    params = document_data.get("parameters") or {}
    erp_commit = document_data.get("erp_commit") or {}
    erp_status = document_data.get("erp_resource_status") or {}
    partner_plan = document_data.get("partner_erp_plan") or {}
    selected_partner = partner_plan.get("selected_partner") or {}
    branch_info = erp_status.get("branch_info") or {}
    block_master = erp_status.get("block_master") or {}

    wo_num = document_data.get("work_order_number") or f"WO-{datetime.now().strftime('%Y%m%d%H%M')}"
    po_num = (
        erp_commit.get("po_number")
        or erp_commit.get("b2b_order_number")
        or erp_commit.get("b2b_sales_order_no")
        or erp_commit.get("b2b_po_ack")
        or f"PO-{str(wo_num).replace('WO-DRAFT-', '').replace('WO-', '')}"
    )
    block_id = params.get("block_id") or block_master.get("block_id") or "DIRECT"
    safe_po_token = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(po_num))
    pdf_filename = f"po_order_{block_id}_{safe_po_token}.pdf"
    pdf_path = daily_dir / pdf_filename

    partner_id = selected_partner.get("partner_id") or erp_commit.get("partner_id") or "-"
    partner_name = selected_partner.get("partner_name") or partner_id
    biz_type = selected_partner.get("business_type") or "OUTSOURCED_B2B"

    branch_code = params.get("branch_code") or branch_info.get("branch_code") or "BR-ULSAN-01"
    branch_name = branch_info.get("branch_name") or branch_code
    work_zone = params.get("work_zone") or branch_info.get("default_zone") or "ZONE-A-BAY1"

    ship_code = params.get("ship_code") or block_master.get("ship_code") or "-"
    block_name = params.get("block_name") or block_master.get("block_name") or "-"
    material = params.get("material") or block_master.get("material") or "-"
    thickness = str(params.get("thickness") or block_master.get("thickness") or "-").replace("T", "")
    process_code = params.get("process") or block_master.get("process") or "-"

    worker_name = (
        erp_commit.get("worker_name")
        or selected_partner.get("worker_name")
        or params.get("worker_name")
        or "-"
    )
    worker_id = (
        erp_commit.get("worker_id")
        or selected_partner.get("assigned_worker_id")
        or selected_partner.get("ext_worker_id")
        or "-"
    )
    cert_code = selected_partner.get("cert_code") or "-"
    daily_rate = selected_partner.get("daily_rate_krw")
    unit_price = (
        selected_partner.get("unit_price_krw")
        or selected_partner.get("trip_rate_krw")
        or selected_partner.get("rate_krw")
        or daily_rate
    )
    wire_code = document_data.get("wire_item_code") or selected_partner.get("hq_item_code") or "-"
    wire_qty = document_data.get("reserve_wire_kg") or erp_commit.get("reserved_wire_kg") or "-"
    vehicle_code = document_data.get("assigned_vehicle_code") or erp_commit.get("assigned_vehicle_code") or "-"

    if biz_type == "OUTSOURCED_WORKER":
        scope_label = "외주 용접 전문 인력 파견 (OUTSOURCED_WORKER)"
        order_item_desc = f"선급 유자격 용접사 파견: {worker_name} ({worker_id} / 자격: {cert_code} / 공정: {process_code})"
        qty_spec_text = "1 명 (블록 공정 완료 시까지)"
        rate_text = f"{int(float(daily_rate)):,} 원 / 일" if daily_rate else "협력사 단가표 기준"
    elif biz_type == "HEAVY_TRANSPORT":
        scope_label = "중량물 블록 운송 및 배차 (HEAVY_TRANSPORT)"
        block_desc = f"대상 블록({block_id}, {block_master.get('weight_ton', '-')}톤)" if block_id not in ("DIRECT", "BLK", "-") else "야드 내 중량물 블록"
        order_item_desc = f"{block_desc} 이송 배차 ({vehicle_code})"
        qty_spec_text = "1 대 / 왕복 배차"
        rate_text = f"{int(float(unit_price)):,} 원 / 회" if unit_price else "협력사 단가표 기준"
    else:
        scope_label = "용접 자재 긴급 조달 (WELDING_MATERIAL)"
        item_nm = selected_partner.get("item_name") or wire_code
        order_item_desc = f"{item_nm} ({wire_code})"
        qty_spec_text = f"{wire_qty} KG"
        rate_text = f"{int(float(unit_price)):,} 원 / KG" if unit_price else "협력사 단가표 기준"

    approval_status = (
        document_data.get("approval_status")
        or document_data.get("status")
        or "DRAFT"
    )
    reviewer_label = document_data.get("reviewed_by") or "담당기안자"

    if approval_status == "APPROVED":
        approver_label = document_data.get("approved_by") or "공정관리자(HITL)"
        status_cell_html = f"<b>APPROVED</b> ({approver_label})"
        sign_line_html = f"기안: {reviewer_label} / 승인: {approver_label}"
    else:
        status_cell_html = f"<b>{approval_status}</b> (관리자 결재 대기)"
        sign_line_html = f"기안: {reviewer_label} / 승인: 결재대기"

    font_family, font_face_css = _prepare_korean_font_css()

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<style>
    {font_face_css}
    @page {{ size: A4 portrait; margin: 12mm 14mm; }}
    * {{ font-family: {font_family}; }}
    body {{ font-family: {font_family}; font-size: 9.0pt; color: #111111; line-height: 1.4; }}
    .header-box {{ border-bottom: 2pt solid #1F4E79; padding-bottom: 8px; margin-bottom: 12px; }}
    .title {{ text-align: center; font-size: 16.0pt; font-weight: bold; color: #1F4E79; letter-spacing: 2px; }}
    .subtitle {{ text-align: center; font-size: 8.5pt; color: #555555; margin-top: 3px; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 10px; }}
    th, td {{ border: 0.6pt solid #333333; padding: 6px 7px; vertical-align: middle; font-family: {font_family}; }}
    th {{ background-color: #E8EEF5; font-weight: bold; text-align: left; width: 20%; }}
    td {{ width: 30%; }}
    .sec-hdr {{ background-color: #1F4E79; color: #FFFFFF; font-weight: bold; font-size: 9.5pt; padding: 5px 8px; }}
    .center {{ text-align: center; }}
    .right {{ text-align: right; }}
    .notice-box {{ border: 0.6pt solid #888888; background-color: #F8FAFC; padding: 8px 10px; font-size: 8.3pt; margin-top: 6px; }}
</style>
</head>
<body>
    <div class="header-box">
        <div class="title">외 주 발 주 주 문 서 (PURCHASE ORDER)</div>
        <div class="subtitle">K-SMART SHIPYARD INTEGRATED SCM &amp; B2B EXECUTION SYSTEM</div>
    </div>

    <table>
        <tr>
            <th>발주번호 (PO NO)</th><td><b>{po_num}</b></td>
            <th>발주일자</th><td>{today_str}</td>
        </tr>
        <tr>
            <th>연계 작업지시번호</th><td><b>{wo_num}</b></td>
            <th>승인상태 / 결재자</th><td>{status_cell_html}</td>
        </tr>
    </table>

    <table>
        <tr><td colspan="4" class="sec-hdr">■ 1. 발주처 및 수주 협력사 정보</td></tr>
        <tr>
            <th>발주처 (Buyer)</th><td><b>K-스마트조선소 본사 (COMP-HQ)</b></td>
            <th>수주 협력사 (Vendor)</th><td><b>{partner_name} ({partner_id})</b></td>
        </tr>
        <tr>
            <th>발주 구분</th><td>{scope_label}</td>
            <th>기안 / 최종 승인</th><td>{sign_line_html}</td>
        </tr>
        <tr>
            <th>납품·투입 야드</th><td><b>{branch_name} ({branch_code})</b></td>
            <th>현장 작업 구역</th><td><b>{work_zone}</b></td>
        </tr>
    </table>

    <table>
        <tr><td colspan="4" class="sec-hdr">■ 2. 대상 선박 블록 및 시공 제원</td></tr>
        <tr>
            <th>대상 호선</th><td>{ship_code}</td>
            <th>대상 블록 코드</th><td><b>{block_id}</b> ({block_name})</td>
        </tr>
        <tr>
            <th>모재 강종 / 두께</th><td>{material} / {thickness} mm</td>
            <th>적용 용접 공정</th><td><b>{process_code}</b></td>
        </tr>
    </table>

    <table>
        <tr><td colspan="4" class="sec-hdr">■ 3. 외주 발주 상세 내역</td></tr>
        <tr>
            <th class="center" style="width:8%;">No</th>
            <th class="center" style="width:46%;">발주 품목 / 외주 역무 내역</th>
            <th class="center" style="width:23%;">수량 / 투입 규격</th>
            <th class="center" style="width:23%;">계약 단가 기준</th>
        </tr>
        <tr>
            <td class="center">1</td>
            <td><b>{order_item_desc}</b></td>
            <td class="center">{qty_spec_text}</td>
            <td class="right">{rate_text}</td>
        </tr>
    </table>

    <div class="notice-box">
        <b>[필수 이행 및 안전 준수 조건]</b><br/>
        1. 본 발주 주문서는 K-스마트조선소 관리자 최종 결재(HITL) 및 B2B SCM 원장 커밋과 동시에 공식 효력이 발생합니다.<br/>
        2. 파견 인력 및 납품 자재는 첨부된 공식 작업지시서(<b>{wo_num}</b>)의 WPS 시공 조건 및 야드 안전 인터록 기준을 반드시 준수해야 합니다.<br/>
        3. 현장 반입 시 본 발주번호(<b>{po_num}</b>)를 야드 게이트 및 자재·노무 검수 담당자에게 제시해 주시기 바랍니다.
    </div>

    <table style="margin-top: 12px;">
        <tr>
            <th>발주처 확인 (본사)</th><td>K-스마트조선소 통합생산관리본부 (직인생략)</td>
            <th>수주사 접수 확인</th><td>{partner_name} 담당자: _______________ (인)</td>
        </tr>
    </table>
</body>
</html>"""

    with open(pdf_path, "wb") as pdf_file:
        status = pisa.CreatePDF(html_content, dest=pdf_file, encoding="utf-8")

    if status.err or not pdf_path.exists():
        return json.dumps({"status": "ERROR", "reason": "PO PDF 렌더링 실패"}, ensure_ascii=False)

    return json.dumps(
        {
            "status": "SUCCESS",
            "po_number": po_num,
            "po_pdf_path": str(pdf_path.resolve()),
            "download_po_pdf_url": f"/api/files/download/{today_str}/{pdf_filename}",
        },
        ensure_ascii=False,
    )


@tool
def log_inspection_history(
    session_id: str, query: str, status: str = "PASS"
) -> str:
    """공정 판정 및 질의 이력을 감사(Audit) 로그 파일에 기록합니다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_file = OUTPUT_DIR / "audit_history.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = (
        f"[{timestamp}] [SESSION: {session_id}] [STATUS: {status}] 질의:"
        f" {query}\n"
    )

    with audit_file.open("a", encoding="utf-8") as f:
        f.write(log_line)

    return f"감사 이력 기록 성공 (세션: {session_id}, 상태: {status})"