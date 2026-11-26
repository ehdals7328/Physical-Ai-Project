# modules/agent/tools/cad_tools.py
"""
[도구 계층 - Pure Deterministic Spatial Extractor & DB/MRP Tools]
- 고정 블록명/고정 개수 하드코딩 0%
- DXF 엔티티의 실제 (X, Y) 좌표를 뷰(데크) 앵커별로 공간 클러스터링하고
  동반 PDF 도면 이미지(base64)를 렌더링해 Team 4 AI 블록 분할기에 전달한다.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from decimal import Decimal

from sqlalchemy import text

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore

from modules.knowledge.cad_parser import (
    detect_vessel_context,
    parse_dxf_metadata,
    render_dxf_to_svg,
)

SHIP_STRUCTURAL_KEYWORDS = {
    "DECK", "DOUBLE BOTTOM", "SINGLE BOTTOM", "TANKTOP", "FRAME", "GIRDER",
    "BULKHEAD", "LONGITUDINAL", "TRANSVERSE", "SECTION", "HP", "BL",
}
STRUCTURAL_LAYER_PREFIXES = ("S0CO", "S0DA")
DEFAULT_SVG_CACHE_DIR = "data/rag_cache/cad_svg"

def _to_plain_python(obj: Any) -> Any:
    """numpy 스칼라 및 Oracle Decimal을 msgpack 직렬화 가능한 순수 파이썬 타입으로 변환"""
    if isinstance(obj, dict):
        return {str(k): _to_plain_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_plain_python(v) for v in obj]
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    if hasattr(obj, "item") and callable(obj.item):
        return obj.item()
    return obj

def _truncate_utf8(val: Any, max_bytes: int) -> str:
    """Oracle VARCHAR2(N BYTE) 초과 방지를 위해 UTF-8 바이트 기준으로 안전 절단"""
    s = str(val or "").strip()
    encoded = s.encode("utf-8")
    if len(encoded) <= max_bytes:
        return s
    return encoded[:max_bytes].decode("utf-8", errors="ignore").strip()

def compute_file_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean_dxf_text(raw: str) -> str:
    if not raw:
        return ""
    t = raw.replace("%%U", "").replace("%%u", "").replace("\\P", " ").replace("\\~", " ")
    t = re.sub(r"\\[A-Za-z][^;]*;", "", t)
    t = t.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", t).strip()


def _extract_pdf_info_and_image(pdf_path: Optional[Path]) -> Dict[str, Any]:
    """PDF 표제란 텍스트/축척 추출 + Vision LLM 판독용 1페이지 PNG base64 렌더링"""
    if not pdf_path or not pdf_path.exists():
        return {"raw_text": "", "scale": 50.0, "image_b64": None}

    doc = fitz.open(str(pdf_path))
    raw_text = "\n".join(page.get_text("text") for page in doc)
    scale_match = re.search(r"1\s*:\s*(\d+)", raw_text)
    scale = float(scale_match.group(1)) if scale_match else 50.0

    image_b64: Optional[str] = None
    try:
        if len(doc) > 0:
            pix = doc[0].get_pixmap(dpi=150, alpha=False)
            image_b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
    except Exception:
        image_b64 = None

    doc.close()
    return {"raw_text": raw_text, "scale": scale, "image_b64": image_b64}


def _extract_spatial_drawing_views(dxf_path: Path) -> Dict[str, Any]:
    """
    DXF Modelspace의 모든 TEXT/MTEXT/DIMENSION과 구조선(S0CO*/S0DA*)을
    실제 (X, Y) 좌표 기준으로 분석하여 도면 내 뷰(데크별 구역) 클러스터를 자동 구성한다.
    """
    import ezdxf  # type: ignore

    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    text_nodes: List[Dict[str, Any]] = []
    view_anchors: List[Dict[str, Any]] = []
    struct_segments: List[tuple[float, float, float]] = []  # (mid_x, mid_y, length)
    excluded_non_steel: List[str] = []

    view_title_re = re.compile(
        r"\b(TANKTOP|DOUBLE\s+BOTTOM|SINGLE\s+BOTTOM|1-DECK|2-DECK|A-DECK|MAIN\s+DECK|WHEELHOUSE|SUPERSTRUCTURE|FOREBODY|PROFILE|SECTION)\b",
        re.IGNORECASE,
    )

    for ent in msp:
        etype = ent.dxftype()
        layer = str(getattr(ent.dxf, "layer", "0") or "0")

        if etype in ("TEXT", "MTEXT"):
            raw_val = (
                ent.plain_text()
                if (etype == "MTEXT" and hasattr(ent, "plain_text"))
                else str(getattr(ent.dxf, "text", "") or "")
            )
            cleaned = _clean_dxf_text(raw_val)
            if not cleaned:
                continue
            ins = getattr(ent.dxf, "insert", (0.0, 0.0))
            x, y = float(ins[0]), float(ins[1])

            if "AL." in cleaned.upper() or "ALUMIN" in cleaned.upper():
                excluded_non_steel.append(cleaned)
                continue

            node = {"x": round(x, 1), "y": round(y, 1), "layer": layer, "text": cleaned}
            text_nodes.append(node)

            # 밑줄(%%U) 제목 또는 데크/구역 대표 타이틀을 뷰 앵커로 자동 등록
            if ("%%U" in raw_val.upper() or layer == "_-0") and view_title_re.search(cleaned):
                view_anchors.append({
                    "view_id": f"VIEW-{len(view_anchors) + 1:02d}",
                    "view_title": cleaned,
                    "anchor_x": round(x, 1),
                    "anchor_y": round(y, 1),
                    "frames": [],
                    "member_specs": [],
                    "dimensions_mm": [],
                    "notes": [],
                    "structural_line_length_raw": 0.0,
                })

        elif etype == "DIMENSION":
            dim_txt = _clean_dxf_text(str(getattr(ent.dxf, "text", "") or ""))
            meas_val: Optional[float] = None
            try:
                m = ent.get_measurement()
                if isinstance(m, (int, float)) and math.isfinite(m) and m > 0:
                    meas_val = round(float(m), 1)
            except Exception:
                pass
            dp = getattr(ent.dxf, "defpoint", (0.0, 0.0))
            x, y = float(dp[0]), float(dp[1])
            label = f"{dim_txt} ({meas_val}mm)" if (dim_txt and dim_txt != "<>" and meas_val) else (
                f"{meas_val}mm" if meas_val else dim_txt
            )
            if not label:
                continue
            if "AL." in label.upper() or "ALUMIN" in label.upper():
                excluded_non_steel.append(label)
                continue
            text_nodes.append({
                "x": round(x, 1),
                "y": round(y, 1),
                "layer": layer,
                "text": label,
                "meas_mm": meas_val,
            })

        elif layer.upper().startswith(STRUCTURAL_LAYER_PREFIXES):
            if etype == "LINE":
                p1, p2 = ent.dxf.start, ent.dxf.end
                seg_len = float(p1.distance(p2))
                if seg_len > 0:
                    struct_segments.append(((float(p1.x) + float(p2.x)) / 2.0, (float(p1.y) + float(p2.y)) / 2.0, seg_len))
            elif etype == "LWPOLYLINE":
                pts = list(ent.get_points("xy"))
                if len(pts) >= 2:
                    mx = sum(p[0] for p in pts) / len(pts)
                    my = sum(p[1] for p in pts) / len(pts)
                    seg_len = sum(
                        ((pts[i + 1][0] - pts[i][0]) ** 2 + (pts[i + 1][1] - pts[i][1]) ** 2) ** 0.5
                        for i in range(len(pts) - 1)
                    )
                    if seg_len > 0:
                        struct_segments.append((mx, my, seg_len))

    # 뷰 타이틀 앵커가 명시적으로 잡히지 않은 도면일 경우 Y좌표 15,000 단위로 자동 클러스터 생성
    if not view_anchors and text_nodes:
        buckets: Dict[int, List[Dict[str, Any]]] = {}
        for nd in text_nodes:
            b_key = int(nd["y"] // 15000)
            buckets.setdefault(b_key, []).append(nd)
        for idx, (b_key, items) in enumerate(sorted(buckets.items(), reverse=True), start=1):
            avg_x = round(sum(i["x"] for i in items) / len(items), 1)
            avg_y = round(sum(i["y"] for i in items) / len(items), 1)
            view_anchors.append({
                "view_id": f"VIEW-{idx:02d}",
                "view_title": f"구조 구역 클러스터 #{idx} (Y≈{int(avg_y)})",
                "anchor_x": avg_x,
                "anchor_y": avg_y,
                "frames": [],
                "member_specs": [],
                "dimensions_mm": [],
                "notes": [],
                "structural_line_length_raw": 0.0,
            })

    prof_re = re.compile(r"\b(HP|W|FL|PL)\s*:?\s*(\d{2,3})\s*[xX×]\s*(\d{1,2}(?:\.\d+)?)\b")

    # 각 텍스트·프레임 번호·부재 규격·치수를 가장 가까운 뷰 앵커에 공간 매핑
    for nd in text_nodes:
        if not view_anchors:
            break
        # 평면도가 수직(Y축)으로 쌓이는 특성을 반영해 Y거리 가중치(2.5x) 적용
        nearest = min(
            view_anchors,
            key=lambda v: ((nd["x"] - v["anchor_x"]) ** 2 + ((nd["y"] - v["anchor_y"]) * 2.5) ** 2),
        )
        txt = nd["text"]
        if txt == nearest["view_title"]:
            continue

        # 프레임 번호 (0 ~ 120 사이 단독 정수)
        if re.fullmatch(r"\d{1,3}", txt) and 0 <= int(txt) <= 120:
            fr_num = int(txt)
            if fr_num not in nearest["frames"]:
                nearest["frames"].append(fr_num)
            continue

        # 형강/판재 부재 규격 추출
        found_profs = [f"{m.group(1)}{m.group(2)}x{m.group(3)}" for m in prof_re.finditer(txt)]
        if found_profs:
            for fp in found_profs:
                if fp not in nearest["member_specs"]:
                    nearest["member_specs"].append(fp)

        if nd.get("meas_mm"):
            nearest["dimensions_mm"].append({"label": txt, "mm": nd["meas_mm"]})
        elif len(txt) >= 3:
            if txt not in nearest["notes"]:
                nearest["notes"].append(txt)

    # 구조 레이어(S0CO*/S0DA*) 선분 길이를 각 뷰에 합산
    for mx, my, slen in struct_segments:
        if not view_anchors:
            break
        nearest = min(
            view_anchors,
            key=lambda v: ((mx - v["anchor_x"]) ** 2 + ((my - v["anchor_y"]) * 2.5) ** 2),
        )
        nearest["structural_line_length_raw"] = round(nearest["structural_line_length_raw"] + slen, 1)

    for v in view_anchors:
        v["frames"].sort()
        v["frame_range"] = (
            f"Fr.{v['frames'][0]}~{v['frames'][-1]}" if len(v["frames"]) >= 2 else (
                f"Fr.{v['frames'][0]}" if v["frames"] else "미표기"
            )
        )
        v["notes"] = v["notes"][:25]
        v["dimensions_mm"] = v["dimensions_mm"][:20]

    # 알루미늄 전용 상부구조 뷰(예: WHEELHOUSE TOP, SUPERSTRUCT. TOP)는 강재 용접 대상에서 분리
    non_steel_keywords = ("WHEELHOUSE", "SUPERSTRUCT", "BRIDGE", "ALUMIN")
    steel_views = [
        v for v in view_anchors
        if not any(k in v["view_title"].upper() for k in non_steel_keywords)
    ]
    non_steel_views = [
        v["view_title"] for v in view_anchors
        if any(k in v["view_title"].upper() for k in non_steel_keywords)
    ]

    return {
        "steel_views": steel_views,
        "excluded_non_steel_views": non_steel_views,
        "excluded_non_steel_notes": list(dict.fromkeys(excluded_non_steel)),
        "total_text_count": len(text_nodes),
    }


# ==============================================================================
# [Tool 1] CAD 도면 Gate 1~2 검증 및 공간 구조 데이터 추출 (하드코딩 0%)
# ==============================================================================
def extract_cad_spatial_facts(
    dxf_path_str: str,
    pdf_path_str: Optional[str] = None,
    branch_code: str = "BR-ULSAN-01",
    work_zone: str = "ZONE-A-BAY1",
    ship_code: str = "HN-GUNNERUS-01",
    user_engine: Any = None,
    svg_cache_dir: str = DEFAULT_SVG_CACHE_DIR,
    is_reanalyze: bool = False,
) -> Dict[str, Any]:
    dxf_path = Path(dxf_path_str)
    pdf_path = Path(pdf_path_str) if pdf_path_str else None

    if dxf_path.suffix.lower() != ".dxf":
        return {
            "status": "REJECTED",
            "gate": "GATE_1_FORMAT",
            "reason": f"지원하지 않는 포맷({dxf_path.suffix})입니다. 2D 선체 구조 도면(.dxf)을 선택해 주세요.",
        }
    if not dxf_path.exists():
        return {"status": "REJECTED", "gate": "GATE_1_NOT_FOUND", "reason": f"파일을 찾을 수 없습니다: {dxf_path}"}

    file_hash = compute_file_sha256(dxf_path)
    has_active_work_in_zone = False
    current_rev_no = 0

    if user_engine is not None:
        with user_engine.connect() as conn:
            if not is_reanalyze:
                dup_cnt = conn.execute(
                    text("""
                        SELECT COUNT(*)
                        FROM CAD_DRAWINGS d
                        WHERE d.BRANCH_CODE = :b_code
                          AND d.WORK_ZONE = :w_zone
                          AND d.FILE_SHA256 = :f_hash
                          AND d.PLAN_STATUS = 'ACTIVE'
                          AND EXISTS (
                              SELECT 1 FROM WORK_PLANS wp
                              WHERE wp.DRAWING_ID = d.DRAWING_ID
                                AND wp.PLAN_STATE IN ('IN_PROGRESS', 'PENDING')
                          )
                    """),
                    {"b_code": branch_code, "w_zone": work_zone, "f_hash": file_hash},
                ).scalar() or 0
                if dup_cnt > 0:
                    return {
                        "status": "REJECTED",
                        "gate": "GATE_1_DUPLICATE",
                        "reason": f"현재 {branch_code} ({work_zone})에 동일한 도면이 이미 ACTIVE 상태로 운영 중입니다.",
                    }

            active_cnt = conn.execute(
                text("""
                    SELECT COUNT(*) FROM WORK_PLANS
                    WHERE BRANCH_CODE = :b_code AND WORK_ZONE = :w_zone
                      AND PLAN_STATE IN ('IN_PROGRESS', 'PENDING')
                """),
                {"b_code": branch_code, "w_zone": work_zone},
            ).scalar() or 0
            has_active_work_in_zone = active_cnt > 0

            max_rev = conn.execute(
                text("""
                    SELECT NVL(MAX(REVISION_NO), -1) FROM CAD_DRAWINGS
                    WHERE BRANCH_CODE = :b_code AND WORK_ZONE = :w_zone
                """),
                {"b_code": branch_code, "w_zone": work_zone},
            ).scalar()
            current_rev_no = int(max_rev) + 1 if max_rev is not None else 0

    meta = parse_dxf_metadata(dxf_path)
    vessel_name, vessel_desc = detect_vessel_context(dxf_path)
    pdf_info = _extract_pdf_info_and_image(pdf_path)

    combined_upper = (
        " ".join((meta.get("texts") or []) + (meta.get("dimensions") or []))
        + " " + pdf_info["raw_text"]
        + " " + dxf_path.stem
    ).upper()
    matched_keywords = [kw for kw in SHIP_STRUCTURAL_KEYWORDS if kw in combined_upper]

    if len(matched_keywords) < 2:
        return {
            "status": "REJECTED",
            "gate": "GATE_1_NON_STRUCTURAL",
            "reason": "용접 부재·선체 구조 정보가 없는 도면입니다. 선체 구조 도면(Construction Plan)을 업로드해 주세요.",
        }

    spatial_data = _extract_spatial_drawing_views(dxf_path)
    if not spatial_data["steel_views"]:
        return {
            "status": "REJECTED",
            "gate": "GATE_1_NO_STEEL_VIEWS",
            "reason": "도면 내 탄소강 선체 구조 뷰를 식별할 수 없습니다.",
        }

    # 기존 data/rag_cache/cad_svg 캐시 재사용
    preview_svg = render_dxf_to_svg(dxf_path, cache_dir=Path(svg_cache_dir) if svg_cache_dir else None)

    return _to_plain_python({
        "status": "EXTRACTED",
        "gate2_status": "COLLISION_WARNING" if has_active_work_in_zone else "READY_TO_CONFIRM",
        "gate2_options": ["QUEUED", "REVISION"] if has_active_work_in_zone else ["ACTIVE"],
        "drawing_meta": {
            "ship_code": ship_code,
            "drawing_title": f"{vessel_name} - {dxf_path.stem}",
            "dxf_file_path": str(dxf_path),
            "pdf_file_path": str(pdf_path) if pdf_path else "",
            "file_sha256": file_hash,
            "revision_no": current_rev_no,
            "branch_code": branch_code,
            "work_zone": work_zone,
            "drawing_scale": pdf_info["scale"],
            "vessel_name": vessel_name,
            "vessel_desc": vessel_desc,
            "layers": meta.get("layers") or [],
            "excluded_non_steel_views": spatial_data["excluded_non_steel_views"],
            "excluded_non_steel_notes": spatial_data["excluded_non_steel_notes"],
        },
        "spatial_views": spatial_data["steel_views"],
        "drawing_image_b64": pdf_info["image_b64"],
        "preview_svg": preview_svg,
    })


# ==============================================================================
# [Tool 2 & 3] 계획 확정(DB 커밋) + 단기(P1~P2) MRP 선제 PO 기안 & 일일 브리핑
# ==============================================================================
def confirm_cad_work_plan(
    drawing_meta: Dict[str, Any],
    edited_packages: List[Dict[str, Any]],
    confirm_mode: str = "ACTIVE",
    uploaded_by: str = "admin",
    user_engine: Any = None,
    erp_engine: Any = None,
) -> Dict[str, Any]:
    if user_engine is None:
        raise ValueError("user_engine이 필요합니다.")

    if not isinstance(confirm_mode, str) or not confirm_mode.strip():
        raise ValueError("confirm_mode는 비어 있을 수 없습니다. (허용값: ACTIVE, QUEUED, REVISION)")

    mode = confirm_mode.strip().upper()
    if mode not in ("ACTIVE", "QUEUED", "REVISION"):
        raise ValueError(
            f"허용되지 않는 confirm_mode입니다: '{confirm_mode}' (허용값: ACTIVE, QUEUED, REVISION)"
        )

    drawing_id = f"DWG-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    branch_code = drawing_meta.get("branch_code", "BR-ULSAN-01")
    work_zone = drawing_meta.get("work_zone", "ZONE-A-BAY1")
    dwg_status = "QUEUED" if mode == "QUEUED" else "ACTIVE"
    plan_initial_state = "QUEUED" if mode == "QUEUED" else "PENDING"

    created_plans: List[Dict[str, Any]] = []

    with user_engine.begin() as conn:
        if mode == "REVISION":
            conn.execute(
                text("""
                    UPDATE CAD_DRAWINGS SET PLAN_STATUS = 'SUPERSEDED'
                    WHERE BRANCH_CODE = :b_code AND WORK_ZONE = :w_zone AND PLAN_STATUS = 'ACTIVE'
                """),
                {"b_code": branch_code, "w_zone": work_zone},
            )
            conn.execute(
                text("""
                    UPDATE WORK_PLANS SET PLAN_STATE = 'LOCKED'
                    WHERE BRANCH_CODE = :b_code AND WORK_ZONE = :w_zone AND PLAN_STATE = 'PENDING'
                """),
                {"b_code": branch_code, "w_zone": work_zone},
            )

        file_sha = drawing_meta.get("file_sha256")
        existing_dwg_id = None
        if file_sha:
            existing_dwg_id = conn.execute(
                text("""
                    SELECT DRAWING_ID FROM CAD_DRAWINGS
                    WHERE BRANCH_CODE = :b_code
                      AND WORK_ZONE = :w_zone
                      AND FILE_SHA256 = :sha256
                    FETCH FIRST 1 ROWS ONLY
                """),
                {"b_code": branch_code, "w_zone": work_zone, "sha256": file_sha},
            ).scalar()

        if existing_dwg_id:
            drawing_id = str(existing_dwg_id)
            conn.execute(
                text("""
                    UPDATE CAD_DRAWINGS
                    SET PLAN_STATUS = :status,
                        REVISION_NO = :rev_no,
                        DRAWING_SCALE = :scale
                    WHERE DRAWING_ID = :drawing_id
                """),
                {
                    "status": dwg_status,
                    "rev_no": int(drawing_meta.get("revision_no", 0)),
                    "scale": float(drawing_meta.get("drawing_scale", 50.0)),
                    "drawing_id": drawing_id,
                },
            )
        else:
            conn.execute(
                text("""
                    INSERT INTO CAD_DRAWINGS (
                        DRAWING_ID, SHIP_CODE, DRAWING_TITLE, DXF_FILE_PATH, PDF_FILE_PATH,
                        FILE_SHA256, REVISION_NO, BRANCH_CODE, WORK_ZONE, DRAWING_SCALE,
                        PLAN_STATUS, UPLOADED_BY
                    ) VALUES (
                        :drawing_id, :ship_code, :title, :dxf_path, :pdf_path,
                        :sha256, :rev_no, :b_code, :w_zone, :scale,
                        :status, :uploaded_by
                    )
                """),
                {
                    "drawing_id": drawing_id,
                    "ship_code": drawing_meta.get("ship_code", "HN-GUNNERUS-01"),
                    "title": drawing_meta.get("drawing_title", "CAD Structure Plan"),
                    "dxf_path": drawing_meta.get("dxf_file_path", ""),
                    "pdf_path": drawing_meta.get("pdf_file_path", ""),
                    "sha256": file_sha or uuid.uuid4().hex,
                    "rev_no": int(drawing_meta.get("revision_no", 0)),
                    "b_code": branch_code,
                    "w_zone": work_zone,
                    "scale": float(drawing_meta.get("drawing_scale", 50.0)),
                    "status": dwg_status,
                    "uploaded_by": uploaded_by,
                },
            )

        for idx, pkg in enumerate(edited_packages, start=1):
            plan_id = f"PLAN-{datetime.now().strftime('%m%d')}-{uuid.uuid4().hex[:5].upper()}"
            rank = int(pkg.get("priority_rank", idx))
            proc_code = str(pkg.get("process_code", "FCAW")).upper()
            wire_code = pkg.get("wire_mat_code") or (
                "WIRE-SAW-L60" if proc_code == "SAW" else "WIRE-CSF71T"
            )

            row_data = {
                "plan_id": plan_id,
                "drawing_id": drawing_id,
                "b_code": branch_code,
                "w_zone": work_zone,
                "rank": rank,
                "block_code": _truncate_utf8(pkg["block_code"], 45),
                "block_name": _truncate_utf8(pkg["block_name"], 190),
                "deck_level": _truncate_utf8(pkg.get("deck_level", "DECK AND DOUBLE BOTTOM"), 95),
                "mat_grade": _truncate_utf8(pkg.get("material_grade", "AH36"), 25),
                "thick_mm": float(pkg.get("thickness_mm", 10.0)),
                "proc_code": _truncate_utf8(proc_code, 25),
                "joint_type": _truncate_utf8(str(pkg.get("joint_type", "BUTT")).upper(), 25),
                "wire_code": _truncate_utf8(wire_code, 45),
                "weld_len": float(pkg.get("est_weld_length_mm", 0.0)),
                "weight_ton": float(pkg.get("est_weight_ton") or 0.0),
                "wire_kg": float(pkg.get("est_wire_kg", 0.0)),
                "days": max(1, int(pkg.get("est_duration_days", 1))),
                "start_dt": _truncate_utf8(pkg.get("planned_start_date", datetime.now().strftime("%Y-%m-%d")), 20),
                "end_dt": _truncate_utf8(pkg.get("planned_end_date", datetime.now().strftime("%Y-%m-%d")), 20),
                "rationale": _truncate_utf8(pkg.get("schedule_rationale") or "", 480),
                "state": plan_initial_state,
            }
            conn.execute(
                text("""
                    INSERT INTO WORK_PLANS (
                        PLAN_ID, DRAWING_ID, BRANCH_CODE, WORK_ZONE, PRIORITY_RANK,
                        BLOCK_CODE, BLOCK_NAME, DECK_LEVEL, MATERIAL_GRADE, THICKNESS_MM,
                        PROCESS_CODE, JOINT_TYPE, WIRE_MAT_CODE, EST_WELD_LENGTH_MM,
                        EST_WEIGHT_TON, EST_WIRE_KG, EST_DURATION_DAYS,
                        PLANNED_START_DATE, PLANNED_END_DATE, PLAN_RATIONALE,
                        PROGRESS_PCT, PLAN_STATE
                    ) VALUES (
                        :plan_id, :drawing_id, :b_code, :w_zone, :rank,
                        :block_code, :block_name, :deck_level, :mat_grade, :thick_mm,
                        :proc_code, :joint_type, :wire_code, :weld_len,
                        :weight_ton, :wire_kg, :days,
                        :start_dt, :end_dt, :rationale,
                        0, :state
                    )
                """),
                row_data,
            )
            created_plans.append(row_data)

    proactive_po_drafts: List[Dict[str, Any]] = []
    mrp_diagnostics: List[Dict[str, Any]] = []

    if mode != "QUEUED" and erp_engine is not None:
        short_term_demand: Dict[str, float] = {}
        for p in created_plans:
            # 전체 블록(또는 단기 착수 구간)의 자재 소요량을 합산해 부족 품목 자동 감지
            w_code = p["wire_code"]
            short_term_demand[w_code] = short_term_demand.get(w_code, 0.0) + p["wire_kg"]

        safety_buffer_kg = 50.0
        with erp_engine.connect() as erp_conn:
            for w_code, req_kg in short_term_demand.items():
                inv_row = erp_conn.execute(
                    text("""
                        SELECT (TOTAL_QTY - RESERVED_QTY) AS AVAILABLE_QTY
                        FROM ERP_BOM_INVENTORY
                        WHERE BRANCH_CODE = :b_code AND ITEM_CODE = :i_code
                    """),
                    {"b_code": branch_code, "i_code": w_code},
                ).mappings().first()

                avail_qty = float(inv_row["available_qty"]) if inv_row else 0.0
                target_need = round(req_kg + safety_buffer_kg, 2)
                is_shortage = avail_qty < target_need

                po_draft = None
                if is_shortage:
                    shortage_kg = round(max(target_need - avail_qty, req_kg), 1)
                    po_draft = _create_proactive_po_draft(
                        user_engine=user_engine,
                        branch_code=branch_code,
                        drawing_id=drawing_id,
                        wire_mat_code=w_code,
                        shortage_kg=shortage_kg,
                        current_avail_kg=avail_qty,
                        requested_by=uploaded_by,
                    )
                    proactive_po_drafts.append(po_draft)

                mrp_status = "SUFFICIENT"
                if is_shortage:
                    mrp_status = (
                        "SHORTAGE_PO_DRAFTED"
                        if (po_draft and po_draft.get("quote_status") == "QUOTED")
                        else "SHORTAGE_QUOTE_REQUIRED"
                    )

                mrp_diagnostics.append({
                    "item_code": w_code,
                    "short_term_req_kg": round(req_kg, 2),
                    "safety_buffer_kg": safety_buffer_kg,
                    "current_avail_kg": avail_qty,
                    "status": mrp_status,
                })

    return {
        "status": "CONFIRMED",
        "drawing_id": drawing_id,
        "confirm_mode": mode,
        "created_plans": created_plans,
        "mrp_diagnostics": mrp_diagnostics,
        "proactive_po_drafts": proactive_po_drafts,
    }


def _create_proactive_po_draft(
    user_engine: Any,
    branch_code: str,
    drawing_id: str,
    wire_mat_code: str,
    shortage_kg: float,
    current_avail_kg: float,
    requested_by: str,
) -> Dict[str, Any]:
    partner_id: Optional[str] = None
    partner_name: Optional[str] = None
    unit_price: Optional[int] = None
    lead_time_hours: Optional[int] = None
    lookup_error: Optional[str] = None

    try:
        from modules.agent.tools.erp_tools import query_partner_erp_b2b
        b2b_res = query_partner_erp_b2b.invoke({
            "outsourcing_domain": "WELDING_MATERIAL",
            "material": wire_mat_code,
        })
        if isinstance(b2b_res, dict) and b2b_res.get("status") == "SUCCESS":
            sel = b2b_res.get("selected_partner") or {}
            partner_id = sel.get("partner_id") or None
            partner_name = sel.get("partner_name") or partner_id
            raw_price = sel.get("unit_price_krw")
            unit_price = int(raw_price) if raw_price is not None and float(raw_price) > 0 else None
            raw_lt = sel.get("lead_time_hours")
            lead_time_hours = int(raw_lt) if raw_lt is not None and int(raw_lt) > 0 else 4
        else:
            lookup_error = (
                b2b_res.get("error_message")
                if isinstance(b2b_res, dict)
                else "협력사 B2B 응답 형식 오류"
            ) or "적격 협력사 견적 조회 실패"
    except Exception as e:
        lookup_error = f"협력사 B2B 조회 예외: {e}"

    po_draft_no = f"PO-MRP-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"

    # 협력사 조회 실패 또는 단가 미확정 시 가짜 기본 업체/단가로 견적서 발행 금지
    if not partner_id or unit_price is None:
        return {
            "is_standalone_po": True,
            "source": "CAD_MRP_PROACTIVE",
            "quote_status": "PARTNER_LOOKUP_FAILED",
            "approval_status": "QUOTE_REQUIRED",
            "drawing_id": drawing_id,
            "po_number": po_draft_no,
            "branch_code": branch_code,
            "partner_id": None,
            "partner_name": None,
            "item_code": wire_mat_code,
            "order_qty_kg": shortage_kg,
            "current_hq_avail_kg": current_avail_kg,
            "unit_price_krw": None,
            "total_amount_krw": None,
            "required_on_site_at": None,
            "error": lookup_error or "협력사 단가/견적 정보 미확보로 수동 견적 확인 필요",
            "db_saved": False,
            "files": {},
        }

    ros_deadline = (datetime.now() + timedelta(hours=max(lead_time_hours or 4, 2))).strftime("%Y-%m-%d %H:00")
    total_amount = int(shortage_kg * unit_price)

    po_payload: Dict[str, Any] = {
        "is_standalone_po": True,
        "source": "CAD_MRP_PROACTIVE",
        "quote_status": "QUOTED",
        "drawing_id": drawing_id,
        "po_number": po_draft_no,
        "branch_code": branch_code,
        "partner_id": partner_id,
        "partner_name": partner_name,
        "item_code": wire_mat_code,
        "order_qty_kg": shortage_kg,
        "current_hq_avail_kg": current_avail_kg,
        "unit_price_krw": unit_price,
        "total_amount_krw": total_amount,
        "required_on_site_at": ros_deadline,
        "approval_status": "DRAFT",
        "db_saved": False,
    }
    try:
        from modules.agent.tools.document_tools import export_outsourcing_po_pdf

        po_doc_for_pdf = {
            "is_standalone_po": True,
            "work_order_number": po_draft_no,
            "title": f"[DRAFT] {branch_code} CAD 선제 자재 발주서 ({partner_name})",
            "status": "DRAFT",
            "approval_status": "DRAFT",
            "execution_type": "OUTSOURCED_B2B",
            "business_type": "WELDING_MATERIAL",
            "wire_item_code": wire_mat_code,
            "reserve_wire_kg": shortage_kg,
            "parameters": {
                "block_id": drawing_id,
                "branch_code": branch_code,
                "material": wire_mat_code,
                "quantity": shortage_kg,
            },
            "partner_erp_plan": {
                "selected_partner": {
                    "partner_id": partner_id,
                    "partner_name": partner_name,
                    "business_type": "WELDING_MATERIAL",
                    "hq_item_code": wire_mat_code,
                    "unit_price_krw": unit_price,
                    "lead_time_hours": lead_time_hours,
                }
            },
        }
        po_export_raw = export_outsourcing_po_pdf.invoke({
            "document_data": po_doc_for_pdf,
            "file_prefix": f"po_order_{drawing_id}",
        })
        po_meta = (
            json.loads(po_export_raw)
            if isinstance(po_export_raw, str)
            else (po_export_raw or {})
        )
        if isinstance(po_meta, dict):
            if not po_meta.get("download_pdf_url") and po_meta.get("download_po_pdf_url"):
                po_meta["download_pdf_url"] = po_meta["download_po_pdf_url"]
            if not po_meta.get("pdf_path") and po_meta.get("po_pdf_path"):
                po_meta["pdf_path"] = po_meta["po_pdf_path"]
        po_payload["files"] = po_meta
        po_payload["pdf_path"] = po_meta.get("pdf_path")
        po_payload["download_pdf_url"] = po_meta.get("download_pdf_url")
    except Exception as po_pdf_err:
        print(f"⚠️ [CAD 선제 PO PDF 생성 경고]: {po_pdf_err}")

    try:
        content_md = (
            f"### [CAD 공정계획 연동 선제 분할 발주서 초안] {po_draft_no}\n"
            f"- **도면 ID:** `{drawing_id}` | **대상 야드:** `{branch_code}`\n"
            f"- **발주 품목:** `{wire_mat_code}` **{shortage_kg} kg** (현재 본사 가용: {current_avail_kg} kg)\n"
            f"- **선정 협력사:** {partner_name} (`{partner_id}`) | **예상 금액:** {total_amount:,}원\n"
            f"- **현장 필요일(ROS) 역산 입고 목표:** {ros_deadline}"
        )
        with user_engine.begin() as conn:
            # 1) 기안자 USERNAME으로 USERS.USER_ID 조회 (없으면 기본 관리자 ID 1 사용)
            req_user_id = conn.execute(
                text("SELECT USER_ID FROM USERS WHERE USERNAME = :uname FETCH FIRST 1 ROWS ONLY"),
                {"uname": str(requested_by or "admin").strip()},
            ).scalar() or 1

            # 2) 부모 테이블 WORK_ORDERS 먼저 적재
            conn.execute(
                text("""
                    INSERT INTO WORK_ORDERS (
                        WORK_NO, WORK_TITLE, WORK_DESCRIPTION, SHIP_CODE, BLOCK_CODE,
                        WORK_REGION, WORK_ZONE, PROCESS_CODE, MATERIAL_GRADE,
                        THICKNESS_MM, WELD_POSITION, STATUS_CODE, REQUESTED_BY
                    ) VALUES (
                        :po_no, :title, :desc_txt, :ship_code, :block_code,
                        :branch_code, :work_zone, :proc_code, :mat_code,
                        0, 'NA', 'READY', :req_by
                    )
                """),
                {
                    "po_no": po_draft_no,
                    "title": f"[DRAFT] {branch_code} CAD 선제 자재 발주서 ({partner_name})",
                    "desc_txt": f"CAD MRP 선제 발주 초안 (기안: {requested_by})",
                    "ship_code": drawing_id[:30],
                    "block_code": wire_mat_code,
                    "branch_code": branch_code,
                    "work_zone": "DOCK-1" if "GEOJE" in branch_code.upper() else "ZONE-A-BAY1",
                    "proc_code": "SAW" if "SAW" in wire_mat_code.upper() else "FCAW",
                    "mat_code": wire_mat_code,
                    "req_by": req_user_id,
                },
            )

            # 3) 발급된 WORK_ID를 받아 자식 테이블 WORK_INSTRUCTIONS에 초안 적재
            new_work_id = conn.execute(
                text("SELECT WORK_ID FROM WORK_ORDERS WHERE WORK_NO = :po_no FETCH FIRST 1 ROWS ONLY"),
                {"po_no": po_draft_no},
            ).scalar()

            if new_work_id is not None:
                conn.execute(
                    text("""
                        INSERT INTO WORK_INSTRUCTIONS (
                            WORK_ID, REVISION_NO, INSTRUCTION_TEXT, WELDING_PARAMS_JSON,
                            SOURCE_REFS_JSON, EXTERNAL_ENV_JSON, APPROVAL_STATUS
                        ) VALUES (
                            :work_id, 'Rev.1', :content_md, :params_json,
                            '[]', :env_json, 'DRAFT'
                        )
                    """),
                    {
                        "work_id": new_work_id,
                        "content_md": content_md,
                        "params_json": json.dumps(po_payload, ensure_ascii=False),
                        "env_json": json.dumps(
                            {"files": po_payload.get("files") or {}, "_draft_payload": po_payload},
                            ensure_ascii=False,
                        ),
                    },
                )
                po_payload["work_id"] = new_work_id
                po_payload["db_saved"] = True
    except Exception as po_db_err:
        print(f"⚠️ [CAD 선제 PO DB 저장 경고]: {po_db_err}")
        po_payload["error"] = f"DB 저장 오류: {po_db_err}"

    return po_payload


def get_daily_work_briefing(user_engine: Any, branch_code: str = "BR-ULSAN-01") -> Dict[str, Any]:
    with user_engine.connect() as conn:
        carry_over = conn.execute(
            text("""
                SELECT PLAN_ID, DRAWING_ID, WORK_ZONE, PRIORITY_RANK, BLOCK_CODE, BLOCK_NAME,
                       PROCESS_CODE, MATERIAL_GRADE, THICKNESS_MM, EST_DURATION_DAYS,
                       PLANNED_START_DATE, PLANNED_END_DATE, PROGRESS_PCT, ASSIGNED_WO_NUMBER
                FROM WORK_PLANS
                WHERE BRANCH_CODE = :b_code AND PLAN_STATE = 'IN_PROGRESS'
                ORDER BY PRIORITY_RANK ASC, CREATED_AT ASC
            """),
            {"b_code": branch_code},
        ).mappings().all()

        pending_queue = conn.execute(
            text("""
                SELECT PLAN_ID, DRAWING_ID, WORK_ZONE, PRIORITY_RANK, BLOCK_CODE, BLOCK_NAME,
                       PROCESS_CODE, MATERIAL_GRADE, THICKNESS_MM, EST_WELD_LENGTH_MM,
                       EST_WEIGHT_TON, EST_WIRE_KG, EST_DURATION_DAYS,
                       PLANNED_START_DATE, PLANNED_END_DATE, PLAN_RATIONALE
                FROM WORK_PLANS
                WHERE BRANCH_CODE = :b_code AND PLAN_STATE = 'PENDING'
                ORDER BY PRIORITY_RANK ASC, CREATED_AT ASC
            """),
            {"b_code": branch_code},
        ).mappings().all()

    return _to_plain_python({
        "branch_code": branch_code,
        "carry_over_tasks": [dict(r) for r in carry_over],
        "pending_tasks": [dict(r) for r in pending_queue],
        "next_p1_task": dict(pending_queue[0]) if pending_queue else None,
    })