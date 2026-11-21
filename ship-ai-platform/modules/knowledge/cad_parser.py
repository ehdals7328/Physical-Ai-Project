# modules/knowledge/cad_parser.py
"""
선박 CAD 도면(.dxf, .igs/.iges, .step/.stp, .stl, 도면 .pdf, .dwg) 전용 파서 및 초경량 2D SVG 미리보기 변환기.
- 2D DXF: ezdxf로 레이어·블록·TEXT/MTEXT/ATTRIB/DIMENSION 치수 및 주석 추출 + 초경량 SVG 도면 렌더링
- 3D IGES / STEP / STL: 무거운 C++ 엔진 없이 순수 파이썬 헤더/엔티티/치수(Bounding Box) 요약 추출
- 도면 PDF: 기존 rag_pipeline.build_chunks_for_pdf(표·그림·시맨틱 청킹) 재사용
"""
from __future__ import annotations

import math
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

from modules.knowledge.rag_pipeline import build_chunks_for_pdf

VESSEL_HINT_RULES: list[tuple[str, str, str]] = [
    ("gunnerus", "R/V Gunnerus 연구선 (NTNU)", "선체 구조·갑판(Deck)·이중저·엔진룸 일반배치 및 생산 CAD 도면"),
    ("kcs", "KRISO KCS 표준 컨테이너선", "컨테이너선 선형·프로펠러(KP505)·러더 국제 벤치마크 CAD 모델 및 제원표"),
    ("kvlcc2", "KRISO KVLCC2 초대형 원유운반선(VLCC)", "초대형 유조선 선체·프로펠러(KP458)·러더 국제 벤치마크 CAD 모델"),
    ("kvlcc1", "KRISO KVLCC1 초대형 원유운반선(VLCC)", "초대형 유조선 선형 국제 벤치마크 CAD 모델"),
    ("5415", "DTMB 5415 표준 함정(군함)", "소나돔·트랜섬 선미를 갖춘 표준 군함 선형 CAD 모델"),
]


def detect_vessel_context(file_path: Path) -> tuple[str, str]:
    joined = " ".join(p.lower() for p in file_path.parts)
    for keyword, vessel_name, desc in VESSEL_HINT_RULES:
        if keyword in joined:
            return vessel_name, desc
    return file_path.parent.name or "선박 도면", "선박 제조 CAD 도면 및 지오메트리 데이터"


def _clean_mtext(raw: str) -> str:
    if not raw:
        return ""
    text = raw.replace("\\P", " / ").replace("\\~", " ")
    text = re.sub(r"\\[A-Za-z][^;]*;", "", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_dxf_ascii_fallback(dxf_path: Path) -> dict[str, Any]:
    raw = dxf_path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in raw.splitlines()]
    layers: Counter[str] = Counter()
    texts: list[str] = []
    entity_types: Counter[str] = Counter()
    xs: list[float] = []
    ys: list[float] = []

    i = 0
    n = len(lines)
    current_entity = ""
    while i + 1 < n:
        code = lines[i]
        val = lines[i + 1]
        i += 2
        if code == "0":
            current_entity = val.upper()
            if current_entity not in ("SECTION", "ENDSEC", "EOF", "TABLE", "ENDTAB"):
                entity_types[current_entity] += 1
        elif code == "8" and val:
            layers[val] += 1
        elif code in ("1", "3") and current_entity in ("TEXT", "MTEXT", "ATTRIB", "DIMENSION"):
            cleaned = _clean_mtext(val)
            if cleaned and len(cleaned) >= 2:
                texts.append(cleaned)
        elif code == "10":
            try:
                xs.append(float(val))
            except ValueError:
                pass
        elif code == "20":
            try:
                ys.append(float(val))
            except ValueError:
                pass

    bbox = None
    if xs and ys:
        bbox = {
            "width": round(max(xs) - min(xs), 2),
            "height": round(max(ys) - min(ys), 2),
        }

    return {
        "format": "DXF (2D CAD)",
        "layers": [k for k, _ in layers.most_common(40)],
        "entity_counts": dict(entity_types.most_common(15)),
        "texts": list(dict.fromkeys(texts)),
        "dimensions": [],
        "blocks": [],
        "bbox": bbox,
    }


def parse_dxf_metadata(dxf_path: Path) -> dict[str, Any]:
    try:
        import ezdxf  # type: ignore
    except ImportError:
        return _parse_dxf_ascii_fallback(dxf_path)

    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    layers: Counter[str] = Counter()
    entity_counts: Counter[str] = Counter()
    texts: list[str] = []
    dimensions: list[str] = []
    blocks: list[str] = []

    for blk in doc.blocks:
        bname = str(blk.name or "").strip()
        if bname and not bname.startswith("*"):
            blocks.append(bname)

    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")

    def _update_pt(pt) -> None:
        nonlocal min_x, min_y, max_x, max_y
        try:
            x, y = float(pt[0]), float(pt[1])
            if math.isfinite(x) and math.isfinite(y):
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
        except Exception:
            pass

    for ent in msp:
        etype = ent.dxftype()
        entity_counts[etype] += 1
        layer = str(getattr(ent.dxf, "layer", "0") or "0")
        layers[layer] += 1

        if etype == "LINE":
            _update_pt(ent.dxf.start)
            _update_pt(ent.dxf.end)
        elif etype in ("CIRCLE", "ARC"):
            _update_pt(ent.dxf.center)
        elif etype == "LWPOLYLINE":
            for pt in ent.get_points("xy"):
                _update_pt(pt)
        elif etype == "TEXT":
            t = _clean_mtext(str(getattr(ent.dxf, "text", "") or ""))
            if t:
                texts.append(f"[{layer}] {t}")
        elif etype == "MTEXT":
            t = _clean_mtext(ent.plain_text() if hasattr(ent, "plain_text") else str(ent.text or ""))
            if t:
                texts.append(f"[{layer}] {t}")
        elif etype == "DIMENSION":
            dim_text = _clean_mtext(str(getattr(ent.dxf, "text", "") or ""))
            try:
                meas = ent.get_measurement()
                if isinstance(meas, (int, float)) and math.isfinite(meas):
                    dim_str = f"{dim_text} ({meas:.1f})" if dim_text and dim_text != "<>" else f"{meas:.1f}"
                else:
                    dim_str = dim_text
            except Exception:
                dim_str = dim_text
            if dim_str:
                dimensions.append(dim_str)
        elif etype == "INSERT":
            _update_pt(getattr(ent.dxf, "insert", (0, 0)))
            for attrib in getattr(ent, "attribs", []):
                tag = str(getattr(attrib.dxf, "tag", "") or "").strip()
                val = _clean_mtext(str(getattr(attrib.dxf, "text", "") or ""))
                if val:
                    texts.append(f"[{tag}] {val}" if tag else val)

    for blk in doc.blocks:
        if str(blk.name or "").startswith("*Model_Space"):
            continue
        for ent in blk:
            etype = ent.dxftype()
            if etype == "TEXT":
                t = _clean_mtext(str(getattr(ent.dxf, "text", "") or ""))
                if t and len(t) >= 2:
                    texts.append(t)
            elif etype == "MTEXT":
                t = _clean_mtext(ent.plain_text() if hasattr(ent, "plain_text") else str(ent.text or ""))
                if t and len(t) >= 2:
                    texts.append(t)

    bbox = None
    if math.isfinite(min_x) and math.isfinite(max_x):
        bbox = {
            "min_x": round(min_x, 2),
            "min_y": round(min_y, 2),
            "max_x": round(max_x, 2),
            "max_y": round(max_y, 2),
            "width": round(max_x - min_x, 2),
            "height": round(max_y - min_y, 2),
        }

    return {
        "format": "DXF (2D CAD)",
        "dxf_version": str(getattr(doc, "dxfversion", "")),
        "layers": [k for k, _ in layers.most_common(50)],
        "entity_counts": dict(entity_counts.most_common(15)),
        "texts": list(dict.fromkeys(texts)),
        "dimensions": list(dict.fromkeys(dimensions))[:80],
        "blocks": list(dict.fromkeys(blocks))[:50],
        "bbox": bbox,
    }


def parse_iges_metadata(igs_path: Path) -> dict[str, Any]:
    start_lines: list[str] = []
    global_chunks: list[str] = []
    entity_type_counts: Counter[str] = Counter()

    iges_type_names = {
        "100": "Circular Arc",
        "102": "Composite Curve",
        "106": "Copious Data",
        "108": "Plane",
        "110": "Line",
        "116": "Point",
        "120": "Surface of Revolution",
        "122": "Tabulated Cylinder",
        "126": "Rational B-Spline Curve (NURBS)",
        "128": "Rational B-Spline Surface (NURBS 선체 곡면)",
        "142": "Curve on Parametric Surface",
        "144": "Trimmed Parametric Surface",
        "186": "Manifold Solid B-Rep",
    }

    with igs_path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            if len(line) < 73:
                continue
            sec = line[72:73].upper()
            body = line[:72].rstrip()
            if sec == "S":
                if body.strip():
                    start_lines.append(body.strip())
            elif sec == "G":
                global_chunks.append(body.strip())
            elif sec == "D":
                code = line[:8].strip()
                if code.isdigit() and (idx % 2 == 0 or code in iges_type_names):
                    label = iges_type_names.get(code, f"IGES-{code}")
                    entity_type_counts[label] += 1
            elif sec == "P":
                break

    global_raw = "".join(global_chunks)
    hollerith_tokens = re.findall(r"\d+H([^,;]{2,80})", global_raw)
    clean_tokens = [t.strip() for t in hollerith_tokens if t.strip() and not t.strip().isdigit()]

    return {
        "format": "IGES (3D CAD 선체·부품 모델)",
        "start_comments": start_lines[:10],
        "header_tokens": list(dict.fromkeys(clean_tokens))[:20],
        "entity_counts": dict(entity_type_counts.most_common(12)),
        "size_kb": round(igs_path.stat().st_size / 1024.0, 2),
    }


def parse_step_metadata(stp_path: Path) -> dict[str, Any]:
    raw_head = stp_path.read_text(encoding="utf-8", errors="ignore")[:250_000]
    products = re.findall(r"PRODUCT\s*\(\s*'([^']*)'\s*,\s*'([^']*)'", raw_head, flags=re.IGNORECASE)
    file_desc = re.findall(r"FILE_DESCRIPTION\s*\(([^;]+)\);", raw_head, flags=re.IGNORECASE)
    file_name = re.findall(r"FILE_NAME\s*\(([^;]+)\);", raw_head, flags=re.IGNORECASE)
    entities = Counter(re.findall(r"=\s*([A-Z0-9_]+)\s*\(", raw_head))

    part_names = []
    for p1, p2 in products:
        label = p2.strip() or p1.strip()
        if label:
            part_names.append(label)

    return {
        "format": "STEP (3D CAD 솔리드/어셈블리)",
        "file_description": _clean_mtext(file_desc[0]) if file_desc else "",
        "file_name_meta": _clean_mtext(file_name[0]) if file_name else "",
        "part_names": list(dict.fromkeys(part_names))[:40],
        "entity_counts": dict(entities.most_common(12)),
        "size_kb": round(stp_path.stat().st_size / 1024.0, 2),
    }


def parse_stl_metadata(stl_path: Path) -> dict[str, Any]:
    data = stl_path.read_bytes()
    tri_count = 0
    bbox = None

    if len(data) >= 84:
        expected_tris = struct.unpack("<I", data[80:84])[0]
        if 84 + expected_tris * 50 == len(data):
            tri_count = expected_tris
            min_x = min_y = min_z = float("inf")
            max_x = max_y = max_z = float("-inf")
            step = max(1, expected_tris // 5000)
            for idx in range(0, expected_tris, step):
                off = 84 + idx * 50 + 12
                vals = struct.unpack("<9f", data[off:off + 36])
                for k in range(0, 9, 3):
                    x, y, z = vals[k], vals[k + 1], vals[k + 2]
                    min_x, max_x = min(min_x, x), max(max_x, x)
                    min_y, max_y = min(min_y, y), max(max_y, y)
                    min_z, max_z = min(min_z, z), max(max_z, z)
            if math.isfinite(min_x):
                bbox = {
                    "length_x": round(max_x - min_x, 2),
                    "width_y": round(max_y - min_y, 2),
                    "height_z": round(max_z - min_z, 2),
                }

    return {
        "format": "STL (3D 메쉬 모델)",
        "triangle_count": tri_count,
        "bbox": bbox,
        "size_kb": round(len(data) / 1024.0, 2),
    }


def extract_cad_summary(file_path: Path) -> dict[str, Any]:
    ext = file_path.suffix.lower()
    vessel_name, vessel_desc = detect_vessel_context(file_path)
    base = {
        "file_name": file_path.name,
        "extension": ext,
        "vessel_name": vessel_name,
        "vessel_desc": vessel_desc,
        "size_kb": round(file_path.stat().st_size / 1024.0, 2) if file_path.exists() else 0.0,
    }
    if not file_path.exists():
        base["format"] = "파일 미존재"
        return base

    if ext == ".dxf":
        base.update(parse_dxf_metadata(file_path))
    elif ext in (".igs", ".iges"):
        base.update(parse_iges_metadata(file_path))
    elif ext in (".stp", ".step"):
        base.update(parse_step_metadata(file_path))
    elif ext == ".stl":
        base.update(parse_stl_metadata(file_path))
    elif ext == ".pdf":
        base["format"] = "PDF (선박 CAD 도면 / 제원표)"
    elif ext == ".dwg":
        base["format"] = "DWG (AutoCAD 원본 도면)"
    else:
        base["format"] = f"CAD 파일 ({ext})"
    return base


def build_chunks_for_cad(
    cad_path: Path,
    *,
    file_hash: str,
    project_root: Path,
    rel_uri: str,
    semantic: bool = True,
    force_reparse: bool = False,
) -> list[dict[str, Any]]:
    ext = cad_path.suffix.lower()
    vessel_name, vessel_desc = detect_vessel_context(cad_path)

    if ext == ".pdf":
        chunks, _, _ = build_chunks_for_pdf(
            cad_path,
            file_hash=file_hash,
            project_root=project_root,
            semantic=semantic,
            ocr_enabled=False,
            force_reparse=force_reparse,
        )
        if not chunks:
            chunks, _, _ = build_chunks_for_pdf(
                cad_path,
                file_hash=file_hash,
                project_root=project_root,
                semantic=semantic,
                ocr_enabled=True,
                force_reparse=True,
            )
        if not chunks:
            chunks = [{
                "text": (
                    f"[선박 CAD 도면 PDF] 선종/데이터셋: {vessel_name}\n"
                    f"도면 파일명: {cad_path.name}\n"
                    f"설명: {vessel_desc}\n"
                    f"원본 저장 경로: {rel_uri}"
                ),
                "page_start": 1,
                "page_end": 1,
                "chunk_type": "CAD_PDF",
                "section_path": f"CAD/{vessel_name}/{cad_path.name}",
                "asset_path": rel_uri,
            }]
        else:
            for ch in chunks:
                ch["text"] = f"[선박 CAD 도면 · {vessel_name} · {cad_path.name}]\n{ch['text']}"
                ch["section_path"] = ch.get("section_path") or f"CAD/{vessel_name}/{cad_path.name}"
                ch["asset_path"] = ch.get("asset_path") or rel_uri
        return chunks

    if ext == ".dxf":
        meta = parse_dxf_metadata(cad_path)
        header_lines = [
            f"[선박 2D CAD 도면 (DXF)] 선종/선박: {vessel_name}",
            f"도면 파일명: {cad_path.name} (저장 경로: {rel_uri})",
            f"도면 개요: {vessel_desc}",
        ]
        if meta.get("bbox"):
            bb = meta["bbox"]
            header_lines.append(f"도면 외곽 범위(DXF 좌표계): 가로 {bb.get('width')} x 세로 {bb.get('height')}")
        if meta.get("layers"):
            header_lines.append("주요 도면 레이어(Layers): " + ", ".join(meta["layers"][:25]))
        if meta.get("blocks"):
            header_lines.append("도면 내부 블록/부품명(Blocks): " + ", ".join(meta["blocks"][:25]))
        if meta.get("dimensions"):
            header_lines.append("주요 표기 치수(Dimensions): " + ", ".join(meta["dimensions"][:30]))

        chunks: list[dict[str, Any]] = [{
            "text": "\n".join(header_lines),
            "page_start": 1,
            "page_end": 1,
            "chunk_type": "CAD_DXF",
            "section_path": f"CAD/{vessel_name}/{cad_path.name}/개요",
            "asset_path": rel_uri,
        }]

        texts = meta.get("texts") or []
        buf: list[str] = []
        buf_len = 0
        part_idx = 1
        for line in texts:
            if buf and buf_len + len(line) + 1 > 1350:
                chunks.append({
                    "text": (
                        f"[선박 2D CAD 주석·제원 · {vessel_name} · {cad_path.name} (파트 {part_idx})]\n"
                        + "\n".join(buf)
                    ),
                    "page_start": 1,
                    "page_end": 1,
                    "chunk_type": "CAD_DXF",
                    "section_path": f"CAD/{vessel_name}/{cad_path.name}/주석-{part_idx}",
                    "asset_path": rel_uri,
                })
                buf = []
                buf_len = 0
                part_idx += 1
            buf.append(line)
            buf_len += len(line) + 1

        if buf:
            chunks.append({
                "text": (
                    f"[선박 2D CAD 주석·제원 · {vessel_name} · {cad_path.name} (파트 {part_idx})]\n"
                    + "\n".join(buf)
                ),
                "page_start": 1,
                "page_end": 1,
                "chunk_type": "CAD_DXF",
                "section_path": f"CAD/{vessel_name}/{cad_path.name}/주석-{part_idx}",
                "asset_path": rel_uri,
            })
        return chunks

    if ext in (".igs", ".iges", ".stp", ".step", ".stl"):
        meta = extract_cad_summary(cad_path)
        lines = [
            f"[선박 3D CAD 지오메트리 모델] 선종/선박: {vessel_name}",
            f"파일명: {cad_path.name} | 형식: {meta.get('format')} | 용량: {meta.get('size_kb')} KB",
            f"개요: {vessel_desc}",
            f"2차 로봇(ROS2/MoveIt2) 연동 원본 경로: {rel_uri}",
        ]
        if meta.get("start_comments"):
            lines.append("모델 주석(Start Section): " + " / ".join(meta["start_comments"]))
        if meta.get("header_tokens"):
            lines.append("CAD 헤더 메타정보: " + ", ".join(meta["header_tokens"]))
        if meta.get("part_names"):
            lines.append("어셈블리 부품(BOM/Parts): " + ", ".join(meta["part_names"]))
        if meta.get("file_description"):
            lines.append("파일 설명: " + meta["file_description"])
        if meta.get("entity_counts"):
            ec_str = ", ".join(f"{k}: {v}개" for k, v in meta["entity_counts"].items())
            lines.append("3D 지오메트리 엔티티 구성: " + ec_str)
        if meta.get("bbox"):
            lines.append(f"3D 외곽 치수(Bounding Box): {meta['bbox']}")
        if meta.get("triangle_count"):
            lines.append(f"메쉬 삼각형 수: {meta['triangle_count']:,}개")

        return [{
            "text": "\n".join(lines),
            "page_start": 1,
            "page_end": 1,
            "chunk_type": "CAD_3D",
            "section_path": f"CAD/{vessel_name}/{cad_path.name}",
            "asset_path": rel_uri,
        }]

    return [{
        "text": (
            f"[선박 AutoCAD 원본 도면 (DWG)] 선종/선박: {vessel_name}\n"
            f"파일명: {cad_path.name} | 용량: {round(cad_path.stat().st_size / 1024.0, 2)} KB\n"
            f"개요: {vessel_desc}\n"
            f"원본 저장 경로: {rel_uri}"
        ),
        "page_start": 1,
        "page_end": 1,
        "chunk_type": "CAD_DWG",
        "section_path": f"CAD/{vessel_name}/{cad_path.name}",
        "asset_path": rel_uri,
    }]


def render_dxf_to_svg(
    dxf_path: Path,
    *,
    cache_dir: Path | None = None,
    max_entities: int = 25000,
) -> str:
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = f"{dxf_path.stem}_{int(dxf_path.stat().st_mtime)}_{dxf_path.stat().st_size}_v2.svg"
        cached_svg = cache_dir / cache_key
        if cached_svg.exists():
            return cached_svg.read_text(encoding="utf-8")

    try:
        import ezdxf  # type: ignore
    except ImportError:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 320" style="background:#0a192f;width:100%;height:100%">'
            '<text x="400" y="150" fill="#93c5fd" font-size="18" text-anchor="middle" font-family="sans-serif">'
            'pip install ezdxf 설치 후 2D CAD 청사진 도면이 표시됩니다.</text>'
            f'<text x="400" y="185" fill="#cbd5e1" font-size="13" text-anchor="middle">{xml_escape(dxf_path.name)}</text>'
            '</svg>'
        )

    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    segments: list[tuple[float, float, float, float]] = []
    circles: list[tuple[float, float, float]] = []
    xs: list[float] = []
    ys: list[float] = []

    def _add_pt(x: float, y: float) -> None:
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)

    def _collect_entity(ent, ox: float = 0.0, oy: float = 0.0, sx: float = 1.0, sy: float = 1.0) -> None:
        if len(segments) + len(circles) >= max_entities:
            return
        etype = ent.dxftype()
        if etype == "LINE":
            x1 = float(ent.dxf.start.x) * sx + ox
            y1 = float(ent.dxf.start.y) * sy + oy
            x2 = float(ent.dxf.end.x) * sx + ox
            y2 = float(ent.dxf.end.y) * sy + oy
            _add_pt(x1, y1)
            _add_pt(x2, y2)
            segments.append((x1, y1, x2, y2))
        elif etype == "LWPOLYLINE":
            pts = list(ent.get_points("xy"))
            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    x1, y1 = pts[i][0] * sx + ox, pts[i][1] * sy + oy
                    x2, y2 = pts[i + 1][0] * sx + ox, pts[i + 1][1] * sy + oy
                    _add_pt(x1, y1)
                    _add_pt(x2, y2)
                    segments.append((x1, y1, x2, y2))
                if ent.closed:
                    x1, y1 = pts[-1][0] * sx + ox, pts[-1][1] * sy + oy
                    x2, y2 = pts[0][0] * sx + ox, pts[0][1] * sy + oy
                    segments.append((x1, y1, x2, y2))
        elif etype == "POLYLINE":
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in ent.vertices]
            for i in range(len(pts) - 1):
                x1, y1 = pts[i][0] * sx + ox, pts[i][1] * sy + oy
                x2, y2 = pts[i + 1][0] * sx + ox, pts[i + 1][1] * sy + oy
                _add_pt(x1, y1)
                _add_pt(x2, y2)
                segments.append((x1, y1, x2, y2))
        elif etype == "CIRCLE":
            cx = float(ent.dxf.center.x) * sx + ox
            cy = float(ent.dxf.center.y) * sy + oy
            r = abs(float(ent.dxf.radius) * sx)
            _add_pt(cx, cy)
            circles.append((cx, cy, r))
        elif etype == "ARC":
            cx = float(ent.dxf.center.x) * sx + ox
            cy = float(ent.dxf.center.y) * sy + oy
            r = abs(float(ent.dxf.radius) * sx)
            a0 = math.radians(float(ent.dxf.start_angle))
            a1 = math.radians(float(ent.dxf.end_angle))
            if a1 <= a0:
                a1 += 2 * math.pi
            steps = 8
            prev_x = cx + r * math.cos(a0)
            prev_y = cy + r * math.sin(a0)
            _add_pt(prev_x, prev_y)
            for s in range(1, steps + 1):
                ang = a0 + (a1 - a0) * (s / steps)
                nx = cx + r * math.cos(ang)
                ny = cy + r * math.sin(ang)
                _add_pt(nx, ny)
                segments.append((prev_x, prev_y, nx, ny))
                prev_x, prev_y = nx, ny
        elif etype == "INSERT":
            try:
                blk = doc.blocks.get(ent.dxf.name)
                if blk is not None:
                    ins = ent.dxf.insert
                    bx = float(ins.x) * sx + ox
                    by = float(ins.y) * sy + oy
                    bsx = float(getattr(ent.dxf, "xscale", 1.0) or 1.0) * sx
                    bsy = float(getattr(ent.dxf, "yscale", 1.0) or 1.0) * sy
                    for sub in blk:
                        _collect_entity(sub, bx, by, bsx, bsy)
                        if len(segments) + len(circles) >= max_entities:
                            break
            except Exception:
                pass

    for ent in msp:
        _collect_entity(ent)
        if len(segments) + len(circles) >= max_entities:
            break

    if (not segments and not circles) or not xs or not ys:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 300" style="background:#0a192f;width:100%;height:100%">'
            f'<text x="400" y="150" fill="#93c5fd" font-size="16" text-anchor="middle">{xml_escape(dxf_path.name)} (2D 선분 정보 없음)</text>'
            '</svg>'
        )

    # 이상치(허공에 떨어진 점/선) 제거: 2% ~ 98% 분위수 기준으로 실제 도면 밀집 영역에 자동 포커스
    xs.sort()
    ys.sort()
    if len(xs) >= 50:
        min_x = xs[int(len(xs) * 0.02)]
        max_x = xs[int(len(xs) * 0.98)]
        min_y = ys[int(len(ys) * 0.02)]
        max_y = ys[int(len(ys) * 0.98)]
    else:
        min_x, max_x = xs[0], xs[-1]
        min_y, max_y = ys[0], ys[-1]

    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    # 5% 여백 추가
    min_x -= width * 0.05
    max_x += width * 0.05
    min_y -= height * 0.05
    max_y += height * 0.05
    width = max_x - min_x
    height = max_y - min_y

    svg_w, svg_h, pad = 1000.0, 560.0, 24.0
    scale = min((svg_w - 2 * pad) / width, (svg_h - 2 * pad) / height)
    off_x = (svg_w - width * scale) / 2.0
    off_y = (svg_h - height * scale) / 2.0

    def _in_bounds(x: float, y: float) -> bool:
        return (min_x <= x <= max_x) and (min_y <= y <= max_y)

    def _tx(x: float) -> float:
        return round((x - min_x) * scale + off_x, 1)

    def _ty(y: float) -> float:
        return round(svg_h - ((y - min_y) * scale + off_y), 1)

    path_cmds = [
        f"M{_tx(x1)} {_ty(y1)}L{_tx(x2)} {_ty(y2)}"
        for x1, y1, x2, y2 in segments
        if _in_bounds(x1, y1) or _in_bounds(x2, y2)
    ]
    circle_els = [
        f'<circle cx="{_tx(cx)}" cy="{_ty(cy)}" r="{max(round(r * scale, 1), 0.6)}"/>'
        for cx, cy, r in circles
        if _in_bounds(cx, cy) and (r * scale) < svg_w
    ]

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {int(svg_w)} {int(svg_h)}" '
        'style="background:linear-gradient(135deg,#071529 0%,#0c2340 100%);width:100%;height:100%;display:block">'
        '<defs><pattern id="cadGrid" width="40" height="40" patternUnits="userSpaceOnUse">'
        '<path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(147,197,253,0.08)" stroke-width="1"/>'
        '</pattern></defs>'
        f'<rect width="{int(svg_w)}" height="{int(svg_h)}" fill="url(#cadGrid)"/>'
        f'<path d="{" ".join(path_cmds)}" fill="none" stroke="#60a5fa" stroke-width="0.9" stroke-linecap="round"/>'
        f'<g fill="none" stroke="#38bdf8" stroke-width="0.9">{"".join(circle_els)}</g>'
        f'<text x="18" y="{int(svg_h) - 14}" fill="#93c5fd" font-size="12" font-family="monospace">'
        f'CAD 2D BLUEPRINT | {xml_escape(dxf_path.name)} | ENTITIES: {len(path_cmds) + len(circle_els)}</text>'
        '</svg>'
    )

    if cache_dir is not None:
        try:
            cached_svg.write_text(svg, encoding="utf-8")
        except Exception:
            pass
    return svg