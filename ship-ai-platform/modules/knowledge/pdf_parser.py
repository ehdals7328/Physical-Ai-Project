# PDF 파싱 (표 병합셀 복원, 캡션/비고 결합, 본문 중복 제거 완성본)
from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ocr_image(image_path: Path, lang: str) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        logger.warning("OCR 패키지 미설치: pytesseract, Pillow를 설치하세요.")
        return ""

    tesseract_cmd = os.getenv("TESSERACT_CMD", "").strip()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    elif not shutil.which("tesseract"):
        logger.warning("Tesseract 실행 파일이 없어 OCR을 건너뜁니다.")
        return ""

    try:
        with Image.open(image_path) as image:
            return pytesseract.image_to_string(image, lang=lang).strip()
    except Exception as exc:
        logger.warning("OCR 실패 (%s): %s", image_path.name, exc)
        return ""


def _clean_cell(val: Any) -> str:
    if val is None:
        return ""
    text = str(val).replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _detect_header_depth(rows: list[list[str]]) -> int:
    if len(rows) <= 2:
        return 1
    header_keywords = ("기호", "처리", "시험", "강도", "연신율", "온도", "에너지", "두께", "방향", "구분", "종류", "호칭")
    depth = 1
    for r_idx in range(min(4, len(rows) - 1)):
        row_str = " ".join(rows[r_idx])
        # 데이터 행(예: AH 43, 420이상 등)이 시작되면 중단
        if re.search(r"\b(AH|DH|EH|FH)\s*\d+\b", row_str) or "이상" in row_str or "~" in row_str:
            break
        if any(k in row_str for k in header_keywords) or r_idx == 0:
            depth = r_idx + 1
    return max(1, depth)


def _format_table_structured(
    raw_rows: list[list[Any]],
    *,
    page_no: int,
    table_index: int,
    caption: str = "",
    notes: str = "",
) -> str:
    if not raw_rows:
        return ""

    cleaned = [[_clean_cell(c) for c in row] for row in raw_rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return ""

    num_cols = max(len(r) for r in cleaned)
    grid = [r + [""] * (num_cols - len(r)) for r in cleaned]

    header_depth = _detect_header_depth(grid)
    header_rows = [list(r) for r in grid[:header_depth]]
    data_rows = [list(r) for r in grid[header_depth:]]

    for r in range(len(header_rows)):
        for c in range(1, num_cols):
            if not header_rows[r][c] and header_rows[r][c - 1]:
                header_rows[r][c] = header_rows[r][c - 1]

    col_names: list[str] = []
    for c in range(num_cols):
        parts: list[str] = []
        for r in range(len(header_rows)):
            val = header_rows[r][c]
            if val and (not parts or parts[-1] != val):
                parts.append(val)
        col_names.append(" > ".join(parts) if parts else f"열{c + 1}")

    if data_rows:
        for r in range(1, len(data_rows)):
            for c in range(num_cols):
                if data_rows[r][c] == "" and data_rows[r - 1][c] not in ("", "-"):
                    data_rows[r][c] = data_rows[r - 1][c]

    title_line = f"[PDF 표 {table_index}, 페이지 {page_no}]"
    if caption:
        title_line += f" {caption}"

    lines: list[str] = [title_line]
    lines.append("| " + " | ".join(col_names) + " |")
    lines.append("| " + " | ".join(["---"] * num_cols) + " |")

    for row in data_rows:
        if not any(row):
            continue
        # 마크다운 행 추가
        lines.append("| " + " | ".join(c if c else "-" for c in row) + " |")

    if notes:
        lines.append(f"\n[표 비고 및 주석]\n{notes}")

    return "\n".join(lines).strip()


def _extract_caption_and_notes(page, table_bbox: tuple[float, float, float, float]) -> tuple[str, str]:
    x0, y0, x1, y1 = table_bbox
    page_rect = page.rect

    cap_rect = (page_rect.x0, max(page_rect.y0, y0 - 65), page_rect.x1, y0)
    caption_raw = page.get_text("text", clip=cap_rect, sort=True).strip()
    caption_lines = [line.strip() for line in caption_raw.splitlines() if line.strip()]
    caption = caption_lines[-1] if caption_lines else ""

    note_rect = (page_rect.x0, y1, page_rect.x1, min(page_rect.y1, y1 + 190))
    notes_raw = page.get_text("text", clip=note_rect, sort=True).strip()
    notes = ""
    if any(k in notes_raw for k in ("(비고)", "비고", "(1)", "주)", "(주)")):
        notes = notes_raw

    return caption, notes


def parse_pdf(
    pdf_path: str | Path,
    *,
    asset_root: str | Path,
    file_hash: str,
    ocr_enabled: bool = False,
    ocr_lang: str = "kor+eng",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF가 설치되지 않았습니다. pip install PyMuPDF") from exc

    pdf_path = Path(pdf_path)
    asset_dir = Path(asset_root) / file_hash[:20]
    asset_dir.mkdir(parents=True, exist_ok=True)

    blocks: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []

    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            page_no = page_index + 1
            table_bboxes: list[tuple[float, float, float, float]] = []

            try:
                finder = page.find_tables()
                found_tables = getattr(finder, "tables", []) or []
                for table_index, table in enumerate(found_tables, start=1):
                    bbox = tuple(table.bbox)
                    table_bboxes.append(bbox)
                    rows = table.extract() or []
                    caption, notes = _extract_caption_and_notes(page, bbox)
                    table_text = _format_table_structured(
                        rows,
                        page_no=page_no,
                        table_index=table_index,
                        caption=caption,
                        notes=notes,
                    )
                    if len(table_text) >= 20:
                        blocks.append({
                            "text": table_text,
                            "page_number": page_no,
                            "chunk_type": "TABLE",
                            "asset_path": "",
                            "section_path": caption[:200] if caption else "",
                        })
            except Exception as exc:
                logger.warning("표 추출 중 오류: page=%s (%s)", page_no, exc)

            if table_bboxes:
                text_blocks = page.get_text("blocks", sort=True)
                non_table_lines: list[str] = []
                for b in text_blocks:
                    bx0, by0, bx1, by1, btext = b[0], b[1], b[2], b[3], b[4]
                    b_center_x = (bx0 + bx1) / 2
                    b_center_y = (by0 + by1) / 2
                    in_table = any(
                        (tx0 - 2 <= b_center_x <= tx1 + 2) and (ty0 - 2 <= b_center_y <= ty1 + 2)
                        for (tx0, ty0, tx1, ty1) in table_bboxes
                    )
                    if not in_table and str(btext).strip():
                        non_table_lines.append(str(btext).strip())
                page_text = "\n\n".join(non_table_lines).strip()
            else:
                page_text = page.get_text("text", sort=True).strip()

            if page_text:
                blocks.append({
                    "text": page_text,
                    "page_number": page_no,
                    "chunk_type": "TEXT",
                    "asset_path": "",
                    "section_path": "",
                })

            seen_xrefs: set[int] = set()
            image_number = 0
            for image_info in page.get_images(full=True):
                xref = int(image_info[0])
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    image_data = document.extract_image(xref)
                    raw = image_data.get("image", b"")
                    if not raw:
                        continue
                    ext = str(image_data.get("ext", "png")).lower()
                    if ext not in {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}:
                        ext = "png"
                    digest = _sha256_bytes(raw)[:12]
                    image_number += 1
                    image_file = asset_dir / f"page_{page_no:04d}_image_{image_number:03d}_{digest}.{ext}"
                    if not image_file.exists():
                        image_file.write_bytes(raw)

                    relative_path = image_file.as_posix()
                    asset = {
                        "page_number": page_no,
                        "asset_type": "IMAGE",
                        "asset_path": relative_path,
                        "file_hash": _sha256_bytes(raw),
                        "ocr_text": "",
                    }
                    if ocr_enabled:
                        ocr_text = _ocr_image(image_file, ocr_lang)
                        asset["ocr_text"] = ocr_text
                        if len(ocr_text.strip()) >= 8:
                            blocks.append({
                                "text": f"[PDF 그림 OCR, 페이지 {page_no}]\n{ocr_text}",
                                "page_number": page_no,
                                "chunk_type": "IMAGE_OCR",
                                "asset_path": relative_path,
                                "section_path": "",
                            })
                    assets.append(asset)
                except Exception as exc:
                    logger.warning("이미지 추출 실패: page=%s xref=%s (%s)", page_no, xref, exc)

            if ocr_enabled and len(page_text) < 30 and not table_bboxes:
                try:
                    pix = page.get_pixmap(dpi=220, alpha=False)
                    scan_file = asset_dir / f"page_{page_no:04d}_scan.png"
                    if not scan_file.exists():
                        pix.save(str(scan_file))
                    ocr_text = _ocr_image(scan_file, ocr_lang)
                    if len(ocr_text.strip()) >= 20:
                        blocks.append({
                            "text": f"[PDF 스캔 페이지 OCR, 페이지 {page_no}]\n{ocr_text}",
                            "page_number": page_no,
                            "chunk_type": "PAGE_OCR",
                            "asset_path": scan_file.as_posix(),
                            "section_path": "",
                        })
                    assets.append({
                        "page_number": page_no,
                        "asset_type": "SCAN_PAGE",
                        "asset_path": scan_file.as_posix(),
                        "file_hash": _sha256_bytes(scan_file.read_bytes()),
                        "ocr_text": ocr_text,
                    })
                except Exception as exc:
                    logger.warning("페이지 OCR 실패: page=%s (%s)", page_no, exc)

        if not blocks and not ocr_enabled:
            logger.info("텍스트 레이어가 없는 스캔 PDF 감지 (%s) -> 자동 OCR 모드로 재추출을 시도합니다.", pdf_path.name)
            return parse_pdf(
                pdf_path,
                asset_root=asset_root,
                file_hash=file_hash,
                ocr_enabled=True,
                ocr_lang=ocr_lang,
            )

    return blocks, assets