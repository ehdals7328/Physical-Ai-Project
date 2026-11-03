from __future__ import annotations

import os
import re
from typing import Any

import numpy as np

from modules.knowledge.embedding import get_embedder

# 조선/선급 규격서 PDF 내 깨지는 특수기호(PUA 영역) 복원 매핑
_PUA_SYMBOL_MAP = str.maketrans({
    "\ue054": "/",   #  -> / (예: N/mm2)
    "\uf032": "²",   #  -> ² (제곱)
    "\uf033": "³",   # 세제곱
    "\ue0b4": "%",   #  -> % (연신율(%))
    "\uf0ec": "L",   #  -> L
    "\uf03d": "=",   #  -> =
    "\uf035": "5",   #  -> 5
    "\ue053": ".",   #  -> .
    "\uf036": "6",   #  -> 6
    "\uf073": "√",   #  -> √ (루트)
    "\uf0c0": "A",   #  -> A
    "\u3161": "",    # ㅡ (불필요한 윗줄 기호 제거)
    "\ufffd": "",    #  (유니코드 대체 문자 제거)
})


def _normalize(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.translate(_PUA_SYMBOL_MAP)
    text = re.sub(r"N\s*[\ue000-\uf8ff□]\s*mm\s*[\ue000-\uf8ff□2²]?", "N/mm²", text)
    text = re.sub(r"연신율\s*\(\s*[\ue000-\uf8ff□%]\s*\)", "연신율(%)", text)
    text = re.sub(r"\(\s*[L\uf0ec□]\s*[=\uf03d□]\s*5\.?65\s*[√\uf073□]*\s*[A\uf0c0□]*\s*\)", "(L=5.65√A)", text)
    text = re.sub(r"[\ue000-\uf8ff]", "", text)
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_long_paragraph(text: str, max_chars: int) -> list[str]:
    text = _normalize(text)
    if len(text) <= max_chars:
        return [text] if text else []

    sentences = re.split(r"(?<=[.!?。！？])\s+|(?<=다\.)\s+", text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                pieces.append(current.strip())
                current = ""
            for start in range(0, len(sentence), max_chars):
                part = sentence[start:start + max_chars].strip()
                if part:
                    pieces.append(part)
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > max_chars and current:
            pieces.append(current.strip())
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current.strip())
    return pieces


def _paragraphs(text: str, max_chars: int) -> list[str]:
    text = _normalize(text)
    if not text:
        return []

    raw_paragraphs = re.split(r"\n\s*\n+", text)
    paragraphs: list[str] = []
    for raw in raw_paragraphs:
        p = _normalize(raw.replace("\n", " "))
        if len(p) < 2:
            continue
        paragraphs.extend(_split_long_paragraph(p, max_chars))

    if len(paragraphs) == 1 and len(paragraphs[0]) > max_chars // 2:
        paragraphs = _split_long_paragraph(paragraphs[0], max_chars)
    return paragraphs


def _pack_fixed(units: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    output: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n{unit}".strip() if current else unit
        if len(candidate) > max_chars and current:
            output.append(current)
            tail = current[-overlap_chars:].strip() if overlap_chars > 0 else ""
            current = f"{tail}\n{unit}".strip() if tail else unit
            if len(current) > max_chars:
                current = unit
        else:
            current = candidate
    if current:
        output.append(current)
    return output


def _semantic_pack(
    units: list[str],
    *,
    max_chars: int,
    min_chars: int,
    similarity_threshold: float,
    overlap_chars: int,
) -> list[str]:
    if len(units) <= 1:
        return _pack_fixed(units, max_chars, overlap_chars)

    model = get_embedder()
    batch_size = int(os.getenv("SEMANTIC_BATCH_SIZE", "32"))
    vectors = model.encode(
        units,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)

    output: list[str] = []
    current_units: list[str] = []
    current_len = 0

    for idx, unit in enumerate(units):
        should_split = False
        if current_units:
            similarity = float(np.dot(vectors[idx - 1], vectors[idx]))
            would_exceed = current_len + len(unit) + 1 > max_chars
            semantic_shift = similarity < similarity_threshold and current_len >= min_chars
            should_split = would_exceed or semantic_shift

        if should_split:
            previous = "\n".join(current_units).strip()
            output.append(previous)
            tail = previous[-overlap_chars:].strip() if overlap_chars > 0 else ""
            current_units = [tail, unit] if tail else [unit]
            current_len = sum(len(x) for x in current_units) + max(0, len(current_units) - 1)
        else:
            current_units.append(unit)
            current_len += len(unit) + (1 if current_len else 0)

    if current_units:
        output.append("\n".join(current_units).strip())
    return [x for x in output if x]


def chunk_blocks(
    blocks: list[dict[str, Any]],
    *,
    semantic: bool = True,
    max_chars: int | None = None,
    min_chars: int | None = None,
    overlap_chars: int | None = None,
    similarity_threshold: float | None = None,
) -> list[dict[str, Any]]:
    max_chars = max_chars or int(os.getenv("RAG_CHUNK_MAX_CHARS", "1800"))
    table_max_chars = int(os.getenv("RAG_TABLE_MAX_CHARS", str(max(max_chars, 4500))))
    min_chars = min_chars or int(os.getenv("RAG_CHUNK_MIN_CHARS", "300"))
    overlap_chars = overlap_chars if overlap_chars is not None else int(os.getenv("RAG_CHUNK_OVERLAP_CHARS", "120"))
    similarity_threshold = similarity_threshold if similarity_threshold is not None else float(
        os.getenv("RAG_SEMANTIC_SIMILARITY_THRESHOLD", "0.58")
    )

    result: list[dict[str, Any]] = []

    for block in blocks:
        text = _normalize(str(block.get("text", "")))
        if not text:
            continue
        page = int(block.get("page_number", 1))
        chunk_type = str(block.get("chunk_type", "TEXT")).upper()
        asset_path = str(block.get("asset_path", ""))

        if chunk_type == "TABLE":
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if not lines:
                pieces = []
            elif len(text) <= table_max_chars:
                pieces = [text]
            else:
                header_lines = lines[:min(3, len(lines))]
                header_str = "\n".join(header_lines)
                table_groups: list[str] = []
                current_lines: list[str] = []
                current_len = 0

                for line_index, line in enumerate(lines):
                    candidate_len = current_len + len(line) + (1 if current_lines else 0)
                    if current_lines and candidate_len > table_max_chars:
                        table_groups.append("\n".join(current_lines))
                        current_lines = list(header_lines) if line_index >= len(header_lines) else []
                        current_len = len(header_str) if current_lines else 0
                    current_lines.append(line)
                    current_len += len(line) + (1 if current_len else 0)
                if current_lines:
                    table_groups.append("\n".join(current_lines))
                pieces = table_groups
        elif chunk_type in {"IMAGE_OCR", "PAGE_OCR"}:
            pieces = _pack_fixed(_split_long_paragraph(text, max_chars), max_chars, 0)
        else:
            units = _paragraphs(text, max_chars)
            if semantic:
                pieces = _semantic_pack(
                    units,
                    max_chars=max_chars,
                    min_chars=min_chars,
                    similarity_threshold=similarity_threshold,
                    overlap_chars=overlap_chars,
                )
            else:
                pieces = _pack_fixed(units, max_chars, overlap_chars)

        for piece in pieces:
            if len(piece.strip()) < 3:
                continue
            result.append({
                "text": piece.strip(),
                "page_start": page,
                "page_end": page,
                "chunk_type": chunk_type,
                "asset_path": asset_path,
                "section_path": str(block.get("section_path", "")),
            })

    return result