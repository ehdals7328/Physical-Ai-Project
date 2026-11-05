from __future__ import annotations   # 미래 버전 타입힌트 쓰겠다 ex) file_path: str | Path 면 file_path가 str 또는 Path라는 파일 경로 객체 타입

import hashlib   # 해싱
import json
import os
from pathlib import Path
from typing import Any

from modules.knowledge.pdf_parser import parse_pdf
from modules.knowledge.semantic_chunker import chunk_blocks


# 파일 내용을 해싱해서 PDF 이름이 바뀌어도 같은 파일로 인식
def sha256_file(file_path: str | Path) -> str:
    """파일을 스트리밍 방식으로 읽어 SHA-256 해시를 계산합니다."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):   # 1MB 씩 쪼개서 해시 누적 (메모리 터짐 방지)
            digest.update(block)
    return digest.hexdigest()


# 캐시 JSON 파일 읽기
def _read_json(path: Path):
    """파싱 또는 청킹 캐시 JSON 파일을 읽습니다."""
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)

# JSON 파일을 저장하다가 서버가 꺼지거나 오류가 나도 파일이 깨지지 않게 보장
def _write_json_atomic(path: Path, payload: Any) -> None:
    """임시 파일 기록 후 교체 방식으로 캐시 JSON을 안전하게 저장합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    temp_path.replace(path)


# 청크 결과 저장할 폴더/경로 잡고, 전에 똑같은 조건으로 잘라둔 파일(캐시)이 있는지 검사 / 신규 PDF 적재시 기존 PDF는 넘어가줌
def build_chunks_for_pdf(
    pdf_path: str | Path,
    *,
    file_hash: str,
    project_root: str | Path,
    semantic: bool = True,
    ocr_enabled: bool = False,
    force_reparse: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """파일 해시와 파이프라인 설정을 기준으로 파싱된/청크 캐시를 사용합니다"""
    project_root = Path(project_root).resolve()
    cache_root = project_root / "data" / "rag_cache"
    parsed_dir = cache_root / "parsed"
    chunk_dir = cache_root / "chunks"
    parsed_file = parsed_dir / f"{file_hash}_ocr{int(ocr_enabled)}.json"

    max_chars = int(os.getenv("RAG_CHUNK_MAX_CHARS", "1800"))
    min_chars = int(os.getenv("RAG_CHUNK_MIN_CHARS", "300"))
    overlap = int(os.getenv("RAG_CHUNK_OVERLAP_CHARS", "120"))
    threshold = float(os.getenv("RAG_SEMANTIC_SIMILARITY_THRESHOLD", "0.58"))
    cache_key = (
        f"{file_hash}_sem{int(semantic)}_ocr{int(ocr_enabled)}"
        f"_max{max_chars}_min{min_chars}_ov{overlap}_thr{threshold:.2f}"
    )
    chunk_file = chunk_dir / f"{cache_key}.json"

    if not force_reparse and chunk_file.exists():
        payload = _read_json(chunk_file)
        return payload.get("chunks", []), payload.get("assets", []), {
            "parsed_cache_hit": parsed_file.exists(),
            "chunk_cache_hit": True,
            "block_count": payload.get("block_count", 0),
        }

    if not force_reparse and parsed_file.exists():
        parsed_payload = _read_json(parsed_file)
        blocks = parsed_payload.get("blocks", [])
        assets = parsed_payload.get("assets", [])
        parsed_cache_hit = True
    else:
        blocks, assets = parse_pdf(
            pdf_path,
            asset_root=project_root / "data" / "extracted_assets",
            file_hash=file_hash,
            ocr_enabled=ocr_enabled,
            ocr_lang=os.getenv("RAG_OCR_LANG", "kor+eng"),
        )
        _write_json_atomic(parsed_file, {"blocks": blocks, "assets": assets})
        parsed_cache_hit = False

    chunks = chunk_blocks(
        blocks,
        semantic=semantic,
        max_chars=max_chars,
        min_chars=min_chars,
        overlap_chars=overlap,
        similarity_threshold=threshold,
    )
    _write_json_atomic(chunk_file, {
        "chunks": chunks,
        "assets": assets,
        "block_count": len(blocks),
    })
    return chunks, assets, {
        "parsed_cache_hit": parsed_cache_hit,
        "chunk_cache_hit": False,
        "block_count": len(blocks),
    }
