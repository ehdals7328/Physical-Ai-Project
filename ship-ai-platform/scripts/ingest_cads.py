#!/usr/bin/env python3
"""
선박 CAD 도면(.dxf, .igs/.iges, .stp/.step, .stl, 도면 .pdf, .dwg) 일괄 RAG 적재 스크립트.
- Oracle DOCUMENT_METADATA 테이블에 CATEGORY_CODE='CAD_DRAWING'으로 등록
- ChromaDB(shipbuilding_docs, RAG_V2)에 무중단 적재하여 기존 AI Agent(retriever_tools.py)가 즉시 검색 가능
- github_pygem/ 테스트 샘플은 기본 제외 (--include-samples 로 포함 가능)

[사용법]
  python scripts/ingest_cads.py
  python scripts/ingest_cads.py --cad-dir ./data/cad_drawings
  python scripts/ingest_cads.py --dry-run
  python scripts/ingest_cads.py --force
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from web import create_web  # noqa: E402
from web.models.schema import db, DocumentMetadata  # noqa: E402
from modules.knowledge.rag_pipeline import sha256_file  # noqa: E402
from modules.knowledge.cad_parser import build_chunks_for_cad, detect_vessel_context  # noqa: E402
from modules.knowledge.retriever import KnowledgeStore  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag.ingest_cads")

SUPPORTED_CAD_EXTS = {".dxf", ".pdf", ".igs", ".iges", ".stp", ".step", ".stl", ".dwg"}
DEFAULT_EXCLUDED_DIRS = {"github_pygem", "__macosx", "_tmp_zips"}


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_cad_collection(cad_path: Path) -> tuple[str, str]:
    return "INTERNAL", "col_internal_ship_sop"


def make_cad_doc_code(cad_path: Path, collection_type: str, file_hash: str) -> str:
    prefix = "CAD-INT" if collection_type == "INTERNAL" else "CAD-PUB"
    ext_tag = cad_path.suffix.lstrip(".").upper()[:4]
    clean_stem = re.sub(r"[^A-Za-z0-9]", "", cad_path.stem)[:12].upper() or "DRAW"
    return f"{prefix}-{ext_tag}-{clean_stem}-{file_hash[-6:].upper()}"[:50]


def make_friendly_title(cad_path: Path) -> str:
    vessel_name, _ = detect_vessel_context(cad_path)
    ext_label = cad_path.suffix.lstrip(".").upper()
    return f"[{vessel_name}] {cad_path.stem} ({ext_label})"[:200]


def find_or_create_cad_metadata(cad_path: Path, file_hash: str, *, dry_run: bool):
    collection_type, vector_collection = resolve_cad_collection(cad_path)
    rel_uri = relative_path(cad_path)
    friendly_title = make_friendly_title(cad_path)

    exact = (
        DocumentMetadata.query
        .filter_by(file_hash_sha256=file_hash)
        .order_by(DocumentMetadata.document_id.desc())
        .first()
    )
    if exact:
        if not dry_run:
            exact.collection_type = collection_type
            exact.vector_collection = vector_collection
            exact.category_code = "CAD_DRAWING"
            exact.document_title = friendly_title
            exact.original_file_name = cad_path.name[:255]
            exact.storage_uri = rel_uri[:500]
            exact.is_active = "Y"
        return exact, False

    if dry_run:
        return None, True

    uploaded_by = int(os.getenv("DEFAULT_INGEST_USER_ID", "1"))
    doc_code = make_cad_doc_code(cad_path, collection_type, file_hash)
    if DocumentMetadata.query.filter_by(document_code=doc_code).first():
        doc_code = f"{doc_code[:42]}-{file_hash[:6].upper()}"

    row = DocumentMetadata(
        document_code=doc_code,
        document_title=friendly_title,
        collection_type=collection_type,
        category_code="CAD_DRAWING",
        original_file_name=cad_path.name[:255],
        storage_uri=rel_uri[:500],
        file_hash_sha256=file_hash[:64],
        vector_collection=vector_collection,
        ingest_status="PENDING",
        is_active="Y",
        uploaded_by=uploaded_by,
    )
    db.session.add(row)
    db.session.flush()
    return row, True


def should_skip_cad(path: Path, include_samples: bool) -> bool:
    if path.name.startswith("._") or path.name.lower() == "cad_manifest.json":
        return True
    if path.suffix.lower() not in SUPPORTED_CAD_EXTS:
        return True
    parts_low = {p.lower() for p in path.parts}
    if not include_samples and (parts_low & DEFAULT_EXCLUDED_DIRS):
        return True
    return False


def process_one_cad(cad_path: Path, store: KnowledgeStore | None, args) -> str:
    file_hash = sha256_file(cad_path)
    rel_uri = relative_path(cad_path)
    collection_type, vector_collection = resolve_cad_collection(cad_path)
    logger.info("CAD 확인 [%s/%s]: %s | SHA256=%s", collection_type, cad_path.suffix.upper(), rel_uri, file_hash[:12])

    row, _ = find_or_create_cad_metadata(cad_path, file_hash, dry_run=args.dry_run)

    if (
        row is not None
        and store is not None
        and not args.force
        and not args.dry_run
        and store.is_document_indexed(row.document_id, file_hash)
    ):
        if row.ingest_status != "COMPLETED":
            row.ingest_status = "COMPLETED"
        db.session.commit()
        logger.info("이미 CAD RAG 적재됨 (스킵): %s (ID=%s)", cad_path.name, row.document_id)
        return "SKIPPED"

    if not args.dry_run:
        assert row is not None
        row.ingest_status = "PENDING"
        db.session.commit()
        document_id = int(row.document_id)
    else:
        document_id = int(row.document_id) if row is not None else 0

    try:
        chunks = build_chunks_for_cad(
            cad_path,
            file_hash=file_hash,
            project_root=PROJECT_ROOT,
            rel_uri=rel_uri,
            semantic=not args.no_semantic,
            force_reparse=args.force,
        )
        if not chunks:
            raise RuntimeError("CAD에서 적재 가능한 텍스트/메타 청크를 생성하지 못했습니다.")

        if args.dry_run:
            logger.info("  [DRY-RUN] 생성 예정 청크 수: %d개 | 샘플: %s", len(chunks), chunks[0]["text"][:120].replace("\n", " "))
            return "DRY_RUN"

        assert row is not None and store is not None
        count = store.replace_document_chunks(
            document_id=document_id,
            document_name=row.document_title,
            category="CAD_DRAWING",
            source=row.collection_type,
            revision="CAD-Rev.1",
            file_hash=file_hash,
            chunks=chunks,
            access_level=row.collection_type,
            document_status="ACTIVE" if row.is_active == "Y" else "INACTIVE",
        )
        row.ingest_status = "COMPLETED"
        row.is_active = "Y"
        db.session.commit()
        logger.info("CAD 적재 완료: ID=%s, chunks=%s, collection=%s", document_id, count, vector_collection)
        return "COMPLETED"
    except Exception:
        if args.dry_run:
            raise
        db.session.rollback()
        failed_row = db.session.get(DocumentMetadata, document_id) if document_id else None
        if failed_row is not None:
            failed_row.ingest_status = "FAILED"
            db.session.commit()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="선박 CAD 도면 일괄 RAG(ChromaDB + Oracle) 적재기")
    parser.add_argument("--cad-dir", default="", help="CAD 폴더 경로 (기본값: data/cad_drawings)")
    parser.add_argument("--dry-run", action="store_true", help="DB/Chroma에 쓰지 않고 대상 및 청크 미리보기")
    parser.add_argument("--force", action="store_true", help="이미 적재된 CAD도 강제 재파싱·재임베딩")
    parser.add_argument("--no-semantic", action="store_true", help="PDF 시맨틱 청킹 끄기")
    parser.add_argument("--include-samples", action="store_true", help="github_pygem 샘플 메쉬까지 모두 포함")
    parser.add_argument("--limit", type=int, default=0, help="최대 처리 파일 수 제한 (테스트용)")
    args = parser.parse_args()

    cad_root = Path(args.cad_dir).resolve() if args.cad_dir else (PROJECT_ROOT / "data" / "cad_drawings")
    if not cad_root.exists():
        logger.error("CAD 폴더가 존재하지 않습니다: %s", cad_root)
        logger.info("먼저 CAD 압축파일을 %s 위치에 풀어주세요.", PROJECT_ROOT / "data" / "cad_drawings")
        return 1

    cad_files = sorted(
        p for p in cad_root.rglob("*")
        if p.is_file() and not should_skip_cad(p, args.include_samples)
    )
    if args.limit > 0:
        cad_files = cad_files[:args.limit]

    if not cad_files:
        logger.warning("적재할 CAD 파일이 없습니다: %s", cad_root)
        return 0

    web = create_web()
    summary = {"COMPLETED": 0, "SKIPPED": 0, "FAILED": 0, "DRY_RUN": 0}

    with web.app_context():
        store = None if args.dry_run else KnowledgeStore()
        for idx, cad_path in enumerate(cad_files, start=1):
            logger.info("[%d/%d] CAD 적재 처리 중: %s", idx, len(cad_files), cad_path.name)
            try:
                status = process_one_cad(cad_path, store, args)
                summary[status] = summary.get(status, 0) + 1
            except Exception as exc:
                db.session.rollback()
                summary["FAILED"] += 1
                logger.exception("CAD 적재 실패: %s (%s)", cad_path.name, exc)

        logger.info("최종 CAD RAG 적재 요약: %s", summary)
    return 1 if summary["FAILED"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
