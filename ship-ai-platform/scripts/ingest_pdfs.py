# scripts/ingest_pdfs.py
# data/internal_pdfs -> COLLECTION_TYPE='INTERNAL', VECTOR_COLLECTION='col_internal_ship_sop'
# data/raw_pdfs      -> COLLECTION_TYPE='PUBLIC',   VECTOR_COLLECTION='col_public_weld_std'
# DB(DOCUMENT_METADATA) 기록은 절대 삭제(DELETE)하지 않고 보존하며,
# 폴더에서 지워진 파일은 IS_ACTIVE='N' 처리 후 ChromaDB 벡터만 정리합니다.

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
from modules.knowledge.rag_pipeline import build_chunks_for_pdf, sha256_file  # noqa: E402
from modules.knowledge.retriever import KnowledgeStore  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag.ingest")

CATEGORY_RULES = [
    ("SAFETY", ("안전", "보건", "비상", "대응", "kosha", "osha", "z49", "safety", "health")),
    ("CRANE_GUIDE", ("중량", "crane", "rigging", "lifting", "hoist", "인양", "장비")),
    ("CAD_DRAWING", ("도면", "cad", "drawing", "블록", "배치도")),
    ("WELDING_SOP", ("용접", "품질", "검사", "인사", "근로", "임금", "보수", "휴일", "휴가", "성과", "윤리", "보안", "문서", "규정", "sop", "가이드", "운영")),
]


def normalize_name(value: str) -> str:
    value = (value or "").strip().casefold().replace("\\", "/")
    value = value.rsplit("/", 1)[-1]
    return re.sub(r"\s+", "", value)


def guess_category_code(pdf_path: Path) -> str:
    search_text = f"{' '.join(p.casefold() for p in pdf_path.parts)} {pdf_path.name.casefold()}"
    for category_code, keywords in CATEGORY_RULES:
        if any(k in search_text for k in keywords):
            return category_code
    return "WELDING_SOP"


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_folder_spec(pdf_path: Path) -> tuple[str, str]:
    parts_lower = [p.lower() for p in pdf_path.parts]
    if "raw_pdfs" in parts_lower or "public_pdfs" in parts_lower:
        return "PUBLIC", "col_public_weld_std"
    return "INTERNAL", "col_internal_ship_sop"


def make_document_code(pdf_path: Path, collection_type: str, file_hash: str) -> str:
    prefix = "DOC-INT" if collection_type == "INTERNAL" else "DOC-PUB"
    m = re.match(r"^(\d{1,3})[_\-\s]", pdf_path.stem)
    num_tag = f"{int(m.group(1)):03d}" if m else file_hash[:4].upper()
    clean_stem = re.sub(r"[^A-Za-z0-9]", "", pdf_path.stem)[:10].upper()
    if clean_stem:
        return f"{prefix}-{clean_stem}-{file_hash[-6:].upper()}"[:50]
    return f"{prefix}-{num_tag}-{file_hash[-6:].upper()}"[:50]


def deactivate_missing_file_rows(store: KnowledgeStore | None, *, dry_run: bool = False) -> int:
    """디스크 폴더에서 삭제된 파일은 DB 기록을 지우지 않고 IS_ACTIVE='N' 으로 보존하며 RAG 벡터만 제거"""
    deactivated = 0
    for row in DocumentMetadata.query.all():
        uri = (row.storage_uri or "").strip()
        abs_p = (PROJECT_ROOT / uri) if uri and not uri.startswith(("http://", "https://", "s3://")) else None
        if abs_p is None or not abs_p.exists():
            if row.is_active == "Y":
                logger.info("폴더 미존재 문서 비활성 보존 처리 (DB유지/RAG제거): [%s] %s", row.document_code, row.document_title)
                if not dry_run:
                    row.is_active = "N"
                    if store is not None:
                        try:
                            res = store.collection.get(where={"document_id": {"$eq": int(row.document_id)}}, include=[])
                            ids = list(res.get("ids") or [])
                            if ids:
                                store.collection.delete(ids=ids)
                        except Exception:
                            pass
                deactivated += 1
    if deactivated and not dry_run:
        db.session.commit()
    return deactivated


def find_or_create_metadata(pdf_path: Path, file_hash: str, *, dry_run: bool):
    collection_type, vector_collection = resolve_folder_spec(pdf_path)
    category_code = guess_category_code(pdf_path)
    rel_uri = relative_path(pdf_path)
    clean_title = pdf_path.stem[:200]

    docs = DocumentMetadata.query.all()

    exact = next((r for r in docs if (r.file_hash_sha256 or "").strip().lower() == file_hash), None)
    if exact:
        if not dry_run:
            exact.collection_type = collection_type
            exact.vector_collection = vector_collection
            exact.category_code = category_code
            exact.original_file_name = pdf_path.name[:255]
            exact.storage_uri = rel_uri[:500]
            exact.is_active = "Y"
        return exact, False

    target_name = normalize_name(pdf_path.name)
    same_name = next((r for r in docs if normalize_name(r.original_file_name) == target_name and r.is_active == "Y"), None)
    if same_name:
        if not dry_run:
            same_name.collection_type = collection_type
            same_name.vector_collection = vector_collection
            same_name.category_code = category_code
            same_name.file_hash_sha256 = file_hash
            same_name.storage_uri = rel_uri[:500]
            same_name.ingest_status = "PENDING"
        return same_name, False

    if dry_run:
        return None, True

    uploaded_by = int(os.getenv("DEFAULT_INGEST_USER_ID", "1"))
    doc_code = make_document_code(pdf_path, collection_type, file_hash)
    if DocumentMetadata.query.filter_by(document_code=doc_code).first():
        doc_code = f"{doc_code[:42]}-{file_hash[:4].upper()}"

    row = DocumentMetadata(
        document_code=doc_code,
        document_title=clean_title,
        collection_type=collection_type,
        category_code=category_code,
        original_file_name=pdf_path.name[:255],
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


def process_one(pdf_path: Path, store: KnowledgeStore | None, args) -> str:
    file_hash = sha256_file(pdf_path)
    collection_type, vector_collection = resolve_folder_spec(pdf_path)
    logger.info("PDF 확인 [%s -> %s]: %s | SHA256=%s", collection_type, vector_collection, pdf_path.name, file_hash[:12])

    row, is_new = find_or_create_metadata(pdf_path, file_hash, dry_run=args.dry_run)

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
        logger.info("이미 인덱싱됨 (스킵): %s (ID=%s)", pdf_path.name, row.document_id)
        return "SKIPPED"

    if not args.dry_run:
        assert row is not None
        row.file_hash_sha256 = file_hash
        row.storage_uri = relative_path(pdf_path)
        row.ingest_status = "PENDING"
        db.session.commit()
        document_id = int(row.document_id)
    else:
        document_id = int(row.document_id) if row is not None else 0

    try:
        chunks, assets, _ = build_chunks_for_pdf(
            pdf_path,
            file_hash=file_hash,
            project_root=PROJECT_ROOT,
            semantic=not args.no_semantic,
            ocr_enabled=args.ocr,
            force_reparse=args.force_reparse,
        )

        if not chunks:
            chunks, assets, _ = build_chunks_for_pdf(
                pdf_path,
                file_hash=file_hash,
                project_root=PROJECT_ROOT,
                semantic=not args.no_semantic,
                ocr_enabled=True,
                force_reparse=True,
            )

        if not chunks:
            raise RuntimeError("추출된 텍스트 청크가 없습니다.")

        if args.dry_run:
            return "DRY_RUN"

        assert row is not None and store is not None
        count = store.replace_document_chunks(
            document_id=document_id,
            document_name=row.document_title,
            category=row.category_code,
            source=row.collection_type,
            revision="Rev.1",
            file_hash=file_hash,
            chunks=chunks,
            access_level=row.collection_type,
            document_status="ACTIVE" if row.is_active == "Y" else "INACTIVE",
        )
        row.ingest_status = "COMPLETED"
        db.session.commit()
        logger.info("적재 완료: ID=%s, chunks=%s, collection=%s", document_id, count, row.vector_collection)
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
    parser = argparse.ArgumentParser(description="internal_pdfs 및 raw_pdfs를 스캔하여 DOCUMENT_METADATA 및 ChromaDB 동기화")
    parser.add_argument("--pdf-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-reparse", action="store_true")
    parser.add_argument("--no-semantic", action="store_true")
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.pdf_dir:
        target_dirs = [Path(args.pdf_dir).resolve()]
    else:
        target_dirs = [
            PROJECT_ROOT / "data" / "internal_pdfs",
            PROJECT_ROOT / "data" / "raw_pdfs",
        ]

    pdf_files: list[Path] = []
    for d in target_dirs:
        if d.exists():
            pdf_files.extend(sorted(p for p in d.rglob("*.pdf") if p.is_file()))

    if args.limit > 0:
        pdf_files = pdf_files[:args.limit]

    web = create_web()
    summary = {"COMPLETED": 0, "SKIPPED": 0, "FAILED": 0, "DRY_RUN": 0}

    with web.app_context():
        store = None if args.dry_run else KnowledgeStore()
        deactivate_missing_file_rows(store, dry_run=args.dry_run)
        if not pdf_files:
            logger.warning("스캔된 PDF 파일이 없습니다: %s", [str(d) for d in target_dirs])
            return 0

        for idx, pdf_path in enumerate(pdf_files, start=1):
            logger.info("[%s/%s] 동기화 중: %s", idx, len(pdf_files), pdf_path.name)
            try:
                status = process_one(pdf_path, store, args)
                summary[status] = summary.get(status, 0) + 1
            except Exception as exc:
                db.session.rollback()
                summary["FAILED"] += 1
                logger.exception("실패: %s (%s)", pdf_path.name, exc)

        logger.info("최종 동기화 요약: %s", summary)
    return 1 if summary["FAILED"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
