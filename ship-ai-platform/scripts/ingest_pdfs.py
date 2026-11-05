"""PDF 원본을 Oracle 문서 메타데이터와 PUBLIC/INTERNAL RAG에 증분 색인합니다."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app  # noqa: E402
from app.models.schema import db, DocumentMetadata  # noqa: E402
from modules.knowledge.rag_pipeline import build_chunks_for_pdf, sha256_file  # noqa: E402
from modules.knowledge.retriever import KnowledgeStore  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag.ingest")


def relative_path(path: Path) -> str:
    """프로젝트 내부 파일은 상대 경로로, 외부 파일은 절대 경로로 반환합니다."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def find_or_create_metadata(
    pdf_path: Path,
    file_hash: str,
    *,
    dry_run: bool,
    access_level: str,
):
    """활성 파일 해시가 일치하는 문서 메타데이터를 재사용하거나 새 행을 생성합니다."""
    existing = DocumentMetadata.query.filter_by(
        file_hash=file_hash,
        access_level=access_level.upper(),
        document_status="ACTIVE",
    ).first()
    if existing:
        return existing, False

    if dry_run:
        return None, True

    row = DocumentMetadata(
        document_name=pdf_path.name[:300],
        category="기술자료",
        source="출처 미기재",
        revision="UNKNOWN",
        access_level=access_level.upper(),
        file_path=relative_path(pdf_path),
        file_hash=file_hash,
        ingest_status="QUEUED",
        document_status="ACTIVE",
        uploaded_by=None,
    )
    db.session.add(row)
    db.session.flush()
    return row, True


def process_one(pdf_path: Path, store: KnowledgeStore | None, args) -> str:
    """PDF 한 건의 해시 확인·파싱·청킹·임베딩·색인과 상태 갱신을 수행합니다."""
    file_hash = sha256_file(pdf_path)
    logger.info("PDF 확인: %s | SHA256=%s", pdf_path.name, file_hash[:16])

    row, is_new = find_or_create_metadata(
        pdf_path,
        file_hash,
        dry_run=args.dry_run,
        access_level=args.access_level,
    )
    if row is not None and str(row.document_status or "ACTIVE").upper() != "ACTIVE":
        logger.warning("비활성 문서이므로 건너뜀: %s (DOCUMENT_ID=%s)", pdf_path.name, row.document_id)
        return "SKIPPED"

    if row is not None:
        row.access_level = args.access_level

    if (
        row is not None
        and store is not None
        and not args.force
        and not args.dry_run
        and store.is_document_indexed(row.document_id, file_hash, args.access_level)
    ):
        row.ingest_status = "COMPLETED"
        row.updated_at = datetime.utcnow()
        db.session.commit()
        logger.info("변경 없음 건너뜀: %s (DOCUMENT_ID=%s)", pdf_path.name, row.document_id)
        return "SKIPPED"

    if not args.dry_run:
        assert row is not None
        row.file_hash = file_hash
        row.file_path = relative_path(pdf_path)
        row.ingest_status = "PROCESSING"
        row.updated_at = datetime.utcnow()
        db.session.commit()
        document_id = int(row.document_id)
    else:
        document_id = int(row.document_id) if row is not None else 0

    try:
        chunks, assets, cache_info = build_chunks_for_pdf(
            pdf_path,
            file_hash=file_hash,
            project_root=PROJECT_ROOT,
            semantic=not args.no_semantic,
            ocr_enabled=args.ocr,
            force_reparse=args.force_reparse,
        )
        logger.info(
            "추출 완료: blocks=%s, chunks=%s, assets=%s, parsed_cache=%s, chunk_cache=%s",
            cache_info.get("block_count", 0), len(chunks), len(assets),
            cache_info.get("parsed_cache_hit"), cache_info.get("chunk_cache_hit"),
        )

        if not chunks:
            raise RuntimeError(
                "검색 가능한 텍스트 청크가 없습니다. 스캔 PDF이면 Tesseract OCR 설치 후 --ocr 옵션으로 다시 실행하세요."
            )

        if args.dry_run:
            logger.info("DRY-RUN: %s -> %s chunks, %s extracted assets", pdf_path.name, len(chunks), len(assets))
            return "DRY_RUN"

        assert row is not None
        assert store is not None
        count = store.replace_document_chunks(
            document_id=document_id,
            document_name=row.document_name,
            category=row.category,
            source=row.source or "출처 미기재",
            revision=row.revision or "UNKNOWN",
            file_hash=file_hash,
            chunks=chunks,
            access_level=args.access_level,
            document_status=row.document_status or "ACTIVE",
        )
        row.ingest_status = "COMPLETED"
        row.updated_at = datetime.utcnow()
        db.session.commit()
        logger.info(
            "적재 완료: DOCUMENT_ID=%s, chunks=%s, assets=%s, new_document=%s",
            document_id, count, len(assets), is_new,
        )
        return "COMPLETED"
    except Exception:
        if args.dry_run:
            raise
        db.session.rollback()
        failed_row = db.session.get(DocumentMetadata, document_id)
        if failed_row is not None:
            failed_row.ingest_status = "FAILED"
            failed_row.updated_at = datetime.utcnow()
            db.session.commit()
        raise


def main() -> int:
    """명령행 옵션을 확인하고 지정된 PDF 폴더의 증분 색인을 실행합니다."""
    parser = argparse.ArgumentParser(description="PDF를 PUBLIC 또는 INTERNAL ChromaDB 컬렉션에 적재")
    parser.add_argument("--pdf-dir", default=str(PROJECT_ROOT / "data" / "raw_pdfs"))
    parser.add_argument("--dry-run", action="store_true", help="DB/벡터 저장 없이 파싱·청킹만 검사")
    parser.add_argument("--force", action="store_true", help="기존 색인도 다시 임베딩")
    parser.add_argument("--force-reparse", action="store_true", help="PDF 파싱/청킹 캐시 무시")
    parser.add_argument("--no-semantic", action="store_true", help="의미 기반 분할을 끄고 기본 청킹 사용")
    parser.add_argument("--ocr", action="store_true", help="Tesseract OCR 사용 (kor+eng 언어 데이터 필요)")
    parser.add_argument("--limit", type=int, default=0, help="처리할 PDF 최대 개수 (0=전체)")
    parser.add_argument("--access-level", choices=("PUBLIC", "INTERNAL"), default="PUBLIC")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir).resolve()
    if not pdf_dir.exists():
        logger.error("PDF 폴더가 없습니다: %s", pdf_dir)
        return 2

    pdf_files = sorted(path for path in pdf_dir.rglob("*.pdf") if path.is_file())
    if args.limit > 0:
        pdf_files = pdf_files[:args.limit]
    if not pdf_files:
        logger.warning("처리할 PDF가 없습니다: %s", pdf_dir)
        return 0

    logger.info("대상 PDF: %s개 | 폴더: %s | access_level=%s", len(pdf_files), pdf_dir, args.access_level)
    app = create_app()
    summary = {"COMPLETED": 0, "SKIPPED": 0, "FAILED": 0, "DRY_RUN": 0}

    with app.app_context():
        store = None if args.dry_run else KnowledgeStore()
        for index, pdf_path in enumerate(pdf_files, start=1):
            logger.info("[%s/%s] 시작: %s", index, len(pdf_files), pdf_path.name)
            try:
                status = process_one(pdf_path, store, args)
                summary[status] = summary.get(status, 0) + 1
            except Exception as exc:
                db.session.rollback()
                summary["FAILED"] += 1
                logger.exception("처리 실패: %s (%s)", pdf_path.name, exc)

        logger.info("처리 결과: %s", summary)
        if store is not None:
            logger.info("현재 PUBLIC=%s / INTERNAL=%s 청크", store.count(), store.count_internal())

    return 1 if summary["FAILED"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
