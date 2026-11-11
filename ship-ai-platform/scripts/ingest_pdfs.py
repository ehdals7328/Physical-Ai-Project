# 이 파일은 원본 PDF 문서들을 스캔하여 관계형 DB(DOCUMENT_METADATA)와 벡터 스토어(ChromaDB) 양쪽에 파일 해시 기반으로 동기화하고,
# 캐시 검증을 통해 변경된 파일만 골라 적재하는 CLI 기반 RAG 증분 데이터 수집/배치 파이프라인(Ingest Controller)이다.

from __future__ import annotations

import argparse
import logging
import os
import re
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

# DDL CHECK (CATEGORY_CODE IN ('WELDING_SOP', 'SAFETY', 'CAD_DRAWING', 'CRANE_GUIDE')) 제약 준수
CATEGORY_RULES = [
    ("SAFETY", ("안전", "보건", "비상", "대응", "kosha", "osha", "z49", "safety", "health")),
    ("CRANE_GUIDE", ("중량", "crane", "rigging", "lifting", "hoist", "인양", "장비")),
    ("CAD_DRAWING", ("도면", "cad", "drawing", "블록", "배치도")),
    ("WELDING_SOP", ("용접", "품질", "검사", "인사", "근로", "임금", "보수", "휴일", "휴가", "성과", "윤리", "보안", "문서", "규정", "sop", "가이드", "운영")),
]


def normalize_name(value: str) -> str:
    """공백과 경로 기호를 제거하고 소문자로 통일하여 파일명 비교 시 오차를 없애는 함수."""
    value = (value or "").strip().casefold().replace("\\", "/")
    value = value.rsplit("/", 1)[-1]
    value = re.sub(r"\s+", "", value)
    return value


def guess_category_code(pdf_path: Path) -> str:
    """파일명과 경로를 분석하여 DDL 허용 CATEGORY_CODE로 변환하는 함수."""
    name = pdf_path.name.casefold()
    folders = " ".join(part.casefold() for part in pdf_path.parts)
    search_text = f"{folders} {name}"

    for category_code, keywords in CATEGORY_RULES:
        if any(keyword in search_text for keyword in keywords):
            return category_code
    return "WELDING_SOP"


def relative_path(path: Path) -> str:
    """프로젝트 루트를 기준으로 상대 경로 문자열을 생성하는 함수."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def make_document_code(pdf_path: Path, collection_type: str, file_hash: str) -> str:
    """DOCUMENT_CODE 유니크 제약조건(UK)을 만족하는 고유 코드를 생성하는 함수."""
    prefix = "DOC-INT" if collection_type == "INTERNAL" else "DOC-PUB"
    clean_stem = re.sub(r"[^A-Za-z0-9_-]", "", pdf_path.stem)[:20].upper()
    if not clean_stem:
        clean_stem = "DOC"
    return f"{prefix}-{clean_stem}-{file_hash[-6:].upper()}"[:50]


def find_or_create_metadata(pdf_path: Path, file_hash: str, *, dry_run: bool):
    """02_SHIP_USER_DDL.txt 스키마에 맞춰 메타데이터를 조회하거나 생성하는 함수."""
    docs = DocumentMetadata.query.all()

    # 1. 기존 파일 해시(FILE_HASH_SHA256) 일치 확인
    exact = next((row for row in docs if (row.file_hash_sha256 or "").strip().lower() == file_hash), None)
    if exact:
        return exact, False

    target_name = normalize_name(pdf_path.name)
    same_name = [row for row in docs if normalize_name(row.original_file_name or row.document_title) == target_name]

    # 2. 기존 레코드 중 해시가 비어있는 행 재활용
    reusable = next((row for row in same_name if not (row.file_hash_sha256 or "").strip()), None)
    if reusable:
        if not dry_run:
            reusable.file_hash_sha256 = file_hash
            reusable.storage_uri = relative_path(pdf_path)
            reusable.ingest_status = "PENDING"
        return reusable, False

    if dry_run:
        return None, True

    # 3. 신규 행 생성 (DDL 제약조건 충족)
    collection_type = os.getenv("DEFAULT_DOCUMENT_ACCESS_LEVEL", "INTERNAL").upper()
    if collection_type not in ("PUBLIC", "INTERNAL"):
        collection_type = "INTERNAL"

    category_code = guess_category_code(pdf_path)
    vector_collection = "col_internal_ship_sop" if collection_type == "INTERNAL" else "col_public_weld_std"
    doc_code = make_document_code(pdf_path, collection_type, file_hash)
    uploaded_by = int(os.getenv("DEFAULT_INGEST_USER_ID", "1"))

    row = DocumentMetadata(
        document_code=doc_code,
        document_title=pdf_path.name[:200],
        collection_type=collection_type,
        category_code=category_code,
        original_file_name=pdf_path.name[:255],
        storage_uri=relative_path(pdf_path)[:500],
        file_hash_sha256=file_hash[:64],
        vector_collection=vector_collection[:100],
        ingest_status="PENDING",
        uploaded_by=uploaded_by,
    )
    db.session.add(row)
    db.session.flush()
    return row, True


def process_one(pdf_path: Path, store: KnowledgeStore | None, args) -> str:
    """개별 PDF에 대해 파싱, 청킹, ChromaDB 적재 및 메타데이터 갱신을 수행하는 함수."""
    file_hash = sha256_file(pdf_path)
    logger.info("PDF 확인: %s | SHA256=%s", pdf_path.name, file_hash[:16])

    row, is_new = find_or_create_metadata(pdf_path, file_hash, dry_run=args.dry_run)

    # 이미 벡터스토어에 인덱싱되어 있는지 확인 (변경 없음 시 스킵)
    if row is not None and store is not None and not args.force and not args.dry_run and store.is_document_indexed(
            row.document_id, file_hash):
        if row.ingest_status != "COMPLETED" and not args.dry_run:
            row.ingest_status = "COMPLETED"
            db.session.commit()
        logger.info("변경 없음 건너뜀: %s (DOCUMENT_ID=%s)", pdf_path.name, row.document_id)
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
            document_name=row.document_title,
            category=row.category_code,
            source=row.collection_type,
            revision="Rev.1",
            file_hash=file_hash,
            chunks=chunks,
            access_level=row.collection_type,
            document_status="ACTIVE",
        )
        row.ingest_status = "COMPLETED"
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
        failed_row = db.session.get(DocumentMetadata, document_id) if document_id else None
        if failed_row is not None:
            failed_row.ingest_status = "FAILED"
            db.session.commit()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="PDF를 BGE-M3 semantic chunking으로 ChromaDB 및 Oracle에 적재")
    parser.add_argument("--pdf-dir", default=str(PROJECT_ROOT / "data" / "internal_pdfs"))
    parser.add_argument("--dry-run", action="store_true", help="Oracle/ChromaDB에 저장하지 않고 추출·청킹만 검사")
    parser.add_argument("--force", action="store_true", help="이미 적재된 PDF도 다시 임베딩")
    parser.add_argument("--force-reparse", action="store_true", help="PDF 파싱/청킹 캐시도 무시하고 다시 생성")
    parser.add_argument("--no-semantic", action="store_true", help="Semantic 경계 분석을 끄고 문단 기반 길이 청킹")
    parser.add_argument("--ocr", action="store_true", help="Tesseract OCR 사용 (kor+eng 언어 데이터 필요)")
    parser.add_argument("--limit", type=int, default=0, help="처리할 PDF 최대 개수 (0=전체)")
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

    logger.info("대상 PDF: %s개 | 폴더: %s", len(pdf_files), pdf_dir)
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
            logger.info("현재 RAG 청크 수: %s", store.count(current_index_only=True))

    return 1 if summary["FAILED"] else 0


if __name__ == "__main__":
    raise SystemExit(main())