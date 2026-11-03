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

CATEGORY_RULES = [
    ("안전보건", ("안전", "kosha", "osha", "z49", "safety", "health")),
    ("품질검사", ("품질", "검사", "inspection", "defect", "ut", "rt", "hull weld")),
    ("장비중량물", ("중량", "crane", "rigging", "lifting", "hoist", "인양")),
    ("작업환경기상", ("기상", "풍속", "습도", "weather", "environment", "wind")),
    ("선급법규", ("선급", "class rule", "class_rules", "kr rule", "korean register", "kr ", "imo", "goal-based", "b2.1")),
    ("사내문서", ("사내", "internal", "work instruction", "승인서")),
]

SOURCE_RULES = [
    (("aws",), "AWS (미국용접학회)"),
    (("kosha",), "안전보건공단(KOSHA)"),
    (("osha",), "OSHA"),
    (("twi",), "TWI (영국용접연구소)"),
    (("kiswel",), "KISWEL (고려용접봉)"),
    (("seah", "세아"), "세아ESAB"),
    (("imo",), "IMO"),
    (("kr ", "korean register", "한국선급"), "한국선급(KR)"),
]


# 공백과 경로 기호를 제거하고 소문자로 통일하여 파일명 비교 시 오차를 없애는 문자열 정규화 함수.
def normalize_name(value: str) -> str:
    value = (value or "").strip().casefold().replace("\\", "/")
    value = value.rsplit("/", 1)[-1]
    value = re.sub(r"\s+", "", value)
    return value


# 파일명과 상위 폴더 경로의 키워드를 미리 정의된 규칙표와 대조하여 문서의 카테고리와 출처(발행처)를 자동 추론하는 함수.
def guess_metadata(pdf_path: Path) -> tuple[str, str]:
    name = pdf_path.name.casefold()
    folders = " ".join(part.casefold() for part in pdf_path.parts)
    search_text = f"{folders} {name}"

    category = "제조기술"
    for candidate, keywords in CATEGORY_RULES:
        if any(keyword in search_text for keyword in keywords):
            category = candidate
            break

    source = "미확인"
    for keywords, candidate_source in SOURCE_RULES:
        if any(keyword in search_text for keyword in keywords):
            source = candidate_source
            break
    if source == "미확인":
        source = "조선소 기술자료실"
    return category, source


# 프로젝트 루트를 기준으로 상대 경로 문자열을 생성하여 운영체제 간 경로 호환성을 보장하는 유틸리티 함수.
def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


# 오라클 DB(DOCUMENT_METADATA)를 조회하여 기존 문서 레코드를 재활용하거나 새 메타데이터 행을 생성해 매핑하는 함수.
def find_or_create_metadata(pdf_path: Path, file_hash: str, *, dry_run: bool):
    docs = DocumentMetadata.query.all()

    exact = next((row for row in docs if (row.file_hash or "").strip().lower() == file_hash), None)
    if exact:
        return exact, False

    target_name = normalize_name(pdf_path.name)
    same_name = [row for row in docs if normalize_name(row.document_name) == target_name]

    reusable = next(
        (
            row for row in same_name
            if not (row.file_hash or "").strip()
            or str(row.file_path or "").strip().upper() == "SEED_DATA"
        ),
        None,
    )
    if reusable:
        if not dry_run:
            reusable.file_hash = file_hash
            reusable.file_path = relative_path(pdf_path)
            reusable.ingest_status = "QUEUED"
            reusable.updated_at = datetime.utcnow()
        return reusable, False

    category, source = guess_metadata(pdf_path)
    if dry_run:
        return None, True

    row = DocumentMetadata(
        document_name=pdf_path.name[:300],
        category=category,
        source=source,
        revision="UNKNOWN",
        access_level=os.getenv("DEFAULT_DOCUMENT_ACCESS_LEVEL", "INTERNAL").upper(),
        file_path=relative_path(pdf_path),
        file_hash=file_hash,
        ingest_status="QUEUED",
        document_status="ACTIVE",
        uploaded_by=None,
    )
    db.session.add(row)
    db.session.flush()
    return row, True


# 개별 PDF의 중복 여부를 검사하고, 파싱·청킹·임베딩을 거쳐 ChromaDB와 오라클 DB의 처리 상태를 일치시키는 핵심 단일 적재 처리 함수.
def process_one(pdf_path: Path, store: KnowledgeStore | None, args) -> str:
    file_hash = sha256_file(pdf_path)
    logger.info("PDF 확인: %s | SHA256=%s", pdf_path.name, file_hash[:16])

    row, is_new = find_or_create_metadata(pdf_path, file_hash, dry_run=args.dry_run)
    if row is not None and str(row.document_status or "ACTIVE").upper() != "ACTIVE":
        logger.warning("비활성 문서이므로 건너뜀: %s (DOCUMENT_ID=%s)", pdf_path.name, row.document_id)
        return "SKIPPED"

    if row is not None and store is not None and not args.force and not args.dry_run and store.is_document_indexed(
            row.document_id, file_hash):
        if row.ingest_status != "COMPLETED" and not args.dry_run:
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
            source=row.source or "미확인",
            revision=row.revision or "UNKNOWN",
            file_hash=file_hash,
            chunks=chunks,
            access_level=row.access_level or "INTERNAL",
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


# CLI 명령행 인자(--dry-run, --ocr, --force 등)를 파싱하고 대상 PDF들을 순회하며 일괄 적재 및 에러 롤백을 총괄하는 진입점 함수.
def main() -> int:
    parser = argparse.ArgumentParser(description="PDF를 BGE-M3 semantic chunking으로 ChromaDB에 증분 적재")
    parser.add_argument("--pdf-dir", default=str(PROJECT_ROOT / "data" / "raw_pdfs"))
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
            logger.info("현재 RAG_V2 청크 수: %s", store.count(current_index_only=True))

    return 1 if summary["FAILED"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
