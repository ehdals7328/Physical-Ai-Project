"""사내 PDF 파일과 Oracle 메타데이터, ChromaDB 색인을 동기화합니다."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from flask import current_app

from app.models.schema import db, DocumentMetadata
from config import Config
from modules.knowledge.rag_pipeline import build_chunks_for_pdf
from modules.knowledge.retriever import get_knowledge_store

logger = logging.getLogger(__name__)


class DuplicateDocumentError(ValueError):
    """같은 접근 등급으로 이미 등록된 PDF를 구분하는 예외입니다."""


def calculate_file_hash(file_path: Path) -> str:
    """파일을 스트리밍 방식으로 읽어 SHA-256 해시를 계산합니다."""
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_internal_pdf(file_path: Path, original_name: str, uploader_id: int) -> DocumentMetadata:
    """업로드된 PDF를 파싱·청킹·임베딩하고 사내 컬렉션에 색인합니다."""
    file_hash = calculate_file_hash(file_path)
    duplicate = DocumentMetadata.query.filter_by(
        file_hash=file_hash,
        access_level="INTERNAL",
        document_status="ACTIVE",
        ingest_status="COMPLETED",
    ).first()
    if duplicate:
        raise DuplicateDocumentError(f"동일한 PDF가 이미 등록되어 있습니다. 문서 ID: {duplicate.document_id}")

    metadata = DocumentMetadata(
        document_name=original_name[:300],
        category="사내문서",
        source="사내 관리자 업로드",
        revision="UNKNOWN",
        access_level="INTERNAL",
        file_path=str(file_path.resolve()),
        file_hash=file_hash,
        ingest_status="PROCESSING",
        document_status="ACTIVE",
        uploaded_by=uploader_id,
    )
    db.session.add(metadata)
    db.session.commit()

    try:
        chunks, _assets, cache_info = build_chunks_for_pdf(
            file_path,
            file_hash=file_hash,
            project_root=Path(current_app.root_path).parent,
            semantic=Config.INTERNAL_UPLOAD_SEMANTIC,
            ocr_enabled=False,
            force_reparse=False,
        )
        if not chunks:
            raise ValueError("PDF에서 검색 가능한 텍스트를 추출하지 못했습니다. 스캔 PDF는 OCR 처리가 필요합니다.")

        chunk_count = get_knowledge_store().replace_document_chunks(
            document_id=metadata.document_id,
            document_name=metadata.document_name,
            category=metadata.category,
            source=metadata.source or "사내 관리자 업로드",
            revision=metadata.revision or "UNKNOWN",
            file_hash=file_hash,
            chunks=chunks,
            access_level="INTERNAL",
            document_status="ACTIVE",
        )
        if chunk_count == 0:
            raise ValueError("색인할 텍스트 청크가 생성되지 않았습니다.")

        metadata.ingest_status = "COMPLETED"
        db.session.commit()
        logger.info(
            "사내 PDF 색인 완료: document_id=%s chunks=%s parsed_cache=%s chunk_cache=%s",
            metadata.document_id,
            chunk_count,
            cache_info.get("parsed_cache_hit"),
            cache_info.get("chunk_cache_hit"),
        )
        return metadata
    except Exception as exc:
        db.session.rollback()
        failed_metadata = db.session.get(DocumentMetadata, metadata.document_id)
        if failed_metadata:
            failed_metadata.ingest_status = "FAILED"
            db.session.commit()
        try:
            get_knowledge_store().delete_document_chunks(metadata.document_id, "INTERNAL")
        except Exception:
            logger.exception("실패한 사내 PDF의 ChromaDB 정리 실패: document_id=%s", metadata.document_id)
        raise


def delete_internal_document(document_id: int) -> bool:
    """사내 문서의 벡터 청크와 메타데이터 상태를 삭제 처리합니다."""
    metadata = db.session.get(DocumentMetadata, document_id)
    if metadata is None or metadata.access_level != "INTERNAL":
        return False

    get_knowledge_store().delete_document_chunks(document_id, "INTERNAL")
    metadata.document_status = "DELETED"
    db.session.commit()

    stored_path = Path(metadata.file_path) if metadata.file_path else None
    owned_folder = Path(Config.INTERNAL_PDF_FOLDER).resolve()
    if stored_path and stored_path.exists() and stored_path.resolve().is_relative_to(owned_folder):
        try:
            stored_path.unlink()
        except OSError:
            logger.exception("원본 PDF 파일 삭제 실패: %s", stored_path)
    return True


def list_internal_documents(limit: int = 100) -> list[DocumentMetadata]:
    """활성 사내 PDF를 최근 등록 순서로 조회합니다."""
    safe_limit = max(1, min(int(limit), 300))
    return (
        DocumentMetadata.query.filter_by(access_level="INTERNAL", document_status="ACTIVE")
        .order_by(DocumentMetadata.document_id.desc())
        .limit(safe_limit)
        .all()
    )
