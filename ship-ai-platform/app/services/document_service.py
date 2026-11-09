from __future__ import annotations
import hashlib
import logging
import uuid
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import current_app
from app.extensions import db
from app.models.ship_user import RagDocument
from config import Config
from modules.knowledge.rag_pipeline import build_chunks_for_pdf
from modules.knowledge.retriever import get_knowledge_store

logger = logging.getLogger(__name__)

class DuplicateDocumentError(ValueError):
    pass

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def ingest_pdf(file_storage, *, collection_type: str, category_code: str, source: str,
               uploader_id: int | None, owner_company_code: str | None = None, revision_no: str = "UNKNOWN") -> RagDocument:
    collection_type = collection_type.upper()
    if collection_type not in {"PUBLIC", "INTERNAL"}:
        raise ValueError("collection_type은 PUBLIC 또는 INTERNAL이어야 합니다.")
    original_name = secure_filename(file_storage.filename or "")
    if not original_name or Path(original_name).suffix.lower() != ".pdf":
        raise ValueError("PDF 파일만 등록할 수 있습니다.")
    if collection_type == "INTERNAL" and not owner_company_code:
        owner_company_code = "GLOBAL"

    folder = Path(Config.PUBLIC_PDF_FOLDER if collection_type == "PUBLIC" else Config.INTERNAL_PDF_FOLDER)
    if collection_type == "INTERNAL" and owner_company_code != "GLOBAL":
        folder = folder / secure_filename(owner_company_code)
    folder.mkdir(parents=True, exist_ok=True)
    file_storage.stream.seek(0)
    temp_name = f"{uuid.uuid4().hex}_{original_name}"
    file_path = folder / temp_name
    file_storage.save(file_path)
    file_hash = sha256_file(file_path)

    duplicate = RagDocument.query.filter_by(file_hash_sha256=file_hash, collection_type=collection_type,
                                           owner_company_code=owner_company_code, document_status="ACTIVE").first()
    if duplicate:
        file_path.unlink(missing_ok=True)
        raise DuplicateDocumentError(f"동일한 문서가 이미 등록되어 있습니다. 문서 ID={duplicate.document_id}")

    collection_name = Config.PUBLIC_COLLECTION if collection_type == "PUBLIC" else Config.INTERNAL_COLLECTION
    row = RagDocument(
        document_code=f"DOC-{uuid.uuid4().hex[:16].upper()}", document_title=original_name[:300],
        collection_type=collection_type, category_code=(category_code or "GENERAL")[:50],
        original_file_name=original_name, storage_uri=str(file_path.resolve()), file_hash_sha256=file_hash,
        revision_no=(revision_no or "UNKNOWN")[:80], source_organization=(source or "").strip()[:200],
        owner_company_code=("PUBLIC" if collection_type == "PUBLIC" else owner_company_code),
        access_scope_code=("PUBLIC" if collection_type == "PUBLIC" else "COMPANY"),
        vector_collection=collection_name, ingest_status="PROCESSING", document_status="ACTIVE", uploaded_by=uploader_id,
    )
    db.session.add(row)
    db.session.commit()
    try:
        chunks, _assets, cache_info = build_chunks_for_pdf(
            file_path, file_hash=file_hash, project_root=Path(current_app.root_path).parent,
            semantic=Config.RAG_SEMANTIC, ocr_enabled=Config.RAG_OCR, force_reparse=False,
        )
        if not chunks:
            raise ValueError("검색 가능한 텍스트 청크가 없습니다. 스캔 PDF는 Tesseract OCR 설정을 확인하세요.")
        count = get_knowledge_store().upsert_document_chunks(
            document_id=row.document_id, document_name=row.document_title, category=row.category_code,
            source=row.source_organization or "출처 미기재", revision=row.revision_no or "UNKNOWN",
            file_hash=file_hash, chunks=chunks, collection_type=collection_type,
            owner_company_code=row.owner_company_code if collection_type == "INTERNAL" else None,
        )
        if count <= 0:
            raise ValueError("VectorDB에 저장된 청크가 없습니다.")
        row.ingest_status = "INGESTED"
        db.session.commit()
        logger.info("PDF indexed: id=%s chunks=%s parsed_cache=%s chunk_cache=%s", row.document_id, count,
                    cache_info.get("parsed_cache_hit"), cache_info.get("chunk_cache_hit"))
        return row
    except Exception as exc:
        db.session.rollback()
        failed = db.session.get(RagDocument, row.document_id)
        if failed:
            failed.ingest_status = "FAILED"
            db.session.commit()
        try:
            get_knowledge_store().delete_document_chunks(row.document_id, collection_type)
        except Exception:
            logger.exception("Failed to remove partially indexed vectors")
        raise exc

def deactivate_document(document_id: int) -> bool:
    row = db.session.get(RagDocument, int(document_id))
    if row is None or row.document_status != "ACTIVE":
        return False
    # DB 메타데이터와 벡터는 분산 저장소이므로 벡터 삭제 성공 후 비활성화한다.
    get_knowledge_store().delete_document_chunks(row.document_id, row.collection_type)
    row.document_status = "DELETED"
    row.ingest_status = "DELETED"
    db.session.commit()
    try:
        file_path = Path(row.storage_uri or "")
        owned_roots = [Path(Config.PUBLIC_PDF_FOLDER).resolve(), Path(Config.INTERNAL_PDF_FOLDER).resolve()]
        if file_path.exists() and any(file_path.resolve().is_relative_to(root) for root in owned_roots):
            file_path.unlink()
    except OSError:
        logger.exception("Could not delete PDF source: %s", row.storage_uri)
    return True
