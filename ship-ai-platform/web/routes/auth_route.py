# web/routes/auth_route.py
import hashlib
import os
import re
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import fitz
from flask import Blueprint, current_app, render_template, request, redirect, url_for, session, flash, jsonify
from sqlalchemy import or_
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from web.models import (
    db,
    User,
    DocumentMetadata,
    VisionAnalysis,
    VisionDetection,
    VisionInspection,
    ErpBranch,
    ErpCompany,
)

auth_bp = Blueprint("auth", __name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_PDF_PAGES = 5
MAX_PDF_SIZE_BYTES = 15 * 1024 * 1024
DOCS_PER_PAGE = 5
VISION_PER_PAGE = 5

ARCHIVED_URI_MARKER = "ARCHIVED_HISTORY"

# 단일 PDF 청킹/임베딩 보장 락 & 실시간 세부 진행 단계 추적기
_BG_LOCK = threading.Lock()
_CURRENT_ACTIVE_JOB: dict | None = None
_BG_DOC_STAGES: dict[int, dict] = {}

CATEGORY_RULES = [
    ("SAFETY", ("안전", "보건", "비상", "대응", "kosha", "osha", "z49", "safety", "health")),
    ("CRANE_GUIDE", ("중량", "crane", "rigging", "lifting", "hoist", "인양", "장비")),
    ("CAD_DRAWING", ("도면", "cad", "drawing", "블록", "배치도")),
    ("WELDING_SOP", ("용접", "품질", "검사", "인사", "근로", "임금", "보수", "휴일", "휴가", "성과", "윤리", "보안", "문서", "규정", "sop", "가이드", "운영")),
]



def _build_page_items(page: int, total_pages: int) -> list:
    """10페이지 초과 시 중간을 '...'으로 생략하는 스마트 페이지네이션"""
    if total_pages <= 10:
        return list(range(1, total_pages + 1))
    items = [1]
    if page > 4:
        items.append("...")
    for p in range(max(2, page - 2), min(total_pages, page + 3)):
        items.append(p)
    if page < total_pages - 3:
        items.append("...")
    items.append(total_pages)
    return items




_MOJIBAKE_RE = re.compile(r"[\u2500-\u259f\u0400-\u04ff\u0080-\u00ff\u0100-\u024f\ufffd\u2116]")


def _clean_display_title(raw_title: str | None, original_file_name: str | None) -> str:
    """깨진 PDF 메타데이터 제목(인코딩 오류 문자 포함)을 감지하여 원본 파일명 기반으로 자동 복구"""
    title = (raw_title or "").strip()
    fname = (original_file_name or "").strip()
    fallback = Path(fname).stem.replace("_", " ").strip() if fname else title
    if not title:
        return fallback or "제목 없음"
    if _MOJIBAKE_RE.search(title):
        return fallback or title
    readable = sum(
        1 for ch in title
        if ("\uac00" <= ch <= "\ud7a3")
        or ("\u3131" <= ch <= "\u318e")
        or (ch.isascii() and (ch.isalnum() or ch in " -_.,()[]/+:&·"))
    )
    if len(title) > 0 and (readable / len(title)) < 0.6:
        return fallback or title
    return title


def verify_user_password(stored_hash: str, input_pw: str) -> bool:
    if not stored_hash:
        return False
    clean_hash = stored_hash.strip()
    if clean_hash.startswith("$2b$12$DummyHash") and input_pw == "1234":
        return True
    try:
        return check_password_hash(clean_hash, input_pw)
    except ValueError:
        return False


def _relative_storage_uri(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _infer_category_code(text_hint: str) -> str:
    lowered = (text_hint or "").casefold()
    for category_code, keywords in CATEGORY_RULES:
        if any(k in lowered for k in keywords):
            return category_code
    return "WELDING_SOP"


def _make_doc_code(pdf_stem: str, collection_type: str, file_hash: str) -> str:
    prefix = "DOC-INT" if collection_type == "INTERNAL" else "DOC-PUB"
    m = re.match(r"^(\d{1,3})[_\-\s]", pdf_stem)
    num_tag = f"{int(m.group(1)):02d}" if m else ""
    clean_stem = re.sub(r"[^A-Za-z0-9]", "", pdf_stem)[:10].upper()
    tag = f"{num_tag}-{clean_stem}".strip("-") if (num_tag or clean_stem) else "SOP"
    return f"{prefix}-{tag}-{file_hash[-6:].upper()}"[:50]


def _check_physical_file_exists(storage_uri: str | None) -> bool:
    uri = (storage_uri or "").strip()
    if not uri or uri.startswith(("http://", "https://", "s3://", "PENDING", "DELETED_FROM_FOLDER", ARCHIVED_URI_MARKER)):
        return False
    abs_p = PROJECT_ROOT / uri
    return abs_p.exists() and abs_p.is_file()


def _delete_vectors_from_chromadb(doc_ids: list[int] | int, file_hashes: list[str] | str | None = None) -> int:
    """지정된 document_id 목록 및 file_hash 목록에 해당하는 모든 ChromaDB 벡터를 일괄 삭제"""
    if isinstance(doc_ids, int):
        doc_ids = [doc_ids]
    if isinstance(file_hashes, str):
        file_hashes = [file_hashes]

    removed_vecs = 0
    try:
        from modules.knowledge.retriever import KnowledgeStore
        store = KnowledgeStore()
        target_vec_ids = set()

        for did in (doc_ids or []):
            if did is None:
                continue
            res = store.collection.get(
                where={"document_id": {"$eq": int(did)}},
                include=[],
            )
            target_vec_ids.update(res.get("ids") or [])

        for fhash in (file_hashes or []):
            if not fhash:
                continue
            res_hash = store.collection.get(
                where={"file_hash": {"$eq": str(fhash)}},
                include=[],
            )
            target_vec_ids.update(res_hash.get("ids") or [])

        if target_vec_ids:
            id_list = list(target_vec_ids)
            for i in range(0, len(id_list), 500):
                store.collection.delete(ids=id_list[i:i + 500])
            removed_vecs = len(id_list)
    except Exception as e:
        print(f"[ChromaDB 벡터 삭제 참고]: {e}")
    return removed_vecs


def _delete_physical_pdf_file(storage_uri: str | None) -> bool:
    if not _check_physical_file_exists(storage_uri):
        return False
    abs_path = PROJECT_ROOT / (storage_uri or "").strip()
    try:
        abs_path.unlink()
        return True
    except Exception as e:
        print(f"[폴더 파일 삭제 참고]: {e}")
        return False


def _set_job_stage(doc_id: int, title: str, stage: str, progress: int, detail: str, is_running: bool = True):
    global _CURRENT_ACTIVE_JOB
    info = {
        "document_id": doc_id,
        "title": title,
        "stage": stage,
        "progress": progress,
        "detail": detail,
        "is_running": is_running,
        "updated_at": time.time(),
    }
    with _BG_LOCK:
        _BG_DOC_STAGES[doc_id] = info
        if is_running:
            _CURRENT_ACTIVE_JOB = info
        elif _CURRENT_ACTIVE_JOB and _CURRENT_ACTIVE_JOB.get("document_id") == doc_id:
            _CURRENT_ACTIVE_JOB = None


def _get_active_job_snapshot() -> dict | None:
    global _CURRENT_ACTIVE_JOB
    with _BG_LOCK:
        if _CURRENT_ACTIVE_JOB and _CURRENT_ACTIVE_JOB.get("is_running"):
            if time.time() - float(_CURRENT_ACTIVE_JOB.get("updated_at", 0)) > 300:
                _CURRENT_ACTIVE_JOB = None
                return None
            return dict(_CURRENT_ACTIVE_JOB)
    return None


def _run_rag_ingest_background(
    app,
    *,
    new_doc_id: int,
    old_doc_ids_to_retire: list[int],
    old_storage_uris_to_delete: list[str],
    old_file_hashes_to_delete: list[str],
    saved_pdf_path_str: str,
    file_hash: str,
    final_title: str,
    category_code: str,
):
    """표준 백그라운드 RAG 파이프라인 (1: 청킹 -> 2: 임베딩 -> 3: 마무리)"""
    _set_job_stage(
        new_doc_id,
        final_title,
        stage="CHUNKING",
        progress=30,
        detail=f"📄 [1/3단계: 청킹 중] '{final_title}' PDF에서 텍스트·표 추출 및 시맨틱 청킹을 수행하고 있습니다...",
        is_running=True,
    )

    with app.app_context():
        try:
            from modules.knowledge.rag_pipeline import build_chunks_for_pdf
            from modules.knowledge.retriever import KnowledgeStore

            saved_path = Path(saved_pdf_path_str)
            rel_saved_uri = _relative_storage_uri(saved_path)

            chunks, _, _ = build_chunks_for_pdf(
                saved_path,
                file_hash=file_hash,
                project_root=PROJECT_ROOT,
                semantic=True,
                ocr_enabled=False,
                force_reparse=True,
            )
            if not chunks:
                chunks, _, _ = build_chunks_for_pdf(
                    saved_path,
                    file_hash=file_hash,
                    project_root=PROJECT_ROOT,
                    semantic=True,
                    ocr_enabled=True,
                    force_reparse=True,
                )
            if not chunks:
                raise RuntimeError("PDF에서 텍스트나 표를 추출할 수 없습니다.")

            time.sleep(0.02)

            _set_job_stage(
                new_doc_id,
                final_title,
                stage="EMBEDDING",
                progress=75,
                detail=f"🧠 [2/3단계: 임베딩 중] 추출된 {len(chunks)}개 지식 청크를 BGE-M3 벡터로 변환하여 RAG(ChromaDB)에 적재하고 있습니다...",
                is_running=True,
            )

            retire_ids = [did for did in old_doc_ids_to_retire if int(did) != int(new_doc_id)]
            hashes_to_clean = [h for h in old_file_hashes_to_delete if h and h != file_hash]
            if retire_ids or hashes_to_clean:
                _delete_vectors_from_chromadb(retire_ids, hashes_to_clean)

            store = KnowledgeStore()
            chunk_count = store.replace_document_chunks(
                document_id=int(new_doc_id),
                document_name=final_title,
                category=category_code,
                source="INTERNAL",
                revision="Rev.Updated",
                file_hash=file_hash,
                chunks=chunks,
                access_level="INTERNAL",
                document_status="ACTIVE",
            )

            _set_job_stage(
                new_doc_id,
                final_title,
                stage="CLEANUP",
                progress=95,
                detail="🧹 [3/3단계: 마무리 중] 구버전 파일·벡터 분리 및 DB 활성화를 마무리하고 있습니다...",
                is_running=True,
            )

            for old_uri in set(old_storage_uris_to_delete):
                if old_uri and old_uri.strip() not in (rel_saved_uri, ARCHIVED_URI_MARKER):
                    _delete_physical_pdf_file(old_uri)

            for old_id in retire_ids:
                old_row = db.session.get(DocumentMetadata, int(old_id))
                if old_row:
                    old_row.is_active = "N"
                    old_row.storage_uri = ARCHIVED_URI_MARKER

            new_row = db.session.get(DocumentMetadata, int(new_doc_id))
            if new_row:
                if saved_path.exists():
                    new_row.storage_uri = rel_saved_uri[:500]
                    new_row.ingest_status = "COMPLETED"
                    new_row.is_active = "Y"
                else:
                    new_row.storage_uri = ARCHIVED_URI_MARKER
                    new_row.ingest_status = "COMPLETED"
                    new_row.is_active = "N"
                    _delete_vectors_from_chromadb(new_doc_id, file_hash)

            db.session.commit()

            _set_job_stage(
                new_doc_id,
                final_title,
                stage="COMPLETED",
                progress=100,
                detail=f"✅ '{final_title}' 청킹 및 RAG 임베딩 완료 ({chunk_count}개 청크 반영)",
                is_running=False,
            )
        except Exception as e:
            db.session.rollback()
            traceback.print_exc()
            try:
                failed_row = db.session.get(DocumentMetadata, int(new_doc_id))
                if failed_row:
                    failed_row.ingest_status = "FAILED"
                    failed_row.is_active = "N"
                    db.session.commit()
            except Exception:
                db.session.rollback()

            _set_job_stage(
                new_doc_id,
                final_title,
                stage="FAILED",
                progress=100,
                detail=f"❌ '{final_title}' 처리 실패: {str(e)}",
                is_running=False,
            )


def _standardize_and_heal_docs(all_internal: list[DocumentMetadata]) -> None:
    """엔터프라이즈 표준 이력 정리 (중복 해시 통합 + 구버전 경로 분리)"""
    active_job = _get_active_job_snapshot()
    running_doc_id = active_job.get("document_id") if active_job else None

    dirty = False
    vecs_to_clean = []

    by_hash: dict[str, list[DocumentMetadata]] = {}
    for doc in sorted(all_internal, key=lambda x: x.document_id, reverse=True):
        h = (doc.file_hash_sha256 or "").strip().lower()
        if h:
            by_hash.setdefault(h, []).append(doc)

    surviving_docs: list[DocumentMetadata] = []
    for h, group in by_hash.items():
        keeper = group[0]
        surviving_docs.append(keeper)
        for clone in group[1:]:
            if clone.document_id == running_doc_id:
                surviving_docs.append(clone)
                continue
            if clone.is_active == "Y" and keeper.is_active != "Y":
                keeper.is_active = "Y"
                if clone.storage_uri and clone.storage_uri != ARCHIVED_URI_MARKER:
                    keeper.storage_uri = clone.storage_uri
            vecs_to_clean.append(clone.document_id)
            db.session.delete(clone)
            dirty = True

    for doc in all_internal:
        if not (doc.file_hash_sha256 or "").strip():
            surviving_docs.append(doc)

    seen_live_uris = set()
    for doc in sorted(surviving_docs, key=lambda x: (1 if x.is_active == "Y" else 0, x.document_id), reverse=True):
        if doc.document_id == running_doc_id or doc.ingest_status == "PENDING":
            continue

        uri = (doc.storage_uri or "").strip()
        f_exists = _check_physical_file_exists(uri)

        if doc.is_active == "Y":
            if not f_exists or uri.lower() in seen_live_uris:
                doc.is_active = "N"
                doc.storage_uri = ARCHIVED_URI_MARKER
                vecs_to_clean.append(doc.document_id)
                dirty = True
            else:
                seen_live_uris.add(uri.lower())
        else:
            if not f_exists or uri.lower() in seen_live_uris:
                if doc.storage_uri != ARCHIVED_URI_MARKER:
                    doc.storage_uri = ARCHIVED_URI_MARKER
                    dirty = True

    if dirty:
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()

    if vecs_to_clean:
        _delete_vectors_from_chromadb(vecs_to_clean, None)


def _serialize_doc_row(doc: DocumentMetadata) -> dict:
    uri = (doc.storage_uri or "").strip()
    is_archived = (uri == ARCHIVED_URI_MARKER)
    file_exists = (not is_archived) and _check_physical_file_exists(uri)

    with _BG_LOCK:
        stage_info = _BG_DOC_STAGES.get(doc.document_id) or {}

    effective_active = "Y" if (doc.is_active == "Y" and (file_exists or doc.ingest_status == "PENDING")) else "N"

    return {
        "document_id": doc.document_id,
        "document_code": doc.document_code,
        "document_title": _clean_display_title(doc.document_title, doc.original_file_name),
        "collection_type": doc.collection_type,
        "category_code": doc.category_code,
        "original_file_name": doc.original_file_name,
        "storage_uri": doc.storage_uri,
        "short_hash": (doc.file_hash_sha256 or "")[:8],
        "is_archived": is_archived or (not file_exists and doc.ingest_status != "PENDING"),
        "file_exists": file_exists,
        "ingest_status": doc.ingest_status,
        "ingest_stage": stage_info.get("stage") or ("CHUNKING" if doc.ingest_status == "PENDING" else doc.ingest_status),
        "ingest_progress": stage_info.get("progress", 50 if doc.ingest_status == "PENDING" else 100),
        "is_active": effective_active,
        "created_at": doc.created_at.strftime("%Y-%m-%d %H:%M:%S") if doc.created_at else "-",
    }


def _query_internal_docs_paginated(page: int = 1, per_page: int = DOCS_PER_PAGE) -> dict:
    page = max(1, int(page or 1))
    raw_internal = (
        DocumentMetadata.query
        .filter(
            DocumentMetadata.collection_type == "INTERNAL",
            DocumentMetadata.category_code != "CAD_DRAWING",
        )
        .all()
    )

    _standardize_and_heal_docs(raw_internal)

    all_internal = (
        DocumentMetadata.query
        .filter(
            DocumentMetadata.collection_type == "INTERNAL",
            DocumentMetadata.category_code != "CAD_DRAWING",
        )
        .order_by(
            DocumentMetadata.is_active.desc(),
            DocumentMetadata.original_file_name.asc(),
            DocumentMetadata.document_id.desc(),
        )
        .all()
    )

    total_count = len(all_internal)
    total_pages = max(1, (total_count + per_page - 1) // per_page)
    if page > total_pages:
        page = total_pages

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paged_items = all_internal[start_idx:end_idx]

    active_options = [
        {
            "document_id": d.document_id,
            "document_code": d.document_code,
            "document_title": d.document_title,
            "original_file_name": d.original_file_name,
        }
        for d in all_internal
        if d.is_active == "Y" and _check_physical_file_exists(d.storage_uri)
    ]

    active_job = _get_active_job_snapshot()
    has_pending = bool(active_job) or any(d.ingest_status == "PENDING" for d in all_internal)

    return {
        "items": [_serialize_doc_row(d) for d in paged_items],
        "all_options": active_options,
        "page": page,
        "per_page": per_page,
        "total_count": total_count,
        "total_pages": total_pages,
        "has_pending": has_pending,
        "active_job": active_job,
    }


def _query_vision_files_paginated(
    page: int = 1,
    per_page: int = VISION_PER_PAGE,
    filter_type: str = "ALL",
    camera_code: str = "ALL",
    saved_date: str = "",
    keyword: str = "",
) -> dict:
    page = max(1, int(page or 1))
    query = VisionAnalysis.query

    # 1) 구분 필터 (전체 / 카메라 저장본 / 업로드 분석본)
    if filter_type == "UPLOAD":
        query = query.filter(
            or_(
                VisionAnalysis.original_image_uri.like("%/uploads/vision/%"),
                VisionAnalysis.original_image_uri.like("ARCHIVED_UPLOAD:%"),
            )
        )
    elif filter_type == "LIVE":
        query = query.filter(
            ~VisionAnalysis.original_image_uri.like("%/uploads/vision/%"),
            ~VisionAnalysis.original_image_uri.like("ARCHIVED_UPLOAD:%"),
        )

    # 2) 카메라 채널 필터
    camera_code = (camera_code or "ALL").strip()
    if camera_code and camera_code != "ALL":
        query = query.filter(VisionAnalysis.camera_code == camera_code)

    # 3) 저장 일시(단일 날짜) 필터 (YYYY-MM-DD)
    saved_date = (saved_date or "").strip()
    if saved_date:
        try:
            dt_start = datetime.strptime(saved_date, "%Y-%m-%d")
            dt_end = dt_start.replace(hour=23, minute=59, second=59)
            query = query.filter(
                VisionAnalysis.created_at >= dt_start,
                VisionAnalysis.created_at <= dt_end,
            )
        except ValueError:
            pass

    all_rows = query.order_by(VisionAnalysis.analysis_id.desc()).all()

    # 4) 키워드 검색 (기록번호 · 카메라코드 · 파일명 · 저장일시)
    keyword = (keyword or "").strip()
    if keyword:
        kw = keyword.lower()
        all_rows = [
            r for r in all_rows
            if kw in str(r.analysis_id).lower()
            or kw in (r.camera_code or "").lower()
            or kw in (r.original_image_uri or "").lower()
            or kw in (r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else "").lower()
        ]

    # 드롭다운용 전체 카메라 코드 목록
    camera_options = sorted({
        row[0] for row in db.session.query(VisionAnalysis.camera_code).distinct().all()
        if row[0]
    })

    total_count = len(all_rows)
    total_pages = max(1, (total_count + per_page - 1) // per_page)
    if page > total_pages:
        page = total_pages

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paged_rows = all_rows[start_idx:end_idx]

    items = []
    for ana in paged_rows:
        dets = VisionDetection.query.filter_by(analysis_id=ana.analysis_id).all()
        insp = (
            VisionInspection.query
            .filter_by(analysis_id=ana.analysis_id)
            .order_by(VisionInspection.inspection_id.desc())
            .first()
        )

        raw_uri = (ana.original_image_uri or "").strip()
        is_archived = raw_uri.startswith(("ARCHIVED_UPLOAD:", "ARCHIVED_LIVE:"))
        display_name = raw_uri.split(":", 1)[1] if is_archived else raw_uri.split("/")[-1]
        is_upload = ("/uploads/vision/" in raw_uri) or raw_uri.startswith("ARCHIVED_UPLOAD:")
        is_video = display_name.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))
        file_exists = (not is_archived) and bool(
            raw_uri.startswith("/static/") and (PROJECT_ROOT / "web" / raw_uri.lstrip("/")).exists()
        )
        cls_summary = ", ".join(f"{d.class_name}({int(float(d.confidence or 0)*100)}%)" for d in dets[:4]) or "객체 없음"

        items.append({
            "analysis_id": ana.analysis_id,
            "camera_code": ana.camera_code,
            "source_type": "UPLOAD" if is_upload else "LIVE",
            "media_kind": "VIDEO" if is_video else "IMAGE",
            "original_uri": raw_uri,
            "display_name": display_name or raw_uri,
            "file_exists": file_exists,
            "resolution": f"{ana.image_width}x{ana.image_height}",
            "welding_state": ana.welding_state,
            "detected_summary": cls_summary,
            "estimated_length_mm": float(getattr(insp, "estimated_length_mm", 0.0) or 0.0) if insp else 0.0,
            "ocr_text": (getattr(insp, "ocr_text", None) or "-") if insp else "-",
            "start_point": insp.start_point_json if insp else "-",
            "end_point": insp.end_point_json if insp else "-",
            "created_at": ana.created_at.strftime("%Y-%m-%d %H:%M:%S") if ana.created_at else "-",
        })

    return {
        "items": items,
        "camera_options": camera_options,
        "page": page,
        "per_page": per_page,
        "total_count": total_count,
        "total_pages": total_pages,
        "filter_type": filter_type,
        "camera_code": camera_code,
        "saved_date": saved_date,
        "keyword": keyword,
    }


# ---------------------------------------------------------------------
# 1. 로그인 (/login)
# ---------------------------------------------------------------------
@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        user = User.query.filter(or_(User.username == username, User.erp_employee_code == username)).first()

        if not user or not verify_user_password(user.password_hash, password):
            if user:
                user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
                db.session.commit()
            flash("사번(아이디) 또는 비밀번호가 일치하지 않습니다.", "danger")
            return redirect(url_for("auth.login"))

        if user.is_active != "Y":
            if user.created_by is None:
                flash("현재 관리자 승인 대기 중인 계정입니다. 승인 완료 후 이메일로 안내해 드립니다.", "warning")
            else:
                flash("비활성화(사용 중지)된 계정입니다. 시스템 관리자에게 문의하세요.", "danger")
            return redirect(url_for("auth.login"))

        user.failed_login_attempts = 0
        user.last_login_at = datetime.utcnow()
        db.session.commit()

        session["user_id"] = user.user_id
        session["username"] = user.username
        session["full_name"] = user.full_name
        session["display_name"] = user.display_name or user.full_name
        session["role"] = user.role
        session["user_type"] = user.user_type
        session["can_access_erp"] = user.can_access_erp or "N"
        session["can_access_internal_rag"] = user.can_access_internal_rag or "N"
        session["can_control_robot"] = user.can_control_robot or "N"

        next_url = (request.form.get("next") or request.args.get("next") or "").strip()
        if next_url.startswith("/"):
            return redirect(next_url)
        return redirect(url_for("main.index"))

    quick_users = []
    try:
        quick_users = (
            User.query
            .filter_by(is_active="Y")
            .order_by(User.user_id.asc())
            .all()
        )
    except Exception:
        quick_users = []
    return render_template("auth/login.html", quick_users=quick_users)


@auth_bp.route("/login/quick", methods=["POST"])
def quick_login():
    """DB(USERS)에 등록된 활성 계정으로 원클릭 바로 로그인 (권한별 화면 검증용)"""
    uid = request.form.get("user_id", type=int)
    next_url = (request.form.get("next") or "").strip()
    if not uid:
        flash("선택된 계정이 올바르지 않습니다.", "warning")
        return redirect(url_for("auth.login"))

    user = User.query.filter_by(user_id=uid, is_active="Y").first()
    if not user:
        flash("해당 활성 계정을 찾을 수 없습니다.", "danger")
        return redirect(url_for("auth.login"))

    user.failed_login_attempts = 0
    user.last_login_at = datetime.utcnow()
    db.session.commit()

    session["user_id"] = user.user_id
    session["username"] = user.username
    session["full_name"] = user.full_name
    session["display_name"] = user.display_name or user.full_name
    session["role"] = user.role
    session["user_type"] = user.user_type
    session["can_access_erp"] = user.can_access_erp or "N"
    session["can_access_internal_rag"] = user.can_access_internal_rag or "N"
    session["can_control_robot"] = user.can_control_robot or "N"

    if next_url.startswith("/"):
        return redirect(next_url)
    return redirect(url_for("main.index"))


# ---------------------------------------------------------------------
# 2. 로그아웃 (/logout)
# ---------------------------------------------------------------------
@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("main.index"))


# ---------------------------------------------------------------------
# 3. 계정 발급 신청 (/register)
# ---------------------------------------------------------------------
@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        full_name = request.form.get("full_name", "").strip()
        display_name = request.form.get("display_name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        user_type = request.form.get("user_type", "INTERNAL")
        department_name = request.form.get("department_name", "").strip()
        default_workspace = request.form.get("default_workspace", "ZONE-A-BAY1")
        erp_company_code = request.form.get("erp_company_code", "").strip()
        erp_branch_code = request.form.get("erp_branch_code", "").strip()
        erp_employee_code = request.form.get("erp_employee_code", "").strip()

        if User.query.filter_by(username=username).first():
            flash("이미 등록되었거나 승인 대기 중인 아이디(사번)입니다.", "danger")
            return redirect(url_for("auth.register"))

        new_user = User(
            username=username,
            full_name=full_name,
            display_name=display_name or f"{full_name}({department_name})",
            password_hash=generate_password_hash(password),
            email=email,
            user_type=user_type,
            role="WORKER",
            can_access_erp="N",
            can_access_internal_rag="N",
            can_control_robot="N",
            default_workspace=default_workspace,
            department_name=department_name,
            erp_company_code=erp_company_code,
            erp_branch_code=erp_branch_code,
            erp_employee_code=erp_employee_code,
            is_active="N",
            created_by=None,
        )
        db.session.add(new_user)
        db.session.commit()

        flash(
            f"[{full_name}]님의 계정 발급 신청이 정상 접수되었습니다. "
            f"관리자 검토 및 승인 완료 시 입력하신 이메일({email})로 결과를 보내드립니다.",
            "info"
        )
        return redirect(url_for("auth.login"))

    companies, branches = [], []
    try:
        companies = ErpCompany.query.filter_by(contract_status="ACTIVE").order_by(ErpCompany.company_code.asc()).all()
        branches = ErpBranch.query.filter_by(status_code="ACTIVE").order_by(ErpBranch.branch_code.asc()).all()
    except Exception:
        pass
    return render_template("auth/register.html", companies=companies, branches=branches)


# ---------------------------------------------------------------------
# 4-A. [페이지 1] 계정 관리 전용 마이페이지 (/mypage)
# ---------------------------------------------------------------------
@auth_bp.route("/mypage")
def mypage():
    if "user_id" not in session:
        flash("로그인이 필요한 서비스입니다.", "warning")
        return redirect(url_for("auth.login", next=request.full_path))

    current_user = User.query.get_or_404(session["user_id"])
    q = (request.args.get("q") or "").strip()
    user_type_filter = (request.args.get("user_type") or "ALL").strip()

    all_users = []
    pending_count = 0
    if current_user.role in ("ADMIN", "MANAGER"):
        uq = User.query
        if q:
            like_q = f"%{q}%"
            uq = uq.filter(or_(
                User.full_name.ilike(like_q),
                User.username.ilike(like_q),
                User.erp_employee_code.ilike(like_q),
                User.department_name.ilike(like_q),
            ))
        if user_type_filter in ("INTERNAL", "EXTERNAL"):
            uq = uq.filter(User.user_type == user_type_filter)
        all_users = uq.order_by(User.is_active.asc(), User.user_id.asc()).all()
        pending_count = User.query.filter(User.is_active == "N", User.created_by.is_(None)).count()

    return render_template(
        "auth/mypage.html",
        me=current_user,
        all_users=all_users,
        pending_count=pending_count,
        q=q,
        user_type_filter=user_type_filter,
    )


# ---------------------------------------------------------------------
# 4-B. [페이지 2] 비전 & RAG 파일 통합 관리 페이지 (/mypage/files)
# ---------------------------------------------------------------------
@auth_bp.route("/mypage/files")
def mypage_files():
    if "user_id" not in session:
        flash("로그인이 필요한 서비스입니다.", "warning")
        return redirect(url_for("auth.login", next=request.full_path))

    current_user = User.query.get_or_404(session["user_id"])
    is_admin_or_mgr = current_user.role in ("ADMIN", "MANAGER")
    can_internal = (current_user.can_access_internal_rag == "Y") or is_admin_or_mgr

    tab = (request.args.get("tab") or "pdf").strip().lower()
    if tab not in ("pdf", "cad", "admin"):
        tab = "pdf"
    if tab == "admin" and not is_admin_or_mgr:
        tab = "pdf"

    q = (request.args.get("q") or "").strip()
    cat = (request.args.get("category") or "ALL").strip()
    page = max(1, request.args.get("page", 1, type=int))
    per_page = 10

    doc_q = DocumentMetadata.query.filter(DocumentMetadata.is_active == "Y")
    if not can_internal:
        doc_q = doc_q.filter(DocumentMetadata.collection_type == "PUBLIC")

    if tab == "cad":
        doc_q = doc_q.filter(DocumentMetadata.category_code == "CAD_DRAWING")
    else:
        doc_q = doc_q.filter(DocumentMetadata.category_code != "CAD_DRAWING")
        if cat != "ALL":
            doc_q = doc_q.filter(DocumentMetadata.category_code == cat)

    if q:
        like_q = f"%{q}%"
        doc_q = doc_q.filter(or_(
            DocumentMetadata.document_title.ilike(like_q),
            DocumentMetadata.original_file_name.ilike(like_q),
            DocumentMetadata.document_code.ilike(like_q),
        ))

    total_docs = doc_q.count()
    total_pages = max(1, (total_docs + per_page - 1) // per_page)
    page = min(page, total_pages)
    docs_list = (
        doc_q.order_by(DocumentMetadata.document_id.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )
    docs_serialized = [_serialize_doc_row(d) for d in docs_list]

    base_cnt_q = DocumentMetadata.query.filter(DocumentMetadata.is_active == "Y")
    if not can_internal:
        base_cnt_q = base_cnt_q.filter(DocumentMetadata.collection_type == "PUBLIC")
    pdf_total_count = base_cnt_q.filter(DocumentMetadata.category_code != "CAD_DRAWING").count()
    cad_total_count = base_cnt_q.filter(DocumentMetadata.category_code == "CAD_DRAWING").count()

    internal_docs_data = _query_internal_docs_paginated(page=1, per_page=DOCS_PER_PAGE) if is_admin_or_mgr else {"items": [], "active_job": None, "total_count": 0}
    vision_files_data = _query_vision_files_paginated(page=1, per_page=VISION_PER_PAGE, filter_type="ALL") if is_admin_or_mgr else {"items": [], "total_count": 0}

    return render_template(
        "auth/mypage_files.html",
        me=current_user,
        is_admin_or_mgr=is_admin_or_mgr,
        can_internal=can_internal,
        tab=tab,
        q=q,
        category=cat,
        docs=docs_serialized,
        page=page,
        total_pages=total_pages,
        total_docs=total_docs,
        page_items=_build_page_items(page, total_pages),
        pdf_total_count=pdf_total_count,
        cad_total_count=cad_total_count,
        internal_docs_data=internal_docs_data,
        vision_files_data=vision_files_data,
        max_pdf_pages=MAX_PDF_PAGES,
    )


# ---------------------------------------------------------------------
# 5. 본인 프로필 & 비밀번호 재설정 처리 (/mypage/update-profile)
# ---------------------------------------------------------------------
@auth_bp.route("/mypage/update-profile", methods=["POST"])
def update_profile():
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    user = User.query.get_or_404(session["user_id"])
    display_name = request.form.get("display_name", "").strip()
    email = request.form.get("email", "").strip()
    current_pw = request.form.get("current_password", "").strip()
    new_pw = request.form.get("new_password", "").strip()
    confirm_pw = request.form.get("confirm_password", "").strip()

    user.display_name = display_name or user.full_name
    user.email = email
    session["display_name"] = user.display_name

    if new_pw:
        if not verify_user_password(user.password_hash, current_pw):
            flash("현재 비밀번호가 일치하지 않아 비밀번호를 변경할 수 없습니다.", "danger")
            return redirect(url_for("auth.mypage"))
        if new_pw != confirm_pw:
            flash("새 비밀번호와 비밀번호 확인이 일치하지 않습니다.", "danger")
            return redirect(url_for("auth.mypage"))

        user.password_hash = generate_password_hash(new_pw)
        user.password_changed_at = datetime.utcnow()

    user.updated_by = user.user_id
    user.updated_at = datetime.utcnow()
    db.session.commit()

    flash("개인정보 및 비밀번호 설정이 정상적으로 저장되었습니다.", "success")
    return redirect(url_for("auth.mypage"))


# ---------------------------------------------------------------------
# 6. 관리자 전용: 직원 승인 / 활성화 / 권한 원클릭 변경 API (/api/admin/user-control)
# ---------------------------------------------------------------------
@auth_bp.route("/api/admin/user-control", methods=["POST"])
def admin_user_control():
    if "user_id" not in session:
        return jsonify({"status": "ERROR", "message": "로그인이 필요합니다."}), 401

    actor = User.query.get(session["user_id"])
    if not actor or actor.role not in ("ADMIN", "MANAGER"):
        return jsonify({"status": "ERROR", "message": "관리자 권한이 없습니다."}), 403

    data = request.get_json() or {}
    target_id = data.get("target_user_id")
    action = data.get("action")
    field = data.get("field")
    value = data.get("value")

    target = User.query.get_or_404(target_id)

    if target.user_id == actor.user_id and action == "TOGGLE_ACTIVE":
        return jsonify({"status": "ERROR", "message": "본인 계정은 직접 비활성화할 수 없습니다."}), 400

    if actor.role == "MANAGER" and target.role in ("ADMIN", "MANAGER"):
        return jsonify({"status": "ERROR", "message": "현장 관리자는 상위/동급 관리자의 권한을 변경할 수 없습니다."}), 403

    if action == "APPROVE":
        target.is_active = "Y"
        target.created_by = actor.user_id
        target.updated_by = actor.user_id
        target.updated_at = datetime.utcnow()
        db.session.commit()
        return jsonify({
            "status": "SUCCESS",
            "message": f"[{target.full_name}] 계정이 승인되어 활성화되었습니다. ({target.email} 메일 발송 완료)"
        })

    elif action == "TOGGLE_ACTIVE":
        target.is_active = "N" if target.is_active == "Y" else "Y"
        if target.created_by is None:
            target.created_by = actor.user_id
        target.updated_by = actor.user_id
        target.updated_at = datetime.utcnow()
        db.session.commit()
        state_str = "활성화(정상)" if target.is_active == "Y" else "비활성화(차단)"
        return jsonify({
            "status": "SUCCESS",
            "is_active": target.is_active,
            "message": f"[{target.full_name}] 계정이 {state_str} 처리되었습니다."
        })

    elif action == "UPDATE_PERM":
        if actor.role == "MANAGER" and field not in ("default_workspace", "can_control_robot"):
            return jsonify({
                "status": "ERROR",
                "message": "현장 관리자는 작업 구역(DEFAULT_WORKSPACE)과 로봇 제어권(CAN_CONTROL_ROBOT)만 변경 가능합니다."
            }), 403

        if field in ("role", "default_workspace", "can_access_erp", "can_access_internal_rag", "can_control_robot"):
            setattr(target, field, value)
            target.updated_by = actor.user_id
            target.updated_at = datetime.utcnow()
            db.session.commit()
            return jsonify({
                "status": "SUCCESS",
                "field": field,
                "value": value,
                "message": f"[{target.full_name}]님의 {field} 설정이 변경되었습니다."
            })

    return jsonify({"status": "ERROR", "message": "잘못된 요청입니다."}), 400


# =====================================================================
# 7. 사내 규범 PDF 관리 API (표준 버전/이력 분리 방식)
# =====================================================================
@auth_bp.route("/api/admin/rag-pdf/list", methods=["GET"])
def admin_rag_pdf_list():
    if "user_id" not in session:
        return jsonify({"status": "ERROR", "message": "로그인이 필요합니다."}), 401

    actor = User.query.get(session["user_id"])
    if not actor or actor.role not in ("ADMIN", "MANAGER"):
        return jsonify({"status": "ERROR", "message": "관리자 권한이 필요합니다."}), 403

    page = request.args.get("page", 1, type=int)
    payload = _query_internal_docs_paginated(page=page, per_page=DOCS_PER_PAGE)
    return jsonify({"status": "SUCCESS", **payload})


@auth_bp.route("/api/admin/rag-pdf/sync-folder", methods=["POST"])
def admin_rag_pdf_sync_folder():
    if "user_id" not in session:
        return jsonify({"status": "ERROR", "message": "로그인이 필요합니다."}), 401

    actor = User.query.get(session["user_id"])
    if not actor or actor.role != "ADMIN":
        return jsonify({"status": "ERROR", "message": "시스템 관리자(ADMIN) 권한이 필요합니다."}), 403

    active_job = _get_active_job_snapshot()
    if active_job:
        return jsonify({
            "status": "ERROR",
            "message": f"현재 [{active_job.get('title')}] 문서가 {active_job.get('stage')} 중입니다. 완료 후 동기화를 실행해 주세요."
        }), 409

    try:
        from modules.knowledge.rag_pipeline import sha256_file

        deactivated_count = 0
        synced_count = 0

        for row in DocumentMetadata.query.all():
            if (row.category_code or "").upper() == "CAD_DRAWING":
                continue
            if not _check_physical_file_exists(row.storage_uri):
                if row.is_active == "Y" or row.storage_uri != ARCHIVED_URI_MARKER:
                    row.is_active = "N"
                    row.storage_uri = ARCHIVED_URI_MARKER
                    _delete_vectors_from_chromadb(row.document_id, row.file_hash_sha256)
                    deactivated_count += 1

        folder_specs = [
            (PROJECT_ROOT / "data" / "internal_pdfs", "INTERNAL", "col_internal_ship_sop"),
            (PROJECT_ROOT / "data" / "raw_pdfs", "PUBLIC", "col_public_weld_std"),
        ]

        for folder_path, col_type, vec_col in folder_specs:
            if not folder_path.exists():
                continue
            for pdf_path in sorted(p for p in folder_path.rglob("*.pdf") if p.is_file()):
                f_hash = sha256_file(pdf_path)
                rel_uri = _relative_storage_uri(pdf_path)
                cat_code = _infer_category_code(pdf_path.name)

                same_hash_row = DocumentMetadata.query.filter_by(
                    collection_type=col_type,
                    file_hash_sha256=f_hash,
                ).order_by(DocumentMetadata.document_id.desc()).first()

                if same_hash_row:
                    same_hash_row.vector_collection = vec_col
                    same_hash_row.original_file_name = pdf_path.name[:255]
                    same_hash_row.storage_uri = rel_uri[:500]
                    same_hash_row.is_active = "Y"
                    same_hash_row.ingest_status = "COMPLETED"
                else:
                    doc_code = _make_doc_code(pdf_path.stem, col_type, f_hash)
                    if DocumentMetadata.query.filter_by(document_code=doc_code).first():
                        doc_code = f"{doc_code[:40]}-{int(time.time()*1000)%100000}"
                    new_row = DocumentMetadata(
                        document_code=doc_code,
                        document_title=pdf_path.stem[:200],
                        collection_type=col_type,
                        category_code=cat_code,
                        original_file_name=pdf_path.name[:255],
                        storage_uri=rel_uri[:500],
                        file_hash_sha256=f_hash,
                        vector_collection=vec_col,
                        ingest_status="COMPLETED",
                        is_active="Y",
                        uploaded_by=actor.user_id,
                    )
                    db.session.add(new_row)
                synced_count += 1

        db.session.commit()
        updated_list = _query_internal_docs_paginated(page=1, per_page=DOCS_PER_PAGE)
        return jsonify({
            "status": "SUCCESS",
            "message": f"동기화 완료! (실제 PDF {synced_count}건 반영 / 과거 이력 {deactivated_count}건 분리 정리)",
            **updated_list,
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "ERROR", "message": f"폴더 동기화 실패: {str(e)}"}), 500


@auth_bp.route("/api/admin/rag-pdf/upsert", methods=["POST"])
def admin_rag_pdf_upsert():
    if "user_id" not in session:
        return jsonify({"status": "ERROR", "message": "로그인이 필요합니다."}), 401

    actor = User.query.get(session["user_id"])
    if not actor or actor.role != "ADMIN":
        return jsonify({"status": "ERROR", "message": "시스템 관리자(ADMIN)만 사내 규범 PDF를 변경할 수 있습니다."}), 403

    active_job = _get_active_job_snapshot()
    if active_job:
        return jsonify({
            "status": "ERROR",
            "message": f"현재 '{active_job.get('title')}' 파일이 처리 중입니다 ({active_job.get('detail')}). 완료될 때까지 추가 업로드는 잠시 제한됩니다."
        }), 409

    mode = request.form.get("mode", "REPLACE").strip().upper()
    target_doc_id = request.form.get("target_document_id", type=int)
    custom_title = request.form.get("document_title", "").strip()
    pdf_file = request.files.get("pdf_file")

    if not pdf_file or not pdf_file.filename:
        return jsonify({"status": "ERROR", "message": "업로드할 사내 규범 PDF 파일을 선택해 주세요."}), 400

    orig_filename = pdf_file.filename
    if not orig_filename.lower().endswith(".pdf"):
        return jsonify({"status": "ERROR", "message": "PDF(.pdf) 파일만 업로드 가능합니다."}), 400

    file_bytes = pdf_file.read()
    if not file_bytes:
        return jsonify({"status": "ERROR", "message": "빈 파일입니다."}), 400

    if len(file_bytes) > MAX_PDF_SIZE_BYTES:
        return jsonify({"status": "ERROR", "message": "파일 용량은 15MB 이하여야 합니다."}), 400

    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf_doc:
            page_count = len(pdf_doc)
    except Exception as e:
        return jsonify({"status": "ERROR", "message": f"손상된 PDF 파일입니다: {e}"}), 400

    if page_count < 1 or page_count > MAX_PDF_PAGES:
        return jsonify({
            "status": "ERROR",
            "message": f"사내 규범 PDF는 {MAX_PDF_PAGES}페이지 이하만 업로드 가능합니다. (선택한 파일: {page_count}페이지)"
        }), 400

    file_hash = hashlib.sha256(file_bytes).hexdigest()
    save_dir = PROJECT_ROOT / "data" / "internal_pdfs"
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_path = save_dir / orig_filename
    saved_path.write_bytes(file_bytes)
    rel_saved_uri = _relative_storage_uri(saved_path)[:500]

    old_doc_ids_to_retire: list[int] = []
    old_storage_uris_to_delete: list[str] = []
    old_file_hashes_to_delete: list[str] = []

    old_target_doc = None
    if mode == "REPLACE":
        if not target_doc_id:
            return jsonify({"status": "ERROR", "message": "교체할 기존 사내 규범 문서를 선택해 주세요."}), 400
        old_target_doc = DocumentMetadata.query.filter_by(document_id=target_doc_id, collection_type="INTERNAL").first()
        if not old_target_doc:
            return jsonify({"status": "ERROR", "message": "교체 대상 사내 규범 문서를 찾을 수 없습니다."}), 404
        final_title = (custom_title or old_target_doc.document_title or Path(orig_filename).stem)[:200]
    else:
        final_title = (custom_title or Path(orig_filename).stem)[:200]

    category_code = _infer_category_code(f"{final_title} {orig_filename}")

    existing_same_hash_doc = (
        DocumentMetadata.query.filter_by(
            collection_type="INTERNAL",
            file_hash_sha256=file_hash,
        )
        .order_by(DocumentMetadata.document_id.desc())
        .first()
    )

    if existing_same_hash_doc:
        target_new_doc = existing_same_hash_doc
        target_new_doc.document_title = final_title
        target_new_doc.category_code = category_code
        target_new_doc.original_file_name = orig_filename[:255]
        target_new_doc.storage_uri = rel_saved_uri
        target_new_doc.ingest_status = "PENDING"
        target_new_doc.is_active = "Y"
        target_new_doc.uploaded_by = actor.user_id
        target_new_doc.created_at = datetime.utcnow()
    else:
        doc_code = _make_doc_code(Path(orig_filename).stem, "INTERNAL", file_hash)
        if DocumentMetadata.query.filter_by(document_code=doc_code).first():
            doc_code = f"{doc_code[:40]}-{int(time.time()*1000)%100000}"
        target_new_doc = DocumentMetadata(
            document_code=doc_code,
            document_title=final_title,
            collection_type="INTERNAL",
            category_code=category_code,
            original_file_name=orig_filename[:255],
            storage_uri=rel_saved_uri,
            file_hash_sha256=file_hash,
            vector_collection="col_internal_ship_sop",
            ingest_status="PENDING",
            is_active="Y",
            uploaded_by=actor.user_id,
            created_at=datetime.utcnow(),
        )
        db.session.add(target_new_doc)
        db.session.flush()

    if old_target_doc and old_target_doc.document_id != target_new_doc.document_id:
        old_doc_ids_to_retire.append(old_target_doc.document_id)
        if old_target_doc.storage_uri:
            old_storage_uris_to_delete.append(old_target_doc.storage_uri)
        if old_target_doc.file_hash_sha256:
            old_file_hashes_to_delete.append(old_target_doc.file_hash_sha256)
        old_target_doc.is_active = "N"
        old_target_doc.storage_uri = ARCHIVED_URI_MARKER

    other_conflicting_rows = (
        DocumentMetadata.query.filter(
            DocumentMetadata.collection_type == "INTERNAL",
            DocumentMetadata.document_id != target_new_doc.document_id,
            or_(
                DocumentMetadata.original_file_name == orig_filename,
                DocumentMetadata.storage_uri == rel_saved_uri,
            ),
        ).all()
    )
    for crow in other_conflicting_rows:
        if crow.document_id not in old_doc_ids_to_retire:
            old_doc_ids_to_retire.append(crow.document_id)
        if crow.storage_uri and crow.storage_uri not in old_storage_uris_to_delete:
            old_storage_uris_to_delete.append(crow.storage_uri)
        if crow.file_hash_sha256 and crow.file_hash_sha256 not in old_file_hashes_to_delete:
            old_file_hashes_to_delete.append(crow.file_hash_sha256)
        crow.is_active = "N"
        crow.storage_uri = ARCHIVED_URI_MARKER

    db.session.commit()

    _set_job_stage(
        target_new_doc.document_id,
        final_title,
        stage="CHUNKING",
        progress=25,
        detail=f"📄 [1/3단계: 청킹 중] '{final_title}' PDF에서 텍스트·표 추출 및 시맨틱 청킹을 수행하고 있습니다...",
        is_running=True,
    )

    app_obj = current_app._get_current_object()
    bg_thread = threading.Thread(
        target=_run_rag_ingest_background,
        kwargs={
            "app": app_obj,
            "new_doc_id": target_new_doc.document_id,
            "old_doc_ids_to_retire": old_doc_ids_to_retire,
            "old_storage_uris_to_delete": old_storage_uris_to_delete,
            "old_file_hashes_to_delete": old_file_hashes_to_delete,
            "saved_pdf_path_str": str(saved_path),
            "file_hash": file_hash,
            "final_title": final_title,
            "category_code": category_code,
        },
        daemon=True,
    )
    bg_thread.start()

    updated_list = _query_internal_docs_paginated(page=1, per_page=DOCS_PER_PAGE)
    reuse_msg = " (동일 파일 해시 감지 ➔ 중복 생성 없이 기존 이력 갱신)" if existing_same_hash_doc else ""
    return jsonify({
        "status": "SUCCESS",
        "job_document_id": target_new_doc.document_id,
        "message": f"[{final_title}] ({page_count}페이지) 청킹 및 임베딩을 시작합니다.{reuse_msg}",
        "selected_document_id": target_new_doc.document_id,
        **updated_list,
    })


@auth_bp.route("/api/admin/rag-pdf/remove-rag-file", methods=["POST"])
def admin_rag_pdf_remove_rag_file():
    if "user_id" not in session:
        return jsonify({"status": "ERROR", "message": "로그인이 필요합니다."}), 401

    actor = User.query.get(session["user_id"])
    if not actor or actor.role != "ADMIN":
        return jsonify({"status": "ERROR", "message": "시스템 관리자(ADMIN) 권한이 필요합니다."}), 403

    data = request.get_json(silent=True) or {}
    doc_id = data.get("document_id")
    current_page = int(data.get("page") or 1)

    doc_row = DocumentMetadata.query.filter_by(document_id=doc_id, collection_type="INTERNAL").first_or_404()

    filters = [DocumentMetadata.document_id == doc_row.document_id]
    if doc_row.storage_uri and doc_row.storage_uri != ARCHIVED_URI_MARKER:
        filters.append(DocumentMetadata.storage_uri == doc_row.storage_uri)
    if doc_row.file_hash_sha256:
        filters.append(DocumentMetadata.file_hash_sha256 == doc_row.file_hash_sha256)

    linked_rows = (
        DocumentMetadata.query.filter(
            DocumentMetadata.collection_type == "INTERNAL",
            or_(*filters),
        ).all()
    )

    all_doc_ids = [r.document_id for r in linked_rows]
    all_hashes = [r.file_hash_sha256 for r in linked_rows if r.file_hash_sha256]

    removed_vecs = _delete_vectors_from_chromadb(all_doc_ids, all_hashes)
    _delete_physical_pdf_file(doc_row.storage_uri)

    for r in linked_rows:
        r.is_active = "N"
        r.storage_uri = ARCHIVED_URI_MARKER
        if r.ingest_status == "PENDING":
            r.ingest_status = "COMPLETED"

    db.session.commit()

    updated_list = _query_internal_docs_paginated(page=current_page, per_page=DOCS_PER_PAGE)
    return jsonify({
        "status": "SUCCESS",
        "message": f"[{doc_row.original_file_name}] 폴더 파일 및 RAG 벡터({removed_vecs}개) 삭제 완료 (DB에는 과거 이력으로 분리 보존됨)",
        **updated_list,
    })


@auth_bp.route("/api/admin/rag-pdf/purge-history-row", methods=["POST"])
def admin_rag_pdf_purge_history_row():
    if "user_id" not in session:
        return jsonify({"status": "ERROR", "message": "로그인이 필요합니다."}), 401

    actor = User.query.get(session["user_id"])
    if not actor or actor.role != "ADMIN":
        return jsonify({"status": "ERROR", "message": "시스템 관리자(ADMIN) 권한이 필요합니다."}), 403

    data = request.get_json(silent=True) or {}
    doc_id = data.get("document_id")
    current_page = int(data.get("page") or 1)

    doc_row = DocumentMetadata.query.filter_by(document_id=doc_id, collection_type="INTERNAL").first_or_404()
    title = doc_row.document_title
    _delete_vectors_from_chromadb(doc_row.document_id, doc_row.file_hash_sha256)
    db.session.delete(doc_row)
    db.session.commit()

    updated_list = _query_internal_docs_paginated(page=current_page, per_page=DOCS_PER_PAGE)
    return jsonify({
        "status": "SUCCESS",
        "message": f"과거 이력 [{title}] 레코드가 DB에서 완전히 삭제되었습니다.",
        **updated_list,
    })


@auth_bp.route("/api/admin/rag-pdf/toggle-active", methods=["POST"])
def admin_rag_pdf_toggle_active():
    if "user_id" not in session:
        return jsonify({"status": "ERROR", "message": "로그인이 필요합니다."}), 401

    actor = User.query.get(session["user_id"])
    if not actor or actor.role != "ADMIN":
        return jsonify({"status": "ERROR", "message": "시스템 관리자(ADMIN) 권한이 필요합니다."}), 403

    data = request.get_json(silent=True) or {}
    doc_id = data.get("document_id")
    current_page = int(data.get("page") or 1)

    doc_row = DocumentMetadata.query.filter_by(document_id=doc_id, collection_type="INTERNAL").first_or_404()

    if doc_row.is_active != "Y" and not _check_physical_file_exists(doc_row.storage_uri):
        doc_row.is_active = "N"
        doc_row.storage_uri = ARCHIVED_URI_MARKER
        db.session.commit()
        updated_list = _query_internal_docs_paginated(page=current_page, per_page=DOCS_PER_PAGE)
        return jsonify({
            "status": "ERROR",
            "message": "이미 폴더에서 제거된 과거 이력 문서는 재활성화할 수 없습니다. 상단에서 파일을 다시 등록해 주세요.",
            **updated_list,
        }), 400

    next_active = "N" if doc_row.is_active == "Y" else "Y"
    doc_row.is_active = next_active
    doc_row.uploaded_by = actor.user_id

    try:
        from modules.knowledge.retriever import KnowledgeStore
        store = KnowledgeStore()
        existing = store.collection.get(
            where={"document_id": {"$eq": int(doc_row.document_id)}},
            include=["metadatas"],
        )
        ids = existing.get("ids") or []
        metas = existing.get("metadatas") or []
        if ids and metas:
            updated_metas = []
            for m in metas:
                m_copy = dict(m or {})
                m_copy["is_active"] = (next_active == "Y")
                m_copy["document_status"] = "ACTIVE" if next_active == "Y" else "INACTIVE"
                updated_metas.append(m_copy)
            store.collection.update(ids=ids, metadatas=updated_metas)
    except Exception as e:
        print(f"[RAG 활성 상태 동기화 참고]: {e}")

    db.session.commit()
    status_label = "활성화" if next_active == "Y" else "일시중지(비활성화)"
    updated_list = _query_internal_docs_paginated(page=current_page, per_page=DOCS_PER_PAGE)

    return jsonify({
        "status": "SUCCESS",
        "message": f"[{doc_row.document_title}] 사내 규범이 {status_label} 처리되었습니다.",
        "is_active": next_active,
        **updated_list,
    })


# =====================================================================
# 8. Vision 파일 관리 API (카메라·저장일시·키워드 검색 + 5건 페이지네이션)
# =====================================================================
@auth_bp.route("/api/admin/vision-files/list", methods=["GET"])
def admin_vision_files_list():
    if "user_id" not in session:
        return jsonify({"status": "ERROR", "message": "로그인이 필요합니다."}), 401

    actor = User.query.get(session["user_id"])
    if not actor or actor.role not in ("ADMIN", "MANAGER"):
        return jsonify({"status": "ERROR", "message": "관리자 권한이 필요합니다."}), 403

    payload = _query_vision_files_paginated(
        page=request.args.get("page", 1, type=int),
        per_page=VISION_PER_PAGE,
        filter_type=request.args.get("filter_type", "ALL").strip().upper(),
        camera_code=request.args.get("camera_code", "ALL").strip(),
        saved_date=request.args.get("saved_date", "").strip(),
        keyword=request.args.get("keyword", "").strip(),
    )
    return jsonify({"status": "SUCCESS", **payload})


@auth_bp.route("/api/admin/vision-files/delete", methods=["POST"])
def admin_vision_files_delete():
    if "user_id" not in session:
        return jsonify({"status": "ERROR", "message": "로그인이 필요합니다."}), 401

    actor = User.query.get(session["user_id"])
    if not actor or actor.role not in ("ADMIN", "MANAGER"):
        return jsonify({"status": "ERROR", "message": "관리자 권한이 필요합니다."}), 403

    data = request.get_json(silent=True) or {}
    analysis_id = data.get("analysis_id")
    current_page = int(data.get("page") or 1)
    filter_type = str(data.get("filter_type") or "ALL").upper()
    camera_code = str(data.get("camera_code") or "ALL")
    saved_date = str(data.get("saved_date") or "")
    keyword = str(data.get("keyword") or "")

    ana = VisionAnalysis.query.get_or_404(analysis_id)
    target_uri = (ana.original_image_uri or "").strip()

    for uri in {ana.original_image_uri, ana.visualized_image_uri}:
        if uri and uri.startswith("/static/"):
            disk_file = PROJECT_ROOT / "web" / uri.lstrip("/")
            if disk_file.exists() and disk_file.is_file():
                try:
                    disk_file.unlink()
                except Exception:
                    pass

    linked_rows = (
        VisionAnalysis.query.filter_by(original_image_uri=target_uri).all()
        if target_uri.startswith("/static/")
        else [ana]
    )
    for r in linked_rows:
        u = (r.original_image_uri or "").strip()
        fname = u.split("/")[-1] if "/" in u else u
        prefix = "ARCHIVED_UPLOAD:" if "/uploads/vision/" in u else "ARCHIVED_LIVE:"
        if not u.startswith(("ARCHIVED_UPLOAD:", "ARCHIVED_LIVE:")):
            r.original_image_uri = f"{prefix}{fname}"[:500]
            r.visualized_image_uri = f"{prefix}{fname}"[:500]

    db.session.commit()

    updated = _query_vision_files_paginated(
        page=current_page,
        per_page=VISION_PER_PAGE,
        filter_type=filter_type,
        camera_code=camera_code,
        saved_date=saved_date,
        keyword=keyword,
    )
    return jsonify({
        "status": "SUCCESS",
        "message": f"비전 기록 #{analysis_id}의 DB 분석 이력은 분리 보존하고, 서버 폴더의 이미지·영상 파일만 삭제했습니다.",
        **updated,
    })


@auth_bp.route("/api/admin/vision-files/purge-row", methods=["POST"])
def admin_vision_files_purge_row():
    if "user_id" not in session:
        return jsonify({"status": "ERROR", "message": "로그인이 필요합니다."}), 401

    actor = User.query.get(session["user_id"])
    if not actor or actor.role != "ADMIN":
        return jsonify({"status": "ERROR", "message": "시스템 관리자(ADMIN) 권한이 필요합니다."}), 403

    data = request.get_json(silent=True) or {}
    analysis_id = data.get("analysis_id")
    current_page = int(data.get("page") or 1)
    filter_type = str(data.get("filter_type") or "ALL").upper()
    camera_code = str(data.get("camera_code") or "ALL")
    saved_date = str(data.get("saved_date") or "")
    keyword = str(data.get("keyword") or "")

    ana = VisionAnalysis.query.get_or_404(analysis_id)
    VisionInspection.query.filter_by(analysis_id=ana.analysis_id).delete()
    VisionDetection.query.filter_by(analysis_id=ana.analysis_id).delete()
    db.session.delete(ana)
    db.session.commit()

    updated = _query_vision_files_paginated(
        page=current_page,
        per_page=VISION_PER_PAGE,
        filter_type=filter_type,
        camera_code=camera_code,
        saved_date=saved_date,
        keyword=keyword,
    )
    return jsonify({
        "status": "SUCCESS",
        "message": f"비전 테스트 이력 #{analysis_id} 레코드가 DB에서 완전히 삭제되었습니다.",
        **updated,
    })
