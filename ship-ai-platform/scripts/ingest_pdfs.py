"""data/raw_pdfs 또는 data/internal_pdfs의 PDF를 선택한 RAG 컬렉션에 증분 색인한다."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from werkzeug.datastructures import FileStorage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import create_app
from app.models.ship_user import AppUser
from app.services.document_service import ingest_pdf, DuplicateDocumentError
from config import Config

def main() -> int:
    parser = argparse.ArgumentParser(description="2개 RAG 컬렉션에 PDF를 색인합니다.")
    parser.add_argument("--collection", choices=["PUBLIC", "INTERNAL"], default="PUBLIC")
    parser.add_argument("--directory", default=None, help="PDF 폴더 (생략 시 컬렉션별 기본 폴더)")
    parser.add_argument("--company-code", default="GLOBAL", help="INTERNAL 문서 소유 회사 코드 또는 GLOBAL")
    parser.add_argument("--category", default="01_제조기술")
    parser.add_argument("--source", default="출처 미기재")
    parser.add_argument("--revision", default="UNKNOWN")
    parser.add_argument("--uploader-id", type=int, default=None, help="업로더 USER_ID; 생략 시 SYSTEM 계정 미지정")
    args = parser.parse_args()
    folder = Path(args.directory or (Config.PUBLIC_PDF_FOLDER if args.collection == "PUBLIC" else Config.INTERNAL_PDF_FOLDER)).resolve()
    if not folder.exists():
        print(f"폴더가 없습니다: {folder}")
        return 2
    files = sorted(folder.rglob("*.pdf"))
    if not files:
        print(f"PDF가 없습니다: {folder}")
        return 0
    app = create_app()
    ok = skipped = failed = 0
    with app.app_context():
        uploader_id = args.uploader_id
        if uploader_id is None:
            uploader = AppUser.query.filter_by(role_code="SYS_ADMIN", is_active="Y").first()
            uploader_id = uploader.user_id if uploader else None
        for index, path in enumerate(files, 1):
            print(f"[{index}/{len(files)}] {path.name}")
            try:
                with path.open("rb") as source:
                    upload = FileStorage(stream=source, filename=path.name)
                    row = ingest_pdf(upload, collection_type=args.collection, category_code=path.parent.name or args.category,
                                     source=args.source, uploader_id=uploader_id,
                                     owner_company_code=args.company_code if args.collection == "INTERNAL" else "PUBLIC",
                                     revision_no=args.revision)
                print(f"  INDEXED document_id={row.document_id}")
                ok += 1
            except DuplicateDocumentError as exc:
                print(f"  SKIP {exc}")
                skipped += 1
            except Exception as exc:
                print(f"  FAILED {type(exc).__name__}: {exc}")
                failed += 1
    print(f"완료: indexed={ok}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())
