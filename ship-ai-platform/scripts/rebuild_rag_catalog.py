"""기존 Chroma 컬렉션의 메타데이터를 새 RAG_DOCUMENTS 카탈로그로 이관한다.

ship_user를 재생성한 뒤 기존 Chroma 벡터를 유지하려는 경우 실행한다.
내부 문서의 회사 소유 범위를 알 수 없으므로 기본값은 LEGACY_REVIEW_REQUIRED이며,
SYS_ADMIN만 검색할 수 있다. 모든 내부 문서가 단일 회사 소유임이 확인된 경우에만
--internal-company-code COMPANY_CODE를 지정한다.
"""
from __future__ import annotations
import argparse
import hashlib
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import create_app
from app.extensions import db
from app.models.ship_user import RagDocument
from modules.knowledge.retriever import get_knowledge_store
from config import Config

def fetch_all(collection, page_size=500):
    offset=0
    while True:
        data=collection.get(include=["metadatas"], limit=page_size, offset=offset)
        ids=data.get("ids",[]); metas=data.get("metadatas",[])
        if not ids: break
        yield from zip(ids, metas)
        if len(ids)<page_size: break
        offset += page_size

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--internal-company-code", default=None,
        help="모든 기존 내부 문서가 이 회사 소유임을 확인한 경우에만 지정")
    parser.add_argument("--dry-run", action="store_true")
    args=parser.parse_args()
    app=create_app()
    with app.app_context():
        store=get_knowledge_store()
        summary={"PUBLIC":0,"INTERNAL":0,"SKIPPED":0}
        for collection_type,collection in (("PUBLIC",store.public),("INTERNAL",store.internal)):
            groups={}
            for chunk_id,meta in fetch_all(collection):
                meta=meta or {}
                old_id=meta.get("document_id")
                if old_id is None:
                    # document_id가 없는 자료는 이름 기준으로 묶되, 출처가 빈 경우 별도 그룹으로 둔다.
                    group_key=f"NOID:{meta.get('document_name','UNKNOWN')}:{meta.get('source','')}"
                else:
                    group_key=str(old_id)
                groups.setdefault(group_key,[]).append((chunk_id,dict(meta)))
            for old_id,rows in groups.items():
                first=rows[0][1]
                title=str(first.get("document_name") or f"LEGACY_{old_id}")[:300]
                category=str(first.get("category") or "LEGACY")[:50]
                source=str(first.get("source") or "기존 Chroma 문서")[:200]
                owner="PUBLIC" if collection_type=="PUBLIC" else (args.internal_company_code or "LEGACY_REVIEW_REQUIRED")
                code=f"LEGACY-{collection_type}-{hashlib.sha1(old_id.encode()).hexdigest()[:16].upper()}"
                existing=RagDocument.query.filter_by(document_code=code).first()
                if existing:
                    if not args.dry_run:
                        new_id=existing.document_id
                    else:
                        new_id=existing.document_id
                else:
                    digest=str(first.get("file_hash") or hashlib.sha256(f"{collection_type}|{old_id}|{title}".encode()).hexdigest())
                    if args.dry_run:
                        new_id=0
                    else:
                        row=RagDocument(document_code=code,document_title=title,collection_type=collection_type,
                            category_code=category,original_file_name=title,storage_uri=None,file_hash_sha256=digest,
                            revision_no=str(first.get("revision") or "LEGACY_UNKNOWN"),source_organization=source,
                            owner_company_code=owner,access_scope_code="PUBLIC" if collection_type=="PUBLIC" else "RESTRICTED",
                            vector_collection=Config.PUBLIC_COLLECTION if collection_type=="PUBLIC" else Config.INTERNAL_COLLECTION,
                            ingest_status="INGESTED",document_status="ACTIVE",uploaded_by=None)
                        db.session.add(row); db.session.flush(); new_id=row.document_id
                if not args.dry_run:
                    chunk_ids=[]; new_metas=[]
                    for chunk_id,meta in rows:
                        meta["document_id"]=int(new_id)
                        meta["owner_company_code"]=owner
                        meta["document_name"]=title
                        meta["category"]=category
                        meta["source"]=source
                        meta["revision"]=str(first.get("revision") or "LEGACY_UNKNOWN")
                        chunk_ids.append(chunk_id); new_metas.append(meta)
                    collection.update(ids=chunk_ids,metadatas=new_metas)
                    db.session.commit()
                summary[collection_type]+=1
            print(f"{collection_type}: {len(groups)}개 문서 그룹 발견")
        print("Catalog rebuild summary:",summary)
        if args.dry_run: print("DRY RUN - DB/Vector metadata 변경 없음")

if __name__=="__main__": main()
