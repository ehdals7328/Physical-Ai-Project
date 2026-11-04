"""
document_tools.py
에이전트가 발행한 구조화 문서를 Markdown/Audit 파일로 저장하는 표준 도구 모음
"""
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from langchain_core.tools import tool

OUTPUT_DIR = Path("./generated_docs")

@tool
def export_work_order_file(document_data: Dict[str, Any], file_prefix: str = "work_doc") -> str:
    """
    공식 작업지시서, 중량물 계획서 또는 규격 분석 결과를 표준 Markdown 파일로 저장합니다.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = OUTPUT_DIR / f"{file_prefix}_{timestamp}.md"

    title = document_data.get("title") or document_data.get("document_title") or "표준 선박 작업 지침서"
    params = document_data.get("parameters") or document_data.get("extracted_params") or {}
    content = document_data.get("content") or ""
    weather = document_data.get("weather") or {}

    lines = [
        f"# {title}",
        f"\n**문서 발행 일시:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**문서 관리 번호:** DOC-{timestamp}\n",
        "---",
        "## 1. 확정 작업 제원 및 엔지니어링 조건"
    ]

    if params:
        for k, v in params.items():
            lines.append(f"- **{k}:** {v}")
    else:
        lines.append("- *기본 표준 조건 준용*")

    if weather:
        lines.extend([
            "\n## 2. 현장 환경 기상 조건",
            f"- **위치:** {weather.get('location', '-')}",
            f"- **기온:** {weather.get('temperature', '-')} ℃",
            f"- **풍속:** {weather.get('wind_speed', '-')} m/s",
            f"- **상태:** {weather.get('condition', '-')}"
        ])

    lines.extend([
        "\n## 3. 작업 지침 및 안전 규격 상세",
        content if content else "*지침 내용 없음*",
        "\n---",
        "*본 문서는 조선 스마트 제조 자율 에이전트 시스템에 의해 자동 생성 및 검증되었습니다.*"
    ])

    full_md = "\n".join(lines)
    with file_path.open("w", encoding="utf-8") as f:
        f.write(full_md)

    return f"문서 저장 완료: {file_path.resolve()}"

@tool
def log_inspection_history(session_id: str, query: str, status: str = "PASS") -> str:
    """공정 판정 및 질의 이력을 감사(Audit) 로그 파일에 기록합니다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_file = OUTPUT_DIR / "audit_history.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] [SESSION: {session_id}] [STATUS: {status}] 질의: {query}\n"
    
    with audit_file.open("a", encoding="utf-8") as f:
        f.write(log_line)
        
    return f"감사 이력 기록 성공 (세션: {session_id}, 상태: {status})"