# agent_system/tools/document_tools.py
"""
Document Tools Package
생성된 지시서/계획서 JSON을 공식 서식 Markdown 파일로 렌더링하고 저장하는 로컬 @tool
"""
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from langchain_core.tools import tool

# 작업 파일 저장 기본 디렉토리
OUTPUT_DIR = Path("./generated_docs")

@tool
def export_work_order_file(document_data: Dict[str, Any], file_prefix: str = "work_order") -> str:
    """
    구조화된 작업지시서 또는 안전계획서 데이터를 표준 Markdown 문서 파일로 렌더링하고 저장합니다.
    
    Args:
        document_data: 에이전트가 생성한 지시서/계획서 JSON 딕셔너리
        file_prefix: 저장할 파일 접두사 ('work_order', 'heavy_plan' 등)
        
    Returns:
        저장 완료 메시지 및 파일 저장 경로
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = OUTPUT_DIR / f"{file_prefix}_{timestamp}.md"

    title = document_data.get("document_title", "표준 작업 문서")
    material = document_data.get("material", "-")
    thickness = document_data.get("thickness", "-")
    current = document_data.get("target_current", "-")
    voltage = document_data.get("target_voltage", "-")
    speed = document_data.get("travel_speed", "-")
    torch_angle = document_data.get("torch_angle", "-")
    weather = document_data.get("weather_condition", {})
    precautions = document_data.get("safety_precautions", [])

    # 표준 공정 양식 렌더링
    md_content = f"""# {title}

**문서 발행 일시:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**관리 번호:** WO-{timestamp}

---

## 1. 공정 및 모재 기본 사양
- **모재 재질 (Material):** {material}
- **모재 두께 (Thickness):** {thickness}
- **권장 토치 각도:** {torch_angle}

## 2. 표준 용접 파라미터 (WPS 기준)
| 구분 | 표준 설정치 | 단위 |
| :--- | :--- | :--- |
| 용접 전류 (Current) | {current} | A |
| 용접 전압 (Voltage) | {voltage} | V |
| 주행 속도 (Travel Speed) | {speed} | mm/s |

## 3. 현장 환경 기상 조건
- **작업 위치:** {weather.get('location', '-')}
- **외기 온도:** {weather.get('temperature', '-')} ℃
- **실시간 풍속:** {weather.get('wind_speed', '-')} m/s
- **습도:** {weather.get('humidity', '-')} %
- **기상 상태:** {weather.get('condition', '-')}

## 4. 필수 안전 및 공정 수칙
"""
    for i, item in enumerate(precautions, 1):
        md_content += f"{i}. {item}\n"

    md_content += "\n---\n*본 문서는 조선 스마트 제조 멀티에이전트 시스템에 의해 자동 생성 및 검증되었습니다.*"

    # 파일 디스크 저장
    with file_path.open("w", encoding="utf-8") as f:
        f.write(md_content)

    return f"문서 파일 생성 완료: {file_path.resolve()}"