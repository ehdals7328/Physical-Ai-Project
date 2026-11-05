"""
modules/agent/supervisor.py
LangGraph 기반 AI-Agent 통합 제어 모듈 (팀원 연계용 표준 인터페이스 스텁).
"""
from typing import Dict, Any

class MockAgentApp:
    """에이전트 개발 팀원의 LangGraph ship_agent_app과 100% 호환되는 인터페이스 스텁."""
    def invoke(self, state: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        user_query = state.get("user_query", "")
        params = state.get("extracted_params", {}) or {}
        vision = state.get("vision_data") or {}

        has_obstacle = False
        if vision and "detections" in vision:
            has_obstacle = any("OBSTACLE" in d.get("class_name", "") for d in vision["detections"])

        mat = params.get("material", "AH36")
        thick = params.get("thickness", "16T")
        proc = params.get("process", "FCAW")
        robot = params.get("assigned_robot", "ROBOT-001")
        worker = params.get("assigned_worker", "전문용접사")

        if has_obstacle:
            final_text = (
                f"⚠️ [주의: 클램프 간섭물 감지 - HOLD 조치]\n"
                f"- 대상: {mat} ({thick}) {proc}\n"
                f"- 감지된 위험: 용접 궤적 부근 고정용 클램프 간섭 발생\n"
                f"- 조치: 관리자(ADMIN) 현장 확인 및 지그 재조정 필요"
            )
        else:
            final_text = (
                f"📋 [표준 선박 용접 작업지시서 발행 완료]\n"
                f"1. 대상 모재: {mat} ({thick}) 맞대기(BUTT) 이음\n"
                f"2. 권장 조건: {proc} 공정, 전류 260~290A, 전압 28~31V, 속도 28~32 cpm\n"
                f"3. 배정 설비: {robot} (6축 협동로봇)\n"
                f"4. 감리 기술자: {worker}\n"
                f"5. 준수 규격: KOSHA 및 선급(KR) 3편 12-20 규정 준용"
            )

        return {
            "final_answer": final_text,
            "generated_document": {
                "title": f"{mat} {thick} 표준 작업지시서",
                "parameters": params,
                "status": "HOLD" if has_obstacle else "PASS"
            },
            "extracted_params": params,
            "task_type": "doc_generation"
        }

ship_agent_app = MockAgentApp()
