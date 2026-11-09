class AgentAdapter:
    """현재 단계에서는 연결 상태만 확인한다. 실제 LangGraph 실행은 후속 통합 범위다."""
    @staticmethod
    def health() -> dict:
        return {"connected": False, "status": "NOT_CONFIGURED", "message": "LangGraph Agent 연결 예정"}
