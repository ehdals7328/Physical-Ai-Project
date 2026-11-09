class VisionAdapter:
    """현재 단계에서는 연결 상태만 확인한다. 실제 YOLO 추론은 후속 통합 범위다."""
    @staticmethod
    def health() -> dict:
        return {"connected": False, "status": "NOT_CONFIGURED", "message": "Vision 모델 연결 예정"}
