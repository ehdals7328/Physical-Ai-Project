"""추후 YOLO·Segmentation 구현체가 사용할 입력·출력 계약입니다."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class VisionRequest:
    """Vision 분석에 필요한 파일과 작업 식별 정보를 정의합니다."""

    work_id: int
    media_path: str
    media_type: str = "IMAGE"
    camera_id: str | None = None
    frame_sequence: int | None = None


@dataclass(slots=True)
class VisionDetectionResult:
    """검출 객체, 이음새 형상, 모델 실행 정보를 전달합니다."""

    status: str
    model_name: str | None = None
    model_version: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    detections: list[dict[str, Any]] = field(default_factory=list)
    seam_geometry: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


class VisionService(Protocol):
    """추후 Vision 분석 서비스의 표준 메서드 시그니처입니다."""

    def analyze(self, request: VisionRequest) -> VisionDetectionResult:
        """이미지 또는 비디오 프레임을 분석해 검출 결과를 반환합니다."""
        ...
