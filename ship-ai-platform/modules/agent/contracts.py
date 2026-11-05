"""추후 LangGraph Agent 구현체가 준수할 입력·출력 계약입니다."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class AgentContext:
    """Agent가 작업 판단에 사용할 DB·Vision·RAG·ERP 컨텍스트입니다."""

    work_order: dict[str, Any]
    rag_references: list[dict[str, Any]] = field(default_factory=list)
    vision_result: dict[str, Any] | None = None
    erp_resources: dict[str, Any] = field(default_factory=dict)
    conversation_id: str | None = None


@dataclass(slots=True)
class AgentResult:
    """Agent가 반환할 실행 상태와 작업지시서 데이터를 정의합니다."""

    status: str
    response_text: str
    instruction_data: dict[str, Any] = field(default_factory=dict)
    source_references: list[dict[str, Any]] = field(default_factory=list)
    execution_trace: list[dict[str, Any]] = field(default_factory=list)


class AgentService(Protocol):
    """추후 Agent 실행 서비스의 표준 메서드 시그니처입니다."""

    def run(self, context: AgentContext) -> AgentResult:
        """작업 컨텍스트를 분석하고 결과 및 근거를 반환합니다."""
        ...
