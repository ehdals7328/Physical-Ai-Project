# agent_system/tools/retriever_tools.py
"""
Retriever Tools Package
Chroma Vector DB 기반 사내 규격(WPS) 및 KOSHA 안전지침 검색 Tool 정의
"""
from typing import List
from langchain_core.tools import BaseTool
from langchain_core.tools import create_retriever_tool
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

# ----------------------------------------------------
# 1. Mock Retriever 정의 (실제 Chroma DB 연동 전까지 호환성 보장)
# ----------------------------------------------------
class MockShipyardRetriever(BaseRetriever):
    """실제 Chroma DB가 없을 때 테스트를 위한 호환용 Retriever"""
    category: str = Field(default="wps")

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        # 모의 사내 WPS 및 KOSHA 지침 지식 베이스
        knowledge_base = {
            "wps": [
                Document(
                    page_content="[사내 WPS-SS400-01] SS400 10T 맞대기 용접 표준: 적정 전류 220A (범위 200~240A), 적정 전압 24V (범위 22~26V), 주행속도 5.0mm/s, 토치 각도 70도, 허용 루트 갭 3.5mm 이하.",
                    metadata={"source": "WPS-SS400-01", "material": "SS400", "thickness": "10T"}
                ),
                Document(
                    page_content="[사내 WPS-SS400-02] SS400 20T 맞대기 다층 용접 표준: 적정 전류 260A, 적정 전압 28V, 주행속도 4.5mm/s, 토치 각도 70도, 루트 갭 4.0mm 이하.",
                    metadata={"source": "WPS-SS400-02", "material": "SS400", "thickness": "20T"}
                ),
                Document(
                    page_content="[품질지침-W-03] 선박 후판 용접 시 언더컷 결함 방지: 과도한 전류 사용 금지(상한선 240A 준수), 이송속도 감속(4~5mm/s), 토치 각도 70도 유지 및 개선 홈 끝단 일시 정지(Pause) 필수.",
                    metadata={"source": "WPS-DEFECT-UNDERCUT", "topic": "undercut"}
                )
            ],
            "safety": [
                Document(
                    page_content="[KOSHA 지침 M-01] 저온 용접 관리 규정: 외기 온도가 5℃ 이하인 환경에서는 모재 수소 유입에 따른 저온 균열 방지를 위해 작업 전 100℃ 이상 예열을 반드시 실시해야 함.",
                    metadata={"source": "KOSHA-M-01", "type": "temperature"}
                ),
                Document(
                    page_content="[KOSHA 지침 C-03] 중량물 인양 안전 기준: 순간풍속 10m/s 초과 시 타워크레인 및 하이드로 크레인 인양 작업을 전면 중지한다. 슬링벨트 체결 각도는 60도 이내 유지, 안전율 5 이상 확보 필수.",
                    metadata={"source": "KOSHA-C-03", "type": "wind"}
                )
            ]
        }
        
        target_docs = knowledge_base.get(self.category, [])
        # 간단한 키워드 필터링 (일치하는 문서 우선 반환)
        matched = [doc for doc in target_docs if any(term in doc.page_content for term in query.split())]
        return matched if matched else target_docs[:2]


# ----------------------------------------------------
# 2. BaseTool 객체 생성 팩토리 함수
# ----------------------------------------------------
def build_wps_retriever_tool(retriever: BaseRetriever = None) -> BaseTool:
    """사내 용접 시방서(WPS) 전용 검색 도구 생성"""
    if retriever is None:
        retriever = MockShipyardRetriever(category="wps")
        
    return create_retriever_tool(
        retriever=retriever,
        name="search_wps_standard",
        description=(
            "사내 표준 용접 시방서(WPS) 규정을 검색합니다. "
            "모재 강종(SS400, AH36 등), 두께(10T, 20T 등), 이음 형상, 결함 방지 기준에 따른 "
            "표준 전류, 전압, 주행 속도, 토치 각도 등의 엔지니어링 파라미터를 조회할 때 사용합니다."
        )
    )

def build_safety_retriever_tool(retriever: BaseRetriever = None) -> BaseTool:
    """KOSHA 및 사내 안전 수칙 전용 검색 도구 생성"""
    if retriever is None:
        retriever = MockShipyardRetriever(category="safety")
        
    return create_retriever_tool(
        retriever=retriever,
        name="search_safety_rules",
        description=(
            "KOSHA 산업안전보건지침 및 중량물 취급 안전 기준을 검색합니다. "
            "크레인 인양 제한 풍속, 슬링벨트/샤클 체결 각도, 저온 예열 규정, "
            "작업 중지 기준을 조회할 때 사용합니다."
        )
    )

# 런타임에서 호출할 도구 싱글톤 인스턴스
wps_retriever_tool = build_wps_retriever_tool()
safety_retriever_tool = build_safety_retriever_tool()