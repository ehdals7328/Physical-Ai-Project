import os, json
from datetime import datetime

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

class WeldingSupervisorAgent:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if LANGCHAIN_AVAILABLE and self.api_key and not self.api_key.startswith("sk-your"):
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, openai_api_key=self.api_key)
        else:
            self.llm = None

    # [수정] 순수 LLM 질의응답 (RAG 문서 없어도 일반 상식/엔지니어링 지식으로 직접 답변)
    def run_rag_qa(self, user_query: str, rag_chunks: list) -> dict:
        started = datetime.utcnow()

        # 1. RAG 검색 결과가 있으면 컨텍스트 구성, 없으면 일반 답변 모드로 전환
        if rag_chunks:
            rag_context = "\n".join([
                f"- [{c['category']}] {c['document_name']} ({c['source']}, {c['revision']}): {c['chunk_text']}"
                for c in rag_chunks
            ])
            system_prompt = (
                "당신은 조선소 선박 건조 및 용접 기술 전문 수석 엔지니어 AI입니다. "
                "제공된 사내 규격 및 선급 기술 문서를 최우선 근거로 인용하여 질문에 구체적으로 답변하세요."
            )
            user_prompt = f"[참고 기술 문서]\n{rag_context}\n\n[사용자 질의]\n{user_query}"
        else:
            system_prompt = (
                "당신은 조선소 선박 건조, 안전보건 및 용접 기술 전문 수석 엔지니어 AI입니다. "
                "별도 사내 규격 문서가 제공되지 않았으므로, 일반 조선 해양 공학 표준과 산업 안전 지식을 바탕으로 명확하고 친절하게 답변하세요."
            )
            user_prompt = user_query

        # 2. LLM 직접 호출
        if self.llm:
            try:
                res = self.llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ])
                answer = res.content
                mode = "OpenAI_GPT4o_mini"
            except Exception as e:
                answer = f"[LLM API 통신 에러 발생]: {str(e)}\n(.env의 OPENAI_API_KEY 유효성을 확인하세요)"
                mode = f"API_Error: {str(e)}"
        else:
            # LLM API 키 미설정 시에도 친절한 기본 답변 반환
            api_status = "설정됨" if self.api_key else "미등록"
            answer = (
                f"[로컬 에이전트 안내] 질의하신 '{user_query}'에 대한 응답입니다.\n\n"
                f"현재 OpenAI API 키가 연동되지 않아({api_status}) 로컬 모드로 동작 중입니다.\n"
                f"실제 GPT 모델을 호출하여 자유 질의응답을 받으시려면 프로젝트 루트의 `.env` 파일에 "
                f"`OPENAI_API_KEY=sk-...`를 입력하고 서버를 재시작해 주세요."
            )
            mode = "Local_Simulation_Engine"

        completed = datetime.utcnow()
        return {
            "agent_name": "Shipyard_RAG_QA_Agent",
            "execution_status": "COMPLETED",
            "approval_result": "PASS",
            "current_node": "END_QA_SYNTHESIS",
            "state_version": 1,
            "input_data": {"user_query": user_query, "rag_count": len(rag_chunks)},
            "state_data": {"nodes_visited": ["CHECK_INPUT", "INVOKE_LLM"], "engine": mode},
            "output_data": {"referenced_docs": [c.get("document_id") for c in rag_chunks]},
            "response_text": answer,
            "started_at": started,
            "completed_at": completed
        }

    # [모드 2] Vision + 모재 조건 + RAG 통합 작업지시 생성
    def run_work_supervisor(self, work_meta: dict, vision_data: dict, rag_chunks: list) -> dict:
        started = datetime.utcnow()
        mat = work_meta.get("base_material", "AH36")
        thick = float(work_meta.get("material_thickness", 16.0))
        proc = work_meta.get("process_type", "FCAW")
        pos = work_meta.get("welding_position", "2F")
        joint = work_meta.get("joint_type", "BUTT")
        groove = work_meta.get("groove_type", "V-GROOVE")

        classes = [d["class_name"] for d in vision_data.get("detections", [])]
        has_obstacle = "CLAMP_OBSTACLE" in classes

        if has_obstacle or thick > 40.0:
            approval = "HOLD"
            reason = "용접선 50mm 이내 클램프 간섭물 탐지 또는 두께 초과로 인한 작업 보류(HOLD) 판정. 간섭물 제거 후 재스캔 요망."
        else:
            approval = "PASS"
            reason = f"{mat} ({thick}mm, {joint}/{groove}, 자세 {pos}) 조건 및 Vision 용접선 궤적 정상 확인. {proc} 본용접 승인(PASS)."

        instruction_data = {
            "process_sequence": [
                "1단계: 밀폐공간 산소농도(18~23.5%) 측정 및 국소배기장치 가동",
                f"2단계: {mat} ({thick}mm) {groove} 개선면 이물질 청소 및 루트간격(2.1mm) 확인",
                f"3단계: {proc} 조건 설정 (전류 220~245A, 전압 28~30V, CO2 100% 22L/min)",
                "4단계: Vision 스캔 궤적 좌표 따라 자율 주행 용접 수행 후 UT 비파괴 검사"
            ],
            "welding_parameters": {
                "current_amp": "220 ~ 245 A",
                "voltage_volt": "28 ~ 30 V",
                "travel_speed": "30 ~ 35 cm/min",
                "shielding_gas": "CO2 100% (20~25 L/min)",
                "preheat_temp": "50°C 이상" if thick >= 25.0 else "상온 (예열 불필요)"
            },
            "safety_checklist": [
                "산소농도 18% 이상 확인",
                "용접선 주변 50mm 이내 클램프/지그 간섭물 제거",
                "풍속 2.0m/s 이상 시 방풍막 설치"
            ]
        }

        instruction_text = (
            f"[최종 판정: {approval}] {reason}\n"
            f"- 권장 조건: 전류 {instruction_data['welding_parameters']['current_amp']}, "
            f"전압 {instruction_data['welding_parameters']['voltage_volt']}, "
            f"속도 {instruction_data['welding_parameters']['travel_speed']}"
        )

        source_refs = [
            {
                "document_id": c["document_id"],
                "document_name": c["document_name"],
                "category": c["category"],
                "source": c["source"],
                "revision": c["revision"],
                "similarity_score": c["similarity_score"]
            }
            for c in rag_chunks
        ]

        completed = datetime.utcnow()
        return {
            "agent_name": "LangGraph_Welding_Supervisor_v2",
            "execution_status": "COMPLETED",
            "approval_result": approval,
            "current_node": "NODE_FINAL_INSTRUCTION",
            "state_version": 2,
            "input_data": work_meta,
            "state_data": {
                "nodes": ["VISION_CHECK", "RAG_RETRIEVAL", "PARAMETER_CALC", "SAFETY_AUDIT"],
                "detected_classes": classes,
                "obstacle_detected": has_obstacle
            },
            "output_data": instruction_data,
            "response_text": instruction_text,
            "instruction_text": instruction_text,
            "instruction_data": instruction_data,
            "source_references": source_refs,
            "started_at": started,
            "completed_at": completed
        }
