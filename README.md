# Physical-AI-Project

**자동화 용접 로봇을 위한 Physical AI 서비스 개발**
> "보고(Vision), 생각하고(LLM), 행동하는 로봇(ROS2, VLA)"

---

##  1차 개발 목표 (11.2 ~ 12.4)
현재 페이즈에서는 로봇을 제어하기 위한 **소프트웨어 완결형 서비스 플랫폼** 구축에 집중합니다.

- **👀 Vision (객체 인식)**
  - 선박 제조 환경 객체 탐지 및 용접 목표 위치 식별
  - Tech: `YOLOv11`, `Object Detection`
- **🧠 Agentic AI (자율 업무)**
  - Agent 협업 기반 전문 지식 제공 및 문서 작성 자동화
  - Tech: `LangGraph`, `MCP`
- **💾 Database & Backend (지식 기반 인프라)**
  - 선박제조 기술 및 회사 내부문서 데이터베이스 구축
  - Tech: `VectorDB`, `Oracle RDB`, `Flask`

---

## 📐 Architecture Diagram
(작성예정)