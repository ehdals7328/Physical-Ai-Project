# agent_system/run_test.py
import json
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from main_graph import ship_agent_app


def execute_turn(turn_label: str, query: str, config: dict):
    print(f"\n#################################################################")
    print(f"▶ [{turn_label} 시작] 질문: \"{query}\"")
    print(f"#################################################################")

    # 새 턴 진입 시 이전 산출물 플래그만 리셋 (누적 대화와 파라미터는 config thread_id로 유지)
    turn_input = {
        "user_query": query,
        "messages": [HumanMessage(content=query)],
        "task_type": None,
        "is_param_complete": None,
        "retrieved_docs": [],
        "generated_document": None,
        "final_answer": None,
    }

    step_count = 1
    final_state = {}

    for output in ship_agent_app.stream(
        turn_input,
        config=config,
        recursion_limit=15,
        stream_mode="updates",
        subgraphs=True,
    ):
        if isinstance(output, tuple):
            subgraph_path, node_updates = output
        else:
            subgraph_path, node_updates = (), output

        for node_name, state_update in (node_updates or {}).items():
            if isinstance(state_update, dict):
                final_state.update(state_update)

            prefix = (
                f"[{'/'.join(subgraph_path)} > {node_name}]"
                if subgraph_path
                else f"[{node_name}]"
            )
            print(f"  [{step_count:02d}단계] 👉 {prefix}")

            if node_name == "supervisor":
                print(
                    "    └─ Supervisor LLM 결정 (next):"
                    f" '{state_update.get('next')}'"
                )
            elif node_name == "parameter_checker":
                print(
                    f"    └─ [파라미터 검증 결과]: is_param_complete ="
                    f" {state_update.get('is_param_complete')}"
                )
            elif node_name == "output_validator":
                print("    └─ [양식 검증 통과]")

            step_count += 1

    print(f"\n--- [{turn_label} 결과] ---")
    doc = final_state.get("generated_document")
    if doc:
        print(f"• 태스크 분류: {final_state.get('task_type')}")
        print(
            f"• 표준 문서 전문:\n{json.dumps(doc, indent=2, ensure_ascii=False)}"
        )
    else:
        print(f"• 에이전트 답변 (역질문 안내):")
        print(final_state.get("final_answer"))
    print("-" * 65)


def test_clarification_flow():
    print("=================================================================")
    print("🔄 [Multi-Agent System] 필수 파라미터 누락 검증 & 역질문 멀티턴 테스트")
    print("=================================================================")

    config = {"configurable": {"thread_id": "session_clarification_test"}}

    # 1. 의도적으로 정보를 빼고 질문
    execute_turn(
        "Turn 1 (정보 누락 질문)",
        "울산 야외 독에서 용접 작업지시서 작성해줘",
        config,
    )

    # 2. 빠진 정보를 채워 넣는 답변
    execute_turn(
        "Turn 2 (누락 조건 보충 답변)", "SS400 10T 맞대기로 할 거야", config
    )


if __name__ == "__main__":
    test_clarification_flow()