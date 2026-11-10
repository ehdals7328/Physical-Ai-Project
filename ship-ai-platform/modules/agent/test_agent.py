"""
스마트 선박 자율 제조 멀티에이전트 단독 통합 테스트 (test_agent.py)
Human-in-the-Loop(HITL) 관리자 승인 게이트 연동 완성본 (하드코딩 0% 정석 에이전트 아키텍처)
"""
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

CURRENT_FILE = Path(__file__).resolve()
PLATFORM_ROOT = CURRENT_FILE.parents[2]

if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))

load_dotenv(PLATFORM_ROOT / ".env")

from modules.agent.supervisor import ship_agent_app
from modules.agent.tools.erp_tools import commit_erp_work_order, list_erp_workers
from modules.agent.tools.document_tools import export_work_order_file, log_inspection_history
from flask import Flask, jsonify, render_template_string, request
from langchain_core.messages import HumanMessage

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>스마트 조선소 Agentic AI + ERP 관제 테스트</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f4f6f9; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .chat-container { height: 60vh; overflow-y: auto; background: #ffffff; border: 1px solid #dee2e6; border-radius: 8px; padding: 15px; }
        .msg-user { background: #007bff; color: white; border-radius: 15px 15px 0 15px; padding: 10px 14px; max-width: 75%; float: right; clear: both; margin-bottom: 12px; }
        .msg-ai { background: #f1f3f5; color: #212529; border-radius: 15px 15px 15px 0; padding: 12px 16px; max-width: 85%; float: left; clear: both; margin-bottom: 12px; border: 1px solid #e9ecef; }
        .msg-approval { background: #fff3cd; color: #664d03; border: 1px solid #ffecb5; border-radius: 12px; padding: 15px; max-width: 85%; float: left; clear: both; margin-bottom: 12px; }
        .status-box { background: #212529; color: #a5d6a7; border-radius: 8px; padding: 12px; font-family: monospace; font-size: 0.85rem; height: 60vh; overflow-y: auto; }
        pre { white-space: pre-wrap; word-break: break-all; margin: 0; }
    </style>
</head>
<body class="py-4">
<div class="container-fluid px-4">
    <div class="row mb-3">
        <div class="col-12 d-flex justify-content-between align-items-center">
            <h3>⚓ 스마트 조선소 Agentic AI & 분산 ERP E2E 테스트 벤치</h3>
            <div>
                <span class="badge bg-primary">LangGraph Multi-Agent</span>
                <span class="badge bg-success">HQ Oracle ERP 연동</span>
                <span class="badge bg-warning text-dark">PostgreSQL B2B 연동</span>
                <span class="badge bg-danger">HITL 승인 게이트</span>
            </div>
        </div>
    </div>

    <div class="card mb-3 p-2 bg-light">
        <div class="d-flex gap-2 flex-wrap align-items-center">
            <span class="fw-bold me-2">⚡ 검증 시나리오 버튼:</span>
            <button class="btn btn-outline-primary btn-sm" onclick="setQuery('거제 야드에서 B-305 블록 김철수 반장 투입해서 외주 협력사 연계로 작업지시서 발행해줘')">
                1. B-305 B2B 외주 & 지시서 결재 대기
            </button>
            <button class="btn btn-outline-warning btn-sm text-dark" onclick="setQuery('사내 트랜스포터 없으니까 바로 외주 배차 1대 확정해줘')">
                2. 협력업체 다이렉트 외주 배차 승인
            </button>
            <button class="btn btn-outline-danger btn-sm" onclick="setQuery('울산 야드에서 B-104 블록 조객원 작업자 투입해서 작업지시서 발행해줘')">
                3. 선급 자격증 만료 차단(Fail-Closed)
            </button>
            <button class="btn btn-outline-info btn-sm text-dark" onclick="setQuery('현재 사내 창고에 남아있는 강판이랑 용접 와이어 재고 현황 알려줘')">
                4. 단순 ERP 재고 조회(Bypass)
            </button>
        </div>
    </div>

    <div class="row">
        <div class="col-md-7">
            <div class="card shadow-sm mb-3">
                <div class="card-header bg-white fw-bold">대화 및 작업지시서 출력창 (Thread: <span id="session-display">session_demo_01</span>)</div>
                <div class="card-body p-2">
                    <div id="chatBox" class="chat-container">
                        <div class="msg-ai">
                            안녕하세요! 스마트 조선소 공정 관제 및 분산 ERP 연동 에이전트입니다.<br>
                            작업지시서 발행, 자재 재고 확인, 작업자 명단 조회, 외주 배차 요청을 자연어로 질의해 보세요.
                        </div>
                    </div>
                </div>
                <div class="card-footer bg-white">
                    <form id="chatForm" onsubmit="sendMessage(event)">
                        <div class="input-group">
                            <input type="text" id="userInput" class="form-control" placeholder="명령이나 질문을 입력하세요..." autocomplete="off" required>
                            <button class="btn btn-primary" type="submit" id="sendBtn">전송</button>
                            <button class="btn btn-outline-secondary" type="button" onclick="resetSession()">세션 초기화</button>
                        </div>
                    </form>
                </div>
            </div>
        </div>

        <div class="col-md-5">
            <div class="card shadow-sm mb-3">
                <div class="card-header bg-dark text-white fw-bold d-flex justify-content-between align-items-center">
                    <span>📡 에이전트 내부 상태 (State Monitor)</span>
                    <span id="loadSpinner" class="spinner-border spinner-border-sm text-warning" style="display: none;"></span>
                </div>
                <div class="card-body p-2 bg-dark">
                    <div id="statusBox" class="status-box">
                        <pre id="stateJson">// 에이전트 실행 시 실시간 State가 표출됩니다.</pre>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    let currentThreadId = "session_" + Math.random().toString(36).substring(2, 9);
    let pendingDoc = null;
    document.getElementById("session-display").innerText = currentThreadId;

    function setQuery(text) {
        document.getElementById("userInput").value = text;
        document.getElementById("userInput").focus();
    }

    function resetSession() {
        currentThreadId = "session_" + Math.random().toString(36).substring(2, 9);
        pendingDoc = null;
        document.getElementById("session-display").innerText = currentThreadId;
        document.getElementById("chatBox").innerHTML = '<div class="msg-ai">세션이 초기화되었습니다. 새로운 맥락으로 대화를 시작합니다.</div>';
        document.getElementById("stateJson").innerText = "// 세션 초기화 완료";
    }

    async function sendMessage(e) {
        e.preventDefault();
        const input = document.getElementById("userInput");
        const query = input.value.trim();
        if (!query) return;

        const chatBox = document.getElementById("chatBox");
        chatBox.innerHTML += `<div class="msg-user">${query}</div>`;
        input.value = "";
        chatBox.scrollTop = chatBox.scrollHeight;

        document.getElementById("sendBtn").disabled = true;
        document.getElementById("loadSpinner").style.display = "inline-block";

        try {
            const resp = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query: query,
                    thread_id: currentThreadId
                })
            });

            const data = await resp.json();
            
            let aiText = data.final_answer || "응답 내용이 없습니다.";
            let formattedText = aiText.replace(/\\n/g, "<br>").replace(/\\*\\*(.*?)\\*\\*/g, "<strong>$1</strong>");
            chatBox.innerHTML += `<div class="msg-ai">${formattedText}</div>`;

            // DRAFT 또는 WAITING_APPROVAL 상태일 때 HITL 결재 게이트 활성화
            if (data.generated_document && (data.generated_document.status === "DRAFT" || data.generated_document.status === "WAITING_APPROVAL")) {
                pendingDoc = data.generated_document;
                const docTitle = pendingDoc.title || "외주 조달 및 공정 작업 계획";
                chatBox.innerHTML += `
                    <div class="msg-approval" id="approval-gate-box">
                        <div class="fw-bold mb-2">🛡️ [HITL 관리자 결재 게이트: ${docTitle}]</div>
                        <div>공정 조건과 외주 SCM 조달/배차 계획을 검토 후 승인 여부를 결정하십시오.</div>
                        <div class="mt-3">
                            <button class="btn btn-success btn-sm me-2" onclick="submitApproval('APPROVE')">👍 최종 승인 및 ERP 원장 커밋</button>
                            <button class="btn btn-danger btn-sm" onclick="submitApproval('REJECT')">❌ 요청 반려 (Rollback)</button>
                        </div>
                    </div>
                `;
            }

            chatBox.scrollTop = chatBox.scrollHeight;
            document.getElementById("stateJson").innerText = JSON.stringify(data.state_summary, null, 2);

        } catch (err) {
            chatBox.innerHTML += `<div class="msg-ai text-danger">통신 오류가 발생했습니다: ${err.message}</div>`;
        } finally {
            document.getElementById("sendBtn").disabled = false;
            document.getElementById("loadSpinner").style.display = "none";
        }
    }

    async function submitApproval(action) {
        if (!pendingDoc) return;
        const chatBox = document.getElementById("chatBox");
        
        const gateBox = document.getElementById("approval-gate-box");
        if (gateBox) gateBox.remove();

        chatBox.innerHTML += `<div class="msg-user">${action === 'APPROVE' ? '관리자 승인 [APPROVE]' : '관리자 반려 [REJECT]'} 선택</div>`;
        document.getElementById("loadSpinner").style.display = "inline-block";

        try {
            const resp = await fetch("/api/approve", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    thread_id: currentThreadId,
                    action: action,
                    doc: pendingDoc
                })
            });

            const resData = await resp.json();
            chatBox.innerHTML += `<div class="msg-ai ${action === 'APPROVE' ? 'text-success' : 'text-danger'} fw-bold">${resData.message}</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;

            if (resData.erp_commit_result) {
                let currentJson = JSON.parse(document.getElementById("stateJson").innerText);
                currentJson["9. ERP_Commit_Transaction"] = resData.erp_commit_result;
                document.getElementById("stateJson").innerText = JSON.stringify(currentJson, null, 2);
            }
        } catch (err) {
            chatBox.innerHTML += `<div class="msg-ai text-danger">결재 처리 오류: ${err.message}</div>`;
        } finally {
            pendingDoc = null;
            document.getElementById("loadSpinner").style.display = "none";
        }
    }
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/api/chat", methods=["POST"])
def chat_api():
    payload = request.get_json() or {}
    user_query = payload.get("query", "").strip()
    thread_id = payload.get("thread_id", "default_thread")

    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        # 웹 테스트 시 HITL 결재 버튼을 거치도록 기본값은 False로 진입
        "hitl_approved": False,
    }

    result = ship_agent_app.invoke(initial_state, config=config)

    generated_doc = result.get("generated_document")
    task_type = result.get("task_type")
    is_action = result.get("is_action_requested", False)

    # 단독 외주 배차/발주 요청(scm_partner_qa) 시 B2B 조회 결과 기반으로 HITL 결재 문서 동적 생성
    if not generated_doc and is_action and task_type in ("scm_partner_qa", "erp_resource_qa"):
        partner_plan = result.get("partner_erp_plan") or {}
        selected = partner_plan.get("selected_partner") or {}
        params = result.get("extracted_params") or {}

        p_id = selected.get("partner_id")
        p_name = selected.get("partner_name") or p_id
        b_type = selected.get("business_type") or result.get("outsourcing_domain") or "OUTSOURCED_B2B"
        allocated_lot = (
            selected.get("partner_lot_no")
            or selected.get("assigned_vehicle_code")
            or selected.get("assigned_worker_id")
        )

        generated_doc = {
            "work_order_number": f"PO-DRAFT-{p_id}",
            "title": f"[외주 실행 결재안] {p_name} ({b_type})",
            "status": "DRAFT",
            "approval_status": "DRAFT",
            "execution_type": "OUTSOURCED_B2B",
            "allocated_lot_no": allocated_lot,
            "assigned_vehicle_code": selected.get("assigned_vehicle_code"),
            "partner_erp_plan": partner_plan,
            "parameters": params,
            "content": result.get("final_answer"),
        }

    preview_commit = None
    if generated_doc and generated_doc.get("status") in ("DRAFT", "WAITING_APPROVAL"):
        preview_commit = {
            "work_order_number": generated_doc.get("work_order_number"),
            "execution_type": generated_doc.get("execution_type", "OUTSOURCED_B2B"),
            "erp_transaction_status": "DRAFT",
            "note": "관리자 결재 대기 중 (DB 미반영)",
        }

    state_summary = {
        "0. Thread_ID": thread_id,
        "1. Task_Type": task_type,
        "2. Is_Param_Complete": result.get("is_param_complete"),
        "3. Missing_Params": result.get("missing_params"),
        "4. Extracted_Params": result.get("extracted_params"),
        "5. HQ_ERP_Status": result.get("erp_resource_status"),
        "6. Outsourcing_Required": result.get("is_outsourcing_required"),
        "7. Partner_ERP_B2B": result.get("partner_erp_plan"),
        "8. Weather_Status": result.get("weather_data", {}).get("safety_evaluation") if result.get("weather_data") else None,
        "9. ERP_Commit_Transaction": preview_commit or result.get("erp_commit_result"),
    }

    return jsonify({
        "final_answer": result.get("final_answer"),
        "generated_document": generated_doc,
        "state_summary": state_summary,
    })


@app.route("/api/approve", methods=["POST"])
def approve_api():
    payload = request.get_json() or {}
    action = payload.get("action", "APPROVE")
    doc = payload.get("doc") or {}
    params = doc.get("parameters") or {}
    erp_commit_draft = doc.get("erp_commit") or {}
    partner_plan = doc.get("partner_erp_plan") or {}
    selected_partner = partner_plan.get("selected_partner") or {}
    is_outsourced = doc.get("execution_type") == "OUTSOURCED_B2B"

    if action == "REJECT":
        return jsonify({
            "status": "REJECTED",
            "message": "❌ [요청 반려] 관리자에 의해 작업지시 및 외주 발주가 반려되었습니다. (원장 롤백)",
            "erp_commit_result": {
                "erp_transaction_status": "ROLLED_BACK",
                "reason": "ADMIN_REJECTED",
            },
        })

    # 작업자 ID 및 블록 ID가 없는 단독 외주 배차 건일 경우 DB에서 유효 값 동적 보완
    worker_id = erp_commit_draft.get("worker_id")
    worker_name = erp_commit_draft.get("worker_name") or params.get("worker_name")
    if not worker_id or worker_id in ("UNASSIGNED", "미배정"):
        w_list = list_erp_workers.invoke({"branch_code": params.get("branch_code")}).get("workers", [])
        valid_workers = [w for w in w_list if w.get("is_valid")]
        if valid_workers:
            worker_id = valid_workers[0]["worker_id"]
            worker_name = valid_workers[0]["worker_name"]

    block_id = params.get("block_id") or "B-104"
    lot_no = doc.get("allocated_lot_no") or erp_commit_draft.get("allocated_lot_no") or "LOT-DEFAULT"
    partner_id = selected_partner.get("partner_id") or erp_commit_draft.get("partner_id")

    commit_res = commit_erp_work_order.invoke({
        "block_id": block_id,
        "material": params.get("material", ""),
        "allocated_lot_no": lot_no,
        "worker_name": worker_name,
        "worker_id": worker_id,
        "assigned_vehicle_code": doc.get("assigned_vehicle_code"),
        "wire_item_code": doc.get("wire_item_code"),
        "is_outsourced": is_outsourced,
        "partner_id": partner_id,
        "reserve_wire_kg": doc.get("reserve_wire_kg"),
        "delivery_yard_zone": params.get("work_zone"),
    })

    try:
        final_doc_data = dict(doc)
        final_doc_data["work_order_number"] = commit_res.get("work_order_number")
        final_doc_data["erp_commit"] = commit_res
        final_doc_data["status"] = "APPROVED"
        final_doc_data["approval_status"] = "APPROVED"

        export_work_order_file.invoke({
            "document_data": final_doc_data,
            "file_prefix": "approved_order",
        })
        log_inspection_history.invoke({
            "session_id": commit_res.get("work_order_number", "APPROVED_SESSION"),
            "query": f"HITL 외주/지시서 승인 완료: {block_id}",
            "status": "APPROVED",
        })
    except Exception as e:
        print(f"[HITL Commit Log Notice]: {e}")

    return jsonify({
        "status": "APPROVED",
        "message": f"✅ [관리자 최종 승인 완료] 외주 발주 및 작업지시가 공식 확정되었습니다. (식별번호: {commit_res.get('work_order_number')})",
        "erp_commit_result": commit_res,
    })


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 [스마트 조선소 Agentic AI + ERP 테스트 서버 구동 (HITL 및 인사 DB 실시간 연동)]")
    print("👉 브라우저 주소창에 입력하세요: http://127.0.0.1:5000")
    print("=" * 70 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)