/* Shipyard Physical AI Platform - 화면 이벤트와 API 통신 */
(() => {
    "use strict";

    const byId = (id) => document.getElementById(id);

    /** API 응답을 JSON으로 읽고 오류 상태를 공통 방식으로 처리합니다. */
    async function apiRequest(url, options = {}) {
        const response = await fetch(url, {
            credentials: "same-origin",
            ...options,
            headers: { Accept: "application/json", ...(options.headers || {}) },
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
            if (response.status === 401) {
                showToast("로그인이 필요합니다. 계정으로 로그인한 뒤 다시 시도하세요.", "인증 필요");
            }
            throw new Error(payload.message || `요청에 실패했습니다. (HTTP ${response.status})`);
        }
        return payload;
    }

    /** 공통 알림 토스트에 제목과 메시지를 표시합니다. */
    function showToast(message, title = "알림") {
        const toastElement = byId("appToast");
        if (!toastElement || !window.bootstrap) {
            window.alert(message);
            return;
        }
        byId("toastTitle").textContent = title;
        byId("toastMessage").textContent = message;
        window.bootstrap.Toast.getOrCreateInstance(toastElement, { delay: 4200 }).show();
    }

    /** 상태 점과 상태 텍스트를 연결 결과에 맞게 갱신합니다. */
    function setConnectionState(dotId, stateId, connected, onlineText = "정상 연결") {
        const dot = byId(dotId);
        const state = byId(stateId);
        if (dot) dot.classList.toggle("is-online", Boolean(connected));
        if (dot) dot.classList.toggle("is-offline", !connected);
        if (state) state.textContent = connected ? onlineText : "연결 확인 필요";
    }

    /** 연결 현황 API를 호출해 DB와 RAG의 상태 및 건수를 표시합니다. */
    async function refreshStatus() {
        try {
            const data = await apiRequest("/api/system/status");
            const userDb = data.databases?.ship_user;
            const erpDb = data.databases?.ship_erp;
            const rag = data.rag;
            setConnectionState("shipUserDot", "shipUserState", userDb?.connected);
            setConnectionState("shipErpDot", "shipErpState", erpDb?.connected);
            setConnectionState("ragDot", "ragState", rag?.connected);

            byId("shipUserDetail").textContent = userDb?.connected ? `${userDb.table_count ?? 9}개 테이블 연결 확인` : "DB 1 연결 또는 테이블 상태 확인 필요";
            byId("shipErpDetail").textContent = erpDb?.connected ? `${erpDb.table_count ?? 7}개 테이블 연결 확인` : "DB 2 연결 또는 테이블 상태 확인 필요";
            byId("ragDetail").textContent = rag?.connected ? `공통 ${rag.counts.public ?? 0}청크 · 사내 ${rag.counts.internal ?? 0}청크` : "ChromaDB 경로 및 컬렉션 상태 확인 필요";
            if (byId("publicChunkCount")) byId("publicChunkCount").textContent = rag?.counts?.public ?? "-";
            if (byId("internalChunkCount")) byId("internalChunkCount").textContent = rag?.counts?.internal ?? "-";
        } catch (error) {
            setConnectionState("shipUserDot", "shipUserState", false);
            setConnectionState("shipErpDot", "shipErpState", false);
            setConnectionState("ragDot", "ragState", false);
            console.error("시스템 상태 조회 실패:", error);
        }
    }

    /** 로그인 정보를 확인하고 관리자 기능의 노출 여부를 제어합니다. */
    async function checkSession() {
        try {
            const data = await apiRequest("/api/auth/session-check");
            const badge = byId("userBadge");
            const loginLink = byId("loginLink");
            const logoutButton = byId("logoutButton");
            if (data.logged_in) {
                badge.textContent = `${data.display_name} · ${data.role}`;
                loginLink.classList.add("d-none");
                logoutButton.classList.remove("d-none");
            } else {
                badge.textContent = "비로그인";
                loginLink.classList.remove("d-none");
                logoutButton.classList.add("d-none");
            }
            const isAdmin = data.logged_in && data.role === "ADMIN";
            document.querySelectorAll("[data-requires-admin]").forEach((element) => {
                element.classList.toggle("d-none", !isAdmin);
            });
            byId("userDocPermissionNote")?.classList.toggle("d-none", !data.logged_in || isAdmin);
            if (isAdmin && byId("internalDocList")) await loadInternalDocs();
            if (data.logged_in && byId("branchSelect")) await loadBranches();
            if (data.logged_in && byId("workOrderList")) await loadWorkOrders();
        } catch (error) {
            console.error("세션 확인 실패:", error);
        }
    }

    /** 관리자 문서 목록을 가져와 안전한 DOM 노드로 렌더링합니다. */
    async function loadInternalDocs() {
        const list = byId("internalDocList");
        if (!list) return;
        list.replaceChildren(makeMessage("문서 목록을 불러오는 중입니다.", "inline-loading"));
        try {
            const data = await apiRequest("/api/admin/internal-docs");
            list.replaceChildren();
            if (!data.documents?.length) {
                list.append(makeMessage("등록된 사내 PDF가 없습니다.", "inline-empty"));
                return;
            }
            data.documents.forEach((doc) => {
                const row = document.createElement("div");
                row.className = "document-row";
                const details = document.createElement("div");
                details.className = "flex-grow-1";
                const name = document.createElement("div");
                name.className = "document-name";
                name.textContent = `#${doc.document_id} · ${doc.document_name}`;
                const meta = document.createElement("div");
                meta.className = "document-meta";
                meta.textContent = `${doc.category || "미분류"} · ${doc.source || "출처 미기재"} · ${doc.created_at || ""}`;
                const state = document.createElement("div");
                state.className = "document-status";
                state.textContent = `색인 상태: ${doc.ingest_status || "UNKNOWN"}`;
                details.append(name, meta, state);
                const remove = document.createElement("button");
                remove.className = "btn btn-sm btn-outline-danger";
                remove.type = "button";
                remove.textContent = "삭제";
                remove.dataset.documentId = String(doc.document_id);
                remove.addEventListener("click", () => deleteInternalDoc(doc.document_id));
                row.append(details, remove);
                list.append(row);
            });
        } catch (error) {
            list.replaceChildren(makeMessage(error.message, "inline-empty"));
        }
    }

    /** 관리자 요청으로 사내 문서와 검색 인덱스를 삭제합니다. */
    async function deleteInternalDoc(documentId) {
        if (!window.confirm(`문서 #${documentId}와 연결된 검색 청크를 삭제할까요?`)) return;
        try {
            const result = await apiRequest(`/api/admin/internal-docs/${documentId}`, { method: "DELETE" });
            showToast(result.message, "문서 삭제 완료");
            await Promise.all([loadInternalDocs(), refreshStatus()]);
        } catch (error) {
            showToast(error.message, "문서 삭제 실패");
        }
    }

    /** 텍스트만 표시하는 DOM 요소를 생성해 API 데이터의 HTML 삽입을 방지합니다. */
    function makeMessage(message, className = "inline-empty") {
        const element = document.createElement("div");
        element.className = className;
        element.textContent = message;
        return element;
    }

    /** RAG 검색 결과 한 건을 원문·출처·점수와 함께 화면에 추가합니다. */
    function createResultCard(item, index) {
        const card = document.createElement("article");
        card.className = "result-item";
        const header = document.createElement("div");
        header.className = "result-item-head";
        const left = document.createElement("div");
        left.className = "flex-grow-1";
        const chip = document.createElement("span");
        chip.className = `result-chip${item.access_level === "INTERNAL" ? " internal" : ""}`;
        chip.textContent = item.rag_type || item.access_level || "RAG";
        const name = document.createElement("div");
        name.className = "result-document";
        name.textContent = `${index + 1}. ${item.document_name || "이름 없는 문서"}`;
        const meta = document.createElement("div");
        meta.className = "result-metadata";
        meta.textContent = `${item.category || "미분류"} · ${item.source || "출처 미기재"} · p.${item.page_start || item.page || "-"}`;
        left.append(chip, name, meta);
        const score = document.createElement("div");
        score.className = "result-score";
        score.textContent = item.rerank_score !== undefined
            ? `재정렬 점수 ${item.rerank_score}`
            : `벡터 거리 ${item.vector_distance ?? "-"}`;
        header.append(left, score);
        const excerpt = document.createElement("p");
        excerpt.className = "result-excerpt";
        excerpt.textContent = item.chunk_text || "원문 청크가 비어 있습니다.";
        card.append(header, excerpt);
        return card;
    }

    /** RAG 검색 API를 호출하고 원문 근거를 결과 영역에 표시합니다. */
    async function submitRagSearch(event) {
        event.preventDefault();
        const button = byId("ragSubmitButton");
        const resultList = byId("ragResults");
        const emptyState = byId("ragEmptyState");
        const query = byId("ragQuery").value.trim();
        if (!query) return;
        button.disabled = true;
        button.textContent = "검색 중...";
        resultList.replaceChildren();
        emptyState.classList.remove("d-none");
        emptyState.querySelector("strong").textContent = "문서를 검색하고 있습니다.";
        emptyState.querySelector("span").textContent = "질의 임베딩과 벡터 검색을 수행하고 있습니다.";
        byId("ragResultMeta").textContent = "검색 실행 중";
        try {
            const data = await apiRequest("/api/rag/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query,
                    rag_mode: byId("ragMode").value,
                    top_k: Number(byId("ragTopK").value),
                }),
            });
            byId("ragResultMeta").textContent = `${data.result_count}건 검색`;
            if (data.references?.length) {
                emptyState.classList.add("d-none");
                data.references.forEach((item, index) => resultList.append(createResultCard(item, index)));
            } else {
                emptyState.classList.remove("d-none");
                emptyState.querySelector("strong").textContent = "검색된 문서 청크가 없습니다.";
                emptyState.querySelector("span").textContent = "선택한 컬렉션에 색인된 문서가 있는지 확인하거나 검색어를 구체화하세요.";
            }
            const note = document.createElement("div");
            note.className = "generation-note";
            note.textContent = data.generation?.message || "문서 검색이 완료되었습니다.";
            resultList.append(note);
        } catch (error) {
            emptyState.classList.remove("d-none");
            emptyState.querySelector("strong").textContent = "검색 요청을 처리하지 못했습니다.";
            emptyState.querySelector("span").textContent = error.message;
            byId("ragResultMeta").textContent = "오류";
        } finally {
            button.disabled = false;
            button.textContent = "⌕ 문서 검색 실행";
        }
    }

    /** 선택한 PDF를 서버에 전송하고 색인 완료 결과를 표시합니다. */
    async function uploadInternalDoc(event) {
        event.preventDefault();
        const file = byId("pdfFileInput").files[0];
        if (!file) return;
        const button = byId("uploadDocButton");
        const body = new FormData();
        body.append("pdf_file", file);
        button.disabled = true;
        button.textContent = "PDF 처리 및 색인 중...";
        try {
            const data = await apiRequest("/api/admin/internal-docs/upload", { method: "POST", body });
            showToast(data.message, "업로드 완료");
            byId("uploadDocForm").reset();
            await Promise.all([loadInternalDocs(), refreshStatus()]);
        } catch (error) {
            showToast(error.message, "업로드 실패");
        } finally {
            button.disabled = false;
            button.textContent = "문서 업로드 및 색인";
        }
    }

    /** ERP DB의 사업장 목록을 가져와 선택 상자를 채웁니다. */
    async function loadBranches() {
        const select = byId("branchSelect");
        if (!select) return;
        try {
            const data = await apiRequest("/api/erp/branches");
            select.replaceChildren();
            if (!data.branches?.length) {
                select.add(new Option("등록된 사업장 없음", ""));
                renderResourceEmpty("사업장 데이터가 없습니다.");
                return;
            }
            data.branches.forEach((branch) => select.add(new Option(branch.branch_name, branch.branch_id)));
            await loadErpResources();
        } catch (error) {
            select.replaceChildren(new Option("사업장 조회 실패", ""));
            showToast(error.message, "ERP 조회 실패");
        }
    }

    /** ERP 자원 목록의 공통 빈 상태 메시지를 표시합니다. */
    function renderResourceEmpty(message) {
        ["erpRobots", "erpWorkers", "erpInventory"].forEach((id) => {
            byId(id)?.replaceChildren(makeMessage(message));
        });
    }

    /** 사업장 ID로 ERP 로봇·기술자·재고 데이터를 가져옵니다. */
    async function loadErpResources() {
        const branchId = byId("branchSelect")?.value;
        if (!branchId) return;
        try {
            const data = await apiRequest(`/api/erp/resources?branch_id=${encodeURIComponent(branchId)}`);
            renderResourceRows("erpRobots", data.robots || [], (item) => ({
                name: item.serial,
                detail: `${item.model || "모델 미기재"} · 가반하중 ${item.payload_kg ?? "-"}kg`,
                status: item.status || "UNKNOWN",
            }));
            renderResourceRows("erpWorkers", data.workers || [], (item) => ({
                name: item.name,
                detail: item.skill_grade || "숙련도 미기재",
                status: item.status || "UNKNOWN",
            }));
            renderResourceRows("erpInventory", data.inventory || [], (item) => ({
                name: item.name,
                detail: `${item.qty ?? 0} ${item.unit || ""} · 안전재고 ${item.min_safety_qty ?? 0}`,
                status: item.is_low ? "재고 부족" : "재고 적정",
                warning: item.is_low,
            }));
            byId("robotCount").textContent = data.robots?.length ?? 0;
            byId("workerCount").textContent = data.workers?.length ?? 0;
            byId("inventoryCount").textContent = data.inventory?.length ?? 0;
        } catch (error) {
            renderResourceEmpty(error.message);
        }
    }

    /** ERP 리소스 배열을 카드 목록으로 변환합니다. */
    function renderResourceRows(targetId, records, mapper) {
        const target = byId(targetId);
        target.replaceChildren();
        if (!records.length) {
            target.append(makeMessage("등록된 데이터가 없습니다."));
            return;
        }
        records.forEach((record) => {
            const mapped = mapper(record);
            const row = document.createElement("div");
            row.className = "resource-row";
            const details = document.createElement("div");
            details.className = "flex-grow-1";
            const name = document.createElement("div");
            name.className = "resource-name";
            name.textContent = mapped.name || "이름 미기재";
            const meta = document.createElement("div");
            meta.className = "resource-meta";
            meta.textContent = mapped.detail;
            details.append(name, meta);
            const state = document.createElement("span");
            state.className = `resource-state${mapped.warning ? " warn" : ""}`;
            state.textContent = mapped.status;
            row.append(details, state);
            target.append(row);
        });
    }

    /** 작업 요청 등록 폼을 API 형식으로 직렬화해 DB 1에 저장합니다. */
    async function submitWorkOrder(event) {
        event.preventDefault();
        const button = byId("workOrderSubmit");
        button.disabled = true;
        button.textContent = "저장 중...";
        const payload = {
            work_area: byId("workArea").value,
            block_no: byId("blockNo").value,
            base_material: byId("baseMaterial").value,
            material_thickness: byId("materialThickness").value,
            process_type: byId("processType").value,
            welding_position: byId("weldingPosition").value,
            joint_type: byId("jointType").value,
            groove_type: byId("grooveType").value,
            request_text: byId("workRequestText").value,
        };
        try {
            const data = await apiRequest("/api/work-orders", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            showToast(`작업 요청 #${data.work_order.work_id}을 저장했습니다.`, "저장 완료");
            await Promise.all([loadWorkOrders(), refreshStatus()]);
        } catch (error) {
            showToast(error.message, "작업 요청 저장 실패");
        } finally {
            button.disabled = false;
            button.textContent = "작업 요청 저장";
        }
    }

    /** 로그인 사용자의 최근 작업 요청 이력을 표로 렌더링합니다. */
    async function loadWorkOrders() {
        const target = byId("workOrderList");
        if (!target) return;
        target.replaceChildren(makeMessage("작업 이력을 조회하고 있습니다.", "inline-loading"));
        try {
            const data = await apiRequest("/api/work-orders");
            target.replaceChildren();
            if (!data.work_orders?.length) {
                target.append(makeMessage("등록된 작업 요청이 없습니다."));
                return;
            }
            const table = document.createElement("table");
            table.className = "table table-hover align-middle";
            const head = document.createElement("thead");
            const headRow = document.createElement("tr");
            ["ID", "블록", "사업장", "모재", "두께", "상태", "등록 시각"].forEach((label) => {
                const th = document.createElement("th");
                th.scope = "col";
                th.textContent = label;
                headRow.append(th);
            });
            head.append(headRow);
            const body = document.createElement("tbody");
            data.work_orders.forEach((item) => {
                const row = document.createElement("tr");
                const values = [
                    `#${item.work_id}`,
                    item.block_no,
                    item.work_area,
                    item.base_material,
                    `${item.material_thickness ?? "-"}mm`,
                    item.work_status,
                    item.created_at ? new Date(item.created_at).toLocaleString("ko-KR") : "-",
                ];
                values.forEach((value, index) => {
                    const cell = document.createElement("td");
                    if (index === 5) {
                        const state = document.createElement("span");
                        state.className = "state-label pending";
                        state.textContent = value || "-";
                        cell.append(state);
                    } else {
                        cell.textContent = value ?? "-";
                    }
                    row.append(cell);
                });
                body.append(row);
            });
            table.append(head, body);
            target.append(table);
        } catch (error) {
            target.replaceChildren(makeMessage(error.message));
        }
    }

    /** 인증 API를 호출해 현재 브라우저 세션을 종료합니다. */
    async function logout() {
        try {
            await apiRequest("/api/auth/logout", { method: "POST" });
            window.location.href = "/login";
        } catch (error) {
            showToast(error.message, "로그아웃 실패");
        }
    }

    /** 로그인 폼의 자격 증명을 전송하고 성공 시 메인 화면으로 이동합니다. */
    async function submitLogin(event) {
        event.preventDefault();
        const errorBox = byId("loginError");
        const button = byId("loginSubmit");
        errorBox.textContent = "";
        button.disabled = true;
        try {
            await apiRequest("/api/auth/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    username: byId("loginUsername").value,
                    password: byId("loginPassword").value,
                }),
            });
            window.location.href = "/";
        } catch (error) {
            errorBox.textContent = error.message;
        } finally {
            button.disabled = false;
        }
    }

    /** 화면의 폼과 버튼 이벤트를 연결하고 초기 상태를 조회합니다. */
    function initializeWorkspace() {
        byId("refreshStatusButton")?.addEventListener("click", refreshStatus);
        byId("logoutButton")?.addEventListener("click", logout);
        byId("ragForm")?.addEventListener("submit", submitRagSearch);
        byId("uploadDocForm")?.addEventListener("submit", uploadInternalDoc);
        byId("refreshDocsButton")?.addEventListener("click", loadInternalDocs);
        byId("branchSelect")?.addEventListener("change", loadErpResources);
        byId("workOrderForm")?.addEventListener("submit", submitWorkOrder);
        byId("refreshWorkOrdersButton")?.addEventListener("click", loadWorkOrders);
        byId("loginForm")?.addEventListener("submit", submitLogin);

        refreshStatus();
        checkSession();
    }

    document.addEventListener("DOMContentLoaded", initializeWorkspace);
})();
