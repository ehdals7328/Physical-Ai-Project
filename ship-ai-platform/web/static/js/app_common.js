/**
 * K-SmartShip Enterprise Common Script (jQuery 3.7.1)
 * - 전체 펼침 메가메뉴 (Hover + 클릭)
 * - 전역 선택 선박/블록 컨텍스트 유지 (sessionStorage)
 * - 모든 HTML 동일 세션 전역 AI 챗봇 (실제 LangGraph 백엔드 /service/api/assistant/* 연동)
 */
$(function () {
  var CTX_KEY = 'ksmartship_active_work_ctx_v2';
  var CHAT_OPEN_KEY = 'ksmartship_chat_open_v2';
  var isLoggedIn = $('body').data('logged-in') === 1 || $('body').data('logged-in') === '1';
  var pollTimer = null;
  var activeSessionId = null;

  /* ---------- 1. 메가메뉴 & 계정 드롭다운 ---------- */
  var megaTimer = null;
  function openMega() {
    clearTimeout(megaTimer);
    $('#mega-menu-panel').stop(true, true).slideDown(140);
  }
  function closeMega() {
    megaTimer = setTimeout(function () {
      $('#mega-menu-panel').stop(true, true).slideUp(120);
    }, 130);
  }

  $('#site-header-nav, #mega-menu-panel')
    .on('mouseenter', openMega)
    .on('mouseleave', closeMega);

  $('#btn-account-toggle').on('click', function (e) {
    e.preventDefault();
    e.stopPropagation();
    $('#account-dropdown-menu').toggle();
  });

  $(document).on('click', function () {
    $('#account-dropdown-menu').hide();
  });

  /* ---------- 2. 전역 선택 선박·블록 작업 컨텍스트 ---------- */
  window.ShipContext = {
    get: function () {
      try {
        return JSON.parse(sessionStorage.getItem(CTX_KEY) || 'null');
      } catch (e) {
        return null;
      }
    },
    set: function (obj) {
      if (!obj) return;
      sessionStorage.setItem(CTX_KEY, JSON.stringify(obj));
      renderContextRibbon();
      updateChatContextBar();
    },
    clear: function () {
      sessionStorage.removeItem(CTX_KEY);
      renderContextRibbon();
      updateChatContextBar();
    }
  };

  function getPageContext() {
    var pageCtx = {
      page_title: document.title,
      url: window.location.pathname + window.location.search
    };
    var el = document.getElementById('page-ai-context');
    if (el) {
      try {
        pageCtx = $.extend(pageCtx, JSON.parse(el.textContent));
      } catch (e) {}
    }
    var saved = window.ShipContext.get();
    if (saved) {
      pageCtx.selected_work = saved;
    }
    return pageCtx;
  }

  function renderContextRibbon() {
    var saved = window.ShipContext.get();
    var $ribbon = $('#global-context-ribbon');
    if (!$ribbon.length) return;
    if (saved && saved.block_id) {
      $('#ctx-ship-label').text(
        (saved.branch_name || saved.branch_code || '') +
        ' · 선박 ' + (saved.ship_code || '-') +
        ' · 블록 ' + saved.block_id
      );
      $ribbon.show();
    } else {
      $ribbon.hide();
    }
  }

  $('#btn-clear-context').on('click', function () {
    window.ShipContext.clear();
  });

  renderContextRibbon();

  /* ---------- 3. 전역 우측 하단 AI 챗봇 (실제 서버 세션 동기화) ---------- */
  function updateChatContextBar() {
    var ctx = getPageContext();
    var label = '현재 화면: ' + (ctx.page_title || window.location.pathname);
    if (ctx.selected_work && ctx.selected_work.block_id) {
      label += ' | 대상: ' + ctx.selected_work.ship_code + ' / ' + ctx.selected_work.block_id;
    }
    $('#ai-chat-ctx-text').text(label);
  }

  function appendBubble(role, text, extraMeta) {
    var $body = $('#ai-chat-body');
    var $wrap = $('<div>').addClass('ai-bubble ' + role);
    $wrap.append($('<div>').text(text));

    if (extraMeta && extraMeta.files) {
      var $links = $('<div class="mt-2 pt-2 border-top d-flex flex-wrap gap-2">');
      if (extraMeta.files.pdf_url) {
        $links.append($('<a>').attr({ href: extraMeta.files.pdf_url, target: '_blank' })
          .addClass('btn btn-outline-primary btn-sm py-0 px-2').text('PDF 미리보기'));
      }
      if (extraMeta.files.docx_url) {
        $links.append($('<a>').attr({ href: extraMeta.files.docx_url, target: '_blank' })
          .addClass('btn btn-outline-secondary btn-sm py-0 px-2').text('DOCX 다운로드'));
      }
      if (extraMeta.approval_status === 'DRAFT') {
        $links.append($('<a>').attr({ href: '/service/erp#hitl-section' })
          .addClass('btn btn-dark btn-sm py-0 px-2').text('승인함에서 검토'));
      }
      $wrap.append($links);
    }

    $body.append($wrap);
    $body.scrollTop($body[0].scrollHeight);
  }

  function showLoadingBubble() {
    removeLoadingBubble();
    var $l = $('<div id="ai-loading-bubble" class="ai-bubble assistant text-muted"></div>')
      .text('현재 화면 정보와 권한별 지식(RAG)·ERP 데이터를 분석하고 있습니다...');
    $('#ai-chat-body').append($l);
    $('#ai-chat-body').scrollTop($('#ai-chat-body')[0].scrollHeight);
  }

  function removeLoadingBubble() {
    $('#ai-loading-bubble').remove();
  }

  function startPolling(sessionId) {
    if (!sessionId) return;
    activeSessionId = sessionId;
    clearInterval(pollTimer);
    showLoadingBubble();
    pollTimer = setInterval(function () {
      $.getJSON('/service/api/assistant/status/' + encodeURIComponent(sessionId), function (res) {
        if (res.status === 'RUNNING') return;
        clearInterval(pollTimer);
        pollTimer = null;
        removeLoadingBubble();
        if (res.answer) {
          appendBubble('assistant', res.answer, {
            files: res.files,
            approval_status: res.approval_status
          });
        }
      }).fail(function () {
        clearInterval(pollTimer);
        pollTimer = null;
        removeLoadingBubble();
      });
    }, 1500);
  }

  function loadServerHistory() {
    var $body = $('#ai-chat-body');
    if (!isLoggedIn) {
      $body.empty();
      appendBubble('assistant', '로그인하시면 현재 화면의 선박·블록 정보와 권한별 사내 규정(RAG)·ERP를 연동해 답변해 드립니다.');
      return;
    }
    $.getJSON('/service/api/assistant/history', function (res) {
      $body.empty();
      activeSessionId = res.session_id;
      if (!res.messages || !res.messages.length) {
        appendBubble('assistant', '현재 보고 계신 화면과 선택된 선박·블록을 인지했습니다. 용접 규정, 자재 발주, 자격증 메일 안내, 작업지시서 작성 등 필요한 업무를 입력해 주세요.');
      } else {
        $.each(res.messages, function (_, m) {
          appendBubble(m.role, m.content, { files: m.files, approval_status: m.approval_status });
        });
      }
      if (res.is_running) {
        startPolling(res.session_id);
      }
    }).fail(function () {
      $body.empty();
      appendBubble('assistant', '대화 이력을 불러오지 못했습니다. 로그인 상태를 확인해 주세요.');
    });
  }

  function setChatOpen(open) {
    if (open) {
      $('#ai-chat-panel').addClass('open');
      sessionStorage.setItem(CHAT_OPEN_KEY, '1');
      updateChatContextBar();
      $('#ai-chat-input').focus();
    } else {
      $('#ai-chat-panel').removeClass('open');
      sessionStorage.setItem(CHAT_OPEN_KEY, '0');
    }
  }

  function sendChatMessage(customPrompt) {
    var text = $.trim(customPrompt !== undefined ? customPrompt : $('#ai-chat-input').val());
    if (!text) return;
    if (!isLoggedIn) {
      setChatOpen(true);
      appendBubble('assistant', '업무 조회 및 실행은 사내 계정 로그인 후 이용할 수 있습니다.');
      return;
    }
    if (customPrompt === undefined) $('#ai-chat-input').val('');

    appendBubble('user', text);
    showLoadingBubble();

    $.ajax({
      url: '/service/api/assistant/ask',
      method: 'POST',
      contentType: 'application/json',
      data: JSON.stringify({
        question: text,
        page_context: getPageContext()
      }),
      success: function (res) {
        if (res.session_id) {
          startPolling(res.session_id);
        }
      },
      error: function (xhr) {
        removeLoadingBubble();
        var msg = (xhr.responseJSON && xhr.responseJSON.message) || '요청 처리 중 오류가 발생했습니다.';
        appendBubble('assistant', msg);
      }
    });
  }

  $('#ai-launcher-btn').on('click', function () {
    setChatOpen(!$('#ai-chat-panel').hasClass('open'));
  });
  $('#ai-chat-close').on('click', function () { setChatOpen(false); });
  $('#ai-chat-reset').on('click', function () {
    if (!isLoggedIn) return;
    $.post('/service/api/assistant/reset', function () {
      clearInterval(pollTimer);
      loadServerHistory();
    });
  });
  $('#ai-chat-send').on('click', function () { sendChatMessage(); });
  $('#ai-chat-input').on('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage();
    }
  });

  window.openAssistantWithPrompt = function (promptText, autoSend) {
    setChatOpen(true);
    if (autoSend !== false) {
      sendChatMessage(promptText);
    } else {
      $('#ai-chat-input').val(promptText).focus();
    }
  };
  // 구버전 호출 호환 alias
  window.openChatWithPrompt = window.openAssistantWithPrompt;

  /* ---------- 4. 스마트 페이지네이션 HTML 헬퍼 (10페이지 초과 시 ...) ---------- */
  window.buildPaginationHtml = function (currentPage, totalPages, pageItems, fnName) {
    var html = '<ul class="pagination pagination-sm justify-content-center mb-0">';
    html += '<li class="page-item ' + (currentPage <= 1 ? 'disabled' : '') + '">' +
      '<a class="page-link" href="javascript:void(0)" onclick="' + fnName + '(' + Math.max(1, currentPage - 1) + ')">이전</a></li>';
    $.each(pageItems, function (_, it) {
      if (it === '...') {
        html += '<li class="page-item disabled"><span class="page-link">...</span></li>';
      } else {
        html += '<li class="page-item ' + (it === currentPage ? 'active' : '') + '">' +
          '<a class="page-link" href="javascript:void(0)" onclick="' + fnName + '(' + it + ')">' + it + '</a></li>';
      }
    });
    html += '<li class="page-item ' + (currentPage >= totalPages ? 'disabled' : '') + '">' +
      '<a class="page-link" href="javascript:void(0)" onclick="' + fnName + '(' + Math.min(totalPages, currentPage + 1) + ')">다음</a></li>';
    html += '</ul>';
    return html;
  };

  updateChatContextBar();
  loadServerHistory();
  if (sessionStorage.getItem(CHAT_OPEN_KEY) === '1') {
    $('#ai-chat-panel').addClass('open');
  }
});
