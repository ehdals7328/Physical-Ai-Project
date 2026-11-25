/* K-SmartShip Shared Live Vision Viewer (used by BOTH 현장 관제 & 통합 업무 > 현장·지시) */
(function (global) {
  var STAGE_LOAD_MSG = {
    transport: '운반 단계(OCR) 비전 모델 분석 중...',
    assembly: 'CCTV 조립 현장 비전 모델 분석 중...',
    welding: '로봇 용접선 탐지 영상으로 전환 중...'
  };

  function esc(str) {
    return (global.KSmartMd && global.KSmartMd.esc) ? global.KSmartMd.esc(str) : String(str == null ? '' : str);
  }

  function mount(opts) {
    var cfg = $.extend({
      imgSelector: '#live-feed',
      videoSelector: '#live-video',
      overlaySelector: '#live-feed-status',
      overlayTextSelector: '#live-feed-status-text',
      badgeSelector: '#live-source-badge',
      ocrSelector: '#live-ocr',
      listSelector: '#live-det-list',
      listMode: 'list', // 'list' (<li> rows) or 'inline' (comma-separated summary)
      useServerless: false,
      isTransportYard: false,
      initStage: 'assembly',
      cameraCode: '',
      fallbackOcr: '',
      fallbackObjects: [],
      btnPrefix: '#btn-cam-',
      autoPoll: true
    }, opts || {});

    var currentStage = cfg.initStage || (cfg.isTransportYard ? 'transport' : 'assembly');
    var isSwitching = false;
    var reqSeq = 0;
    var pollTimer = null;
    var destroyed = false;

    function showOverlay(msg, showSpinner) {
      var $ov = $(cfg.overlaySelector);
      if (!$ov.length) return;
      $(cfg.overlayTextSelector).text(msg || '불러오는 중...');
      $ov.find('.spinner-border').toggle(showSpinner !== false);
      $ov.css('display', 'flex');
    }

    function hideOverlay() {
      $(cfg.overlaySelector).hide();
    }

    function updateBadge(source) {
      var $b = $(cfg.badgeSelector);
      if (!$b.length) return;
      if (source === 'LIVE' || source === 'RUNPOD_SERVERLESS') {
        $b.attr('class', 'status-chip status-ok').text('실시간');
      } else if (source === 'LOADING' || source === 'WAIT') {
        $b.attr('class', 'status-chip status-info').text('수신 중');
      } else {
        $b.attr('class', 'status-chip status-warn').text('미연결');
      }
    }

    function renderDetections(res) {
      if (destroyed || !res) return;
      updateBadge(res.source || 'WAIT');

      var $ocr = $(cfg.ocrSelector);
      var $list = $(cfg.listSelector);

      if (res.source === 'LOADING') {
        if (cfg.listMode === 'list') {
          $list.html('<li class="py-2 text-muted">RunPod 추론 결과를 불러오는 중입니다...</li>');
        } else {
          $list.text('분석 중...');
        }
        return;
      }

      if (res.error && (!res.predictions || !res.predictions.length)) {
        $ocr.text(cfg.fallbackOcr || '연결 오류');
        if (cfg.listMode === 'list') {
          $list.html('<li class="py-2 text-danger">' + esc(res.error) + '</li>');
        } else {
          $list.text(cfg.fallbackObjects && cfg.fallbackObjects.length ? cfg.fallbackObjects.join(', ') : '연결 오류');
        }
        return;
      }

      var preds = res.predictions || [];
      var ocrItem = preds.find(function (p) { return p.ocr_text || p.block_id || p.ocr_id; });
      var ocrText = ocrItem ? (ocrItem.ocr_text || ocrItem.block_id || ocrItem.ocr_id) : (cfg.fallbackOcr || '미감지');
      $ocr.text(ocrText);

      if (!preds.length) {
        if (cfg.listMode === 'list') {
          $list.html('<li class="text-muted py-2">탐지된 객체가 없습니다.</li>');
        } else {
          $list.text(cfg.fallbackObjects && cfg.fallbackObjects.length ? cfg.fallbackObjects.join(', ') : '없음');
        }
        return;
      }

      if (cfg.listMode === 'inline') {
        var names = [];
        preds.forEach(function (p) {
          var lbl = p.class_ko || p.class || p.class_name || '객체';
          if (names.indexOf(lbl) === -1) names.push(lbl);
        });
        $list.text(names.join(', '));
      } else {
        $list.html(
          preds.map(function (p) {
            var label = p.class_ko || p.class || p.class_name || '객체';
            var confVal = (p.confidence !== undefined && p.confidence !== null) ? p.confidence : (p.conf || 0);
            var pct = confVal <= 1 ? Math.round(confVal * 100) : Math.round(confVal);
            return (
              '<li class="py-2 border-bottom d-flex justify-content-between align-items-center gap-2">' +
                '<span class="fw-semibold text-dark">' + esc(label) + '</span>' +
                '<span class="status-chip status-muted">' + pct + '%</span>' +
              '</li>'
            );
          }).join('')
        );
      }
    }

    function switchStage(mode, ignoreGating) {
      if (destroyed) return;
      if (!ignoreGating) {
        if (cfg.isTransportYard && mode !== 'transport') return;
        if (!cfg.isTransportYard && mode === 'transport') return;
      }

      currentStage = mode;
      isSwitching = true;
      reqSeq += 1;
      var mySeq = reqSeq;

      var activeModes = cfg.isTransportYard ? ['transport'] : ['assembly', 'welding'];
      activeModes.forEach(function (m) {
        var $btn = $(cfg.btnPrefix + m);
        if ($btn.length) {
          $btn.toggleClass('btn-primary', mode === m).toggleClass('btn-outline-secondary', mode !== m);
        }
      });

      updateBadge('LOADING');
      if (cfg.listMode === 'list') {
        $(cfg.listSelector).html('<li class="py-2 text-muted">' + esc(STAGE_LOAD_MSG[mode] || '전환 중...') + '</li>');
      } else {
        $(cfg.listSelector).text('수신 중...');
      }
      showOverlay(STAGE_LOAD_MSG[mode] || '비전 영상 불러오는 중...', true);

      var url = '/vision/change_video/' + encodeURIComponent(mode);
      if (cfg.cameraCode) url += '?bay_id=' + encodeURIComponent(cfg.cameraCode);

      $.getJSON(url, function (res) {
        if (destroyed || mySeq !== reqSeq) return;
        isSwitching = false;

        var $img = $(cfg.imgSelector);
        var $vid = $(cfg.videoSelector);

        if (res.video_data_url && $vid.length) {
          $img.hide();
          $vid.attr('src', res.video_data_url).show();
          hideOverlay();
        } else if (res.image_data_url) {
          if ($vid.length) $vid.hide().attr('src', '');
          $img.off('load.kv').on('load.kv', function () { hideOverlay(); });
          $img.attr('src', res.image_data_url).show();
          hideOverlay();
        } else {
          if ($vid.length) $vid.hide().attr('src', '');
          var feedUrl = '/vision/video_feed?stage=' + encodeURIComponent(mode) + '&t=' + Date.now();
          $img.off('load.kv').on('load.kv', function () { hideOverlay(); });
          $img.attr('src', feedUrl).show();
          setTimeout(hideOverlay, 1200);
        }

        if (res.detection_data) {
          renderDetections(res.detection_data);
        } else {
          pollDetections();
        }
      }).fail(function (xhr) {
        if (destroyed || mySeq !== reqSeq) return;
        isSwitching = false;
        var msg = (xhr.responseJSON && xhr.responseJSON.message) || '현장 비전 서버 연결 실패';
        var $b = $(cfg.badgeSelector);
        if ($b.length) $b.attr('class', 'status-chip status-danger').text('오류');
        showOverlay(msg, false);
        if (cfg.listMode === 'list') {
          $(cfg.listSelector).html('<li class="py-2 text-danger">' + esc(msg) + '</li>');
        }
      });
    }

    function pollDetections() {
      if (destroyed || isSwitching) return;
      $.getJSON('/vision/detections', function (res) {
        if (destroyed || isSwitching) return;
        if (res && res.stage && res.stage !== currentStage && cfg.useServerless) return;
        renderDetections(res);
      });
    }

    function destroy() {
      destroyed = true;
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    }

    // 초기화: Serverless면 switchStage로 이미지+JSON 동시 로드, Pod 스트림이면 video_feed + 폴링
    if (cfg.useServerless) {
      switchStage(currentStage, true);
      if (cfg.autoPoll) pollTimer = setInterval(pollDetections, 3000);
    } else {
      var $img = $(cfg.imgSelector);
      var feedUrl = '/vision/video_feed?stage=' + encodeURIComponent(currentStage);
      if (cfg.cameraCode) feedUrl += '&bay_id=' + encodeURIComponent(cfg.cameraCode);
      $img.off('error.kv').on('error.kv', function () {
        showOverlay('실시간 카메라 스트림에 연결할 수 없습니다. RunPod 서버 상태를 확인해 주세요.', false);
      });
      $img.off('load.kv').on('load.kv', function () { hideOverlay(); });
      $img.attr('src', feedUrl).show();
      pollDetections();
      if (cfg.autoPoll) pollTimer = setInterval(pollDetections, 2000);
    }

    return {
      switchStage: switchStage,
      poll: pollDetections,
      destroy: destroy
    };
  }

  global.KSmartVision = { mount: mount };
})(window);
