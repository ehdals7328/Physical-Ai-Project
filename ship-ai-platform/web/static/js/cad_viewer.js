/* K-SmartShip Interactive 3D IGES Wireframe Viewer (Zero External Dependencies) */
(function (global) {
  function mountIgesViewer(containerEl, wireframeData, metaInfo) {
    if (!containerEl || !wireframeData || !wireframeData.segments || !wireframeData.segments.length) {
      return;
    }
    var segs = wireframeData.segments;
    var bbox = wireframeData.raw_bbox || null;

    containerEl.innerHTML =
      '<div class="p-3 bg-white">' +
        '<div class="d-flex justify-content-between align-items-center flex-wrap gap-2 mb-2">' +
          '<div class="small text-secondary">마우스 드래그: 360° 회전 · 휠: 확대/축소' +
            (bbox ? ' · 치수(X×Y×Z): <strong>' + bbox.x + ' × ' + bbox.y + ' × ' + bbox.z + '</strong>' : '') +
          '</div>' +
          '<div class="btn-group btn-group-sm">' +
            '<button type="button" class="btn btn-outline-secondary" data-view="iso">3D 등각</button>' +
            '<button type="button" class="btn btn-outline-secondary" data-view="side">측면</button>' +
            '<button type="button" class="btn btn-outline-secondary" data-view="top">평면</button>' +
            '<button type="button" class="btn btn-outline-secondary" data-view="front">정면</button>' +
          '</div>' +
        '</div>' +
        '<canvas class="w-100 rounded border" style="height:500px;background:#0b192f;cursor:grab;display:block;"></canvas>' +
      '</div>';

    var canvas = containerEl.querySelector('canvas');
    var ctx = canvas.getContext('2d');
    var rotX = 0.42;
    var rotY = -0.68;
    var zoom = 1.0;
    var dragging = false;
    var lastX = 0;
    var lastY = 0;

    function resize() {
      var rect = canvas.getBoundingClientRect();
      var dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(320, Math.floor(rect.width * dpr));
      canvas.height = Math.max(320, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    function project(x, y, z, w, h) {
      var cy = Math.cos(rotY), sy = Math.sin(rotY);
      var cx = Math.cos(rotX), sx = Math.sin(rotX);
      var x1 = x * cy - z * sy;
      var z1 = x * sy + z * cy;
      var y2 = y * cx - z1 * sx;
      var z2 = y * sx + z1 * cx;
      var dist = 3.2;
      var persp = dist / (dist - z2 * 0.45);
      var scale = Math.min(w, h) * 0.36 * zoom * persp;
      return [w / 2 + x1 * scale, h / 2 - y2 * scale];
    }

    function draw() {
      var w = canvas.clientWidth || 800;
      var h = canvas.clientHeight || 500;
      ctx.clearRect(0, 0, w, h);

      // 미세한 설계 격자 배경
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.12)';
      ctx.lineWidth = 1;
      for (var gx = 0; gx < w; gx += 40) {
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
      }
      for (var gy = 0; gy < h; gy += 40) {
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
      }

      // 3D 선체 와이어프레임 렌더링
      ctx.strokeStyle = '#60a5fa';
      ctx.lineWidth = 1.15;
      ctx.beginPath();
      for (var i = 0; i < segs.length; i++) {
        var s = segs[i];
        var p1 = project(s[0], s[1], s[2], w, h);
        var p2 = project(s[3], s[4], s[5], w, h);
        ctx.moveTo(p1[0], p1[1]);
        ctx.lineTo(p2[0], p2[1]);
      }
      ctx.stroke();
    }

    canvas.addEventListener('mousedown', function (e) {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      canvas.style.cursor = 'grabbing';
    });
    window.addEventListener('mouseup', function () {
      dragging = false;
      canvas.style.cursor = 'grab';
    });
    window.addEventListener('mousemove', function (e) {
      if (!dragging) return;
      var dx = e.clientX - lastX;
      var dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      rotY += dx * 0.008;
      rotX += dy * 0.008;
      draw();
    });
    canvas.addEventListener('wheel', function (e) {
      e.preventDefault();
      zoom = Math.max(0.4, Math.min(3.5, zoom * (e.deltaY < 0 ? 1.1 : 0.9)));
      draw();
    }, { passive: false });

    containerEl.querySelectorAll('[data-view]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var v = btn.getAttribute('data-view');
        zoom = 1.0;
        if (v === 'iso') { rotX = 0.42; rotY = -0.68; }
        else if (v === 'side') { rotX = 0; rotY = 0; }
        else if (v === 'top') { rotX = Math.PI / 2; rotY = 0; }
        else if (v === 'front') { rotX = 0; rotY = Math.PI / 2; }
        draw();
      });
    });

    setTimeout(resize, 60);
  }

  global.KSmartCad = { mountIgesViewer: mountIgesViewer };
})(window);
