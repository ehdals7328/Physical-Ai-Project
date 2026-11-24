/* K-SmartShip Shared Markdown Renderer & HTML Escaper */
(function (global) {
  function escHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function formatInlineMd(text) {
    var s = escHtml(text);
    s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
    s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    return s;
  }

  function renderRichMarkdown(rawText) {
    if (!rawText) {
      return '<div class="text-muted">표시할 내용이 없습니다.</div>';
    }
    var normalized = String(rawText).replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    var lines = normalized.split('\n');
    var html = '';
    var i = 0;

    while (i < lines.length) {
      var line = lines[i];
      var trimmed = line.trim();

      if (!trimmed) {
        i++;
        continue;
      }

      if (/^-{3,}$/.test(trimmed) || /^\*{3,}$/.test(trimmed)) {
        html += '<hr>';
        i++;
        continue;
      }

      if (
        trimmed.indexOf('|') === 0 &&
        i + 1 < lines.length &&
        /^\|[\s:|-]+\|?$/.test(lines[i + 1].trim())
      ) {
        var headers = trimmed.replace(/^\|/, '').replace(/\|$/, '').split('|').map(function (c) {
          return c.trim();
        });
        i += 2;
        var rows = [];
        while (i < lines.length && lines[i].trim().indexOf('|') === 0) {
          var cols = lines[i].trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(function (c) {
            return c.trim();
          });
          rows.push(cols);
          i++;
        }
        html += '<div class="md-table-wrap"><table><thead><tr>';
        for (var h = 0; h < headers.length; h++) {
          html += '<th>' + formatInlineMd(headers[h]) + '</th>';
        }
        html += '</tr></thead><tbody>';
        for (var r = 0; r < rows.length; r++) {
          html += '<tr>';
          for (var c = 0; c < rows[r].length; c++) {
            html += '<td>' + formatInlineMd(rows[r][c]) + '</td>';
          }
          html += '</tr>';
        }
        html += '</tbody></table></div>';
        continue;
      }

      var hMatch = trimmed.match(/^(#{1,4})\s+(.*)$/);
      if (hMatch) {
        var lvl = Math.min(4, Math.max(2, hMatch[1].length));
        html += '<h' + lvl + '>' + formatInlineMd(hMatch[2]) + '</h' + lvl + '>';
        i++;
        continue;
      }

      if (trimmed.indexOf('>') === 0) {
        var qLines = [];
        while (i < lines.length && lines[i].trim().indexOf('>') === 0) {
          qLines.push(formatInlineMd(lines[i].trim().replace(/^>\s?/, '')));
          i++;
        }
        html += '<blockquote>' + qLines.join('<br>') + '</blockquote>';
        continue;
      }

      if (/^[-*•]\s+/.test(trimmed)) {
        html += '<ul>';
        while (i < lines.length && /^[-*•]\s+/.test(lines[i].trim())) {
          html += '<li>' + formatInlineMd(lines[i].trim().replace(/^[-*•]\s+/, '')) + '</li>';
          i++;
        }
        html += '</ul>';
        continue;
      }

      if (/^\d+\.\s+/.test(trimmed)) {
        html += '<ol>';
        while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
          html += '<li>' + formatInlineMd(lines[i].trim().replace(/^\d+\.\s+/, '')) + '</li>';
          i++;
        }
        html += '</ol>';
        continue;
      }

      html += '<p>' + formatInlineMd(line) + '</p>';
      i++;
    }

    return '<div class="md-render">' + html + '</div>';
  }

  global.KSmartMd = {
    esc: escHtml,
    render: renderRichMarkdown
  };
})(window);
