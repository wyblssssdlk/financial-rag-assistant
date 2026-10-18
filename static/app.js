'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const HISTORY_KEY = 'chengming-financial-chat-v1';
  const sourceEmpty = $('sources').firstElementChild.cloneNode(true);
  let history = [], status = null, sending = false, initializing = false, storageFailed = false;

  function node(tag, className, text) {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (text !== undefined) item.textContent = text;
    return item;
  }
  function icon(name) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
    use.setAttribute('href', '#i-' + name);
    svg.setAttribute('aria-hidden', 'true');
    svg.append(use);
    return svg;
  }
  function notice(message, success = false) {
    $('notice').textContent = message;
    $('notice').className = 'notice' + (success ? ' success' : '');
    $('notice').hidden = !message;
  }
  async function api(path, body) {
    const response = await fetch(path, body === undefined ? {cache: 'no-store'} : {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '请输入 1–2000 字的财报问题。');
    return data;
  }
  function saveHistory() {
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(-60))); }
    catch { storageFailed = true; notice('浏览器无法保存聊天记录，本次会话仍可正常使用。'); }
  }
  function loadHistory() {
    try {
      const data = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
      if (!Array.isArray(data)) throw new Error('invalid history');
      history = data.filter(item => item && ['user', 'assistant'].includes(item.role) && typeof item.text === 'string')
        .slice(-60).map(item => ({...item, sources: Array.isArray(item.sources) ? item.sources.filter(source => source && typeof source.text === 'string') : []}));
    } catch { history = []; notice('未能读取历史记录，已打开新的会话。'); }
  }

  // Build Markdown nodes explicitly. Neither answers nor retrieved documents are
  // inserted as HTML, so document scripts, handlers and unsafe links cannot run.
  function inline(parent, text) {
    const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
    for (const part of parts) {
      if (part.startsWith('**') && part.endsWith('**')) parent.append(node('strong', '', part.slice(2, -2)));
      else if (part.startsWith('`') && part.endsWith('`')) parent.append(node('code', '', part.slice(1, -1)));
      else parent.append(document.createTextNode(part));
    }
  }
  function markdown(text) {
    const container = node('div', 'answer-body');
    const lines = String(text).replace(/\r\n/g, '\n').split('\n');
    let list = null, nextOrderedNumber = 1;
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (!line.trim()) { list = null; continue; }
      if (line.startsWith('```')) {
        const code = [];
        while (++i < lines.length && !lines[i].startsWith('```')) code.push(lines[i]);
        container.append(node('pre', '', code.join('\n'))); list = null; continue;
      }
      if (line.includes('|') && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}:?\s*\|/.test(lines[i + 1])) {
        const wrap = node('div', 'answer-table-wrap'), table = node('table');
        function cells(rowText, type) {
          const tr = node('tr');
          rowText.trim().replace(/^\||\|$/g, '').split('|').forEach(value => {
            const td = node(type); inline(td, value.trim()); tr.append(td);
          });
          return tr;
        }
        const head = node('thead'); head.append(cells(line, 'th')); table.append(head);
        const body = node('tbody'); i += 2;
        while (i < lines.length && lines[i].includes('|') && lines[i].trim()) body.append(cells(lines[i++], 'td'));
        i--; table.append(body); wrap.append(table); container.append(wrap); list = null; continue;
      }
      const heading = line.match(/^(#{1,6})\s+(.+)$/);
      const bullet = line.match(/^\s*(?:([-*])|(\d+)[.)])\s+(.+)$/);
      if (heading) { const el = node('h' + Math.min(heading[1].length + 1, 4)); inline(el, heading[2]); container.append(el); list = null; nextOrderedNumber = 1; }
      else if (bullet) {
        const type = bullet[1] ? 'ul' : 'ol';
        const authoredNumber = type === 'ol' ? Number(bullet[2]) : null;
        const number = authoredNumber === 1 ? nextOrderedNumber : authoredNumber;
        if (!list || list.tagName.toLowerCase() !== type) {
          list = node(type);
          if (type === 'ol') list.start = number;
          container.append(list);
        }
        const li = node('li');
        if (type === 'ol') { li.value = number; nextOrderedNumber = number + 1; }
        else nextOrderedNumber = 1;
        inline(li, bullet[3]); list.append(li);
      } else { const p = node('p'); inline(p, line); container.append(p); list = null; }
    }
    return container;
  }
  function readableSource(text) {
    return text.replace(/<(script|style)\b[^>]*>[\s\S]*?<\/\1>/gi, '')
      .replace(/<\/tr>/gi, '\n').replace(/<\/t[dh]>/gi, '  |  ')
      .replace(/<[^>]+>/g, '').replace(/&nbsp;/gi, ' ').replace(/&amp;/gi, '&')
      .replace(/&lt;/gi, '<').replace(/&gt;/gi, '>').replace(/&quot;/gi, '"');
  }
  function showSources(sources) {
    $('sources').replaceChildren();
    $('sourceCount').textContent = String(sources.length).padStart(2, '0');
    if (!sources.length) { $('sources').append(sourceEmpty.cloneNode(true)); return; }
    sources.forEach((source, index) => {
      const card = node('details', 'source-card'), summary = node('summary');
      card.open = index === 0;
      const head = node('div', 'source-card-head');
      head.append(node('span', 'source-number', String(index + 1).padStart(2, '0')),
        node('h3', '', String(source.source || '财报片段').replace(/_content_list\.json$/, '.pdf')));
      const text = readableSource(source.text);
      summary.append(head, node('p', '', text)); card.append(summary, node('div', 'source-text', text));
      const meta = node('div', 'source-meta');
      meta.append(node('span', '', Number.isInteger(source.page_idx) ? `第 ${source.page_idx + 1} 页` : '财报原文片段'),
        node('span', '', Number.isFinite(source.distance) ? `检索距离 ${source.distance.toFixed(4)}` : '来源已关联'));
      card.append(meta); $('sources').append(card);
    });
  }
  function openSources(sources, mobile = false) {
    showSources(sources);
    if (mobile && window.innerWidth <= 1020) {
      $('sourcePanel').classList.add('mobile-open'); $('toggleSources').setAttribute('aria-expanded', 'true');
    }
  }
  function appendMessage(item) {
    const article = node('article', 'message message-' + item.role + (item.error ? ' message-error' : ''));
    const label = node('div', 'message-label');
    if (item.role === 'assistant') label.append(icon('spark'));
    label.append(node('span', '', item.role === 'user' ? '我的问题' : '澄明 · 财报研究助手'));
    article.append(label);
    if (item.role === 'user') article.append(node('div', 'user-bubble', item.text));
    else {
      article.append(markdown(item.text));
      const actions = node('div', 'answer-actions'), buttons = node('div');
      const sourceButton = node('button', 'text-button', `查看 ${item.sources.length} 个来源`);
      sourceButton.prepend(icon('link')); sourceButton.onclick = () => openSources(item.sources, true);
      const copy = node('button', 'text-button', '复制'); copy.prepend(icon('copy'));
      copy.onclick = async () => {
        try { await navigator.clipboard.writeText(item.text); notice('已复制回答。', true); }
        catch { notice('复制失败，请选中回答文字后复制。'); }
      };
      buttons.append(sourceButton, copy);
      actions.append(buttons, node('span', '', Number.isFinite(item.elapsed_seconds) ? `${item.elapsed_seconds.toFixed(1)} 秒 · 基于资料检索` : ''));
      article.append(actions);
    }
    $('messages').append(article);
    return article;
  }
  function renderHistory() {
    $('messages').replaceChildren(); $('welcome').hidden = history.length > 0;
    history.forEach(appendMessage);
    const latest = [...history].reverse().find(item => item.role === 'assistant');
    showSources(latest ? latest.sources : []);
  }
  function scrollBottom() { $('conversation').scrollTop = $('conversation').scrollHeight; }
  function setBusy(value) {
    sending = value; $('send').disabled = value; $('newChat').disabled = value;
    $('question').readOnly = value;
    $('send').setAttribute('aria-label', value ? '正在生成回答' : '发送问题');
  }
  async function sendQuestion(event) {
    event.preventDefault();
    const question = $('question').value.trim();
    if (!question || sending) return;
    if (!status || status.state !== 'ready') {
      notice(!status ? '无法连接本地服务，请确认启动窗口仍在运行。' :
        status.state === 'error' ? `知识库初始化未完成：${status.message}` :
        status.state === 'initializing' ? '知识库正在初始化，请等到显示“知识库就绪”后再发送。' :
        '请先点击“初始化知识库”，就绪后即可提问。');
      return;
    }
    notice(''); setBusy(true); $('welcome').hidden = true;
    const user = {role: 'user', text: question, sources: []}; history.push(user); appendMessage(user);
    $('question').value = ''; $('question').style.height = '';
    const pending = node('div', 'thinking'); pending.setAttribute('role', 'status');
    pending.append(node('i'), node('i'), node('i'), node('span', '', '正在检索资料并整理回答…'));
    $('messages').append(pending); scrollBottom(); saveHistory();
    try {
      const result = await api('/api/ask', {question});
      const answer = {role: 'assistant', text: result.answer, sources: result.sources || [], elapsed_seconds: result.elapsed_seconds};
      history.push(answer); pending.remove(); appendMessage(answer); openSources(answer.sources); saveHistory();
    } catch (error) {
      pending.remove(); $('question').value = question;
      const answer = {role: 'assistant', text: error.message || '连接失败，请检查服务是否仍在运行。', sources: [], error: true};
      history.push(answer); appendMessage(answer); showSources([]); saveHistory();
      notice('问题已保留在输入框中，可以重试。');
    } finally { setBusy(false); scrollBottom(); $('question').focus(); }
  }

  function reportTitle(name) {
    const parts = name.replace(/\.pdf$/i, '').split('__');
    return parts.length >= 6 ? `${parts[1]} · ${parts[4]}${parts[5]}` : name.replace(/\.pdf$/i, '');
  }
  function drawStatus(data) {
    status = data;
    $('fileCount').textContent = data.files.length;
    $('chunkCount').textContent = data.state === 'ready' ? data.chunk_count.toLocaleString() : '—';
    $('dbMessage').textContent = data.message;
    $('dbDot').className = 'status-dot ' + data.state;
    const labels = {idle: '等待初始化', initializing: '正在初始化', ready: '知识库就绪', error: '需要处理'};
    $('connectionText').textContent = labels[data.state] || '准备中';
    $('connection').firstElementChild.className = 'status-dot ' + data.state;
    $('initialize').disabled = initializing || ['initializing', 'ready'].includes(data.state);
    $('initialize').replaceChildren(document.createTextNode(data.state === 'ready' ? '知识库已就绪' : data.state === 'initializing' ? '正在初始化…' : data.state === 'error' ? '重新初始化' : '初始化知识库'), icon('arrow'));
    $('modelName').textContent = data.model;
    $('activeReport').textContent = data.files.length ? reportTitle(data.files[0].name) : '请将财报放入 my_pdfs 文件夹';
    $('activeReport').title = data.files.map(file => file.name).join('\n');
    $('fileList').replaceChildren();
    for (const file of data.files) {
      const item = node('div', 'file-item'), detail = node('div');
      item.title = file.name;
      detail.append(node('strong', '', reportTitle(file.name)), node('small', '', `PDF · ${(file.size / 1048576).toFixed(1)} MB`));
      item.append(icon('file'), detail); $('fileList').append(item);
    }
    if (!data.files.length) $('fileList').append(node('p', 'sidebar-muted', '暂无财报。将 PDF 放入项目的 my_pdfs 文件夹。'));
  }
  async function refreshStatus() {
    try { drawStatus(await api('/api/status')); }
    catch { $('connectionText').textContent = '连接中断'; $('dbMessage').textContent = '请确认本地服务仍在运行。'; status = null; }
  }
  $('initialize').onclick = async () => {
    initializing = true; $('initialize').disabled = true; notice('');
    try { await api('/api/initialize', {}); }
    catch (error) { notice(error.message || '无法连接本地服务。'); }
    finally { initializing = false; await refreshStatus(); }
  };
  $('newChat').onclick = () => {
    history = []; saveHistory(); renderHistory(); $('question').value = ''; $('question').style.height = '';
    if (!storageFailed) notice(''); $('question').focus();
  };
  $('questionForm').addEventListener('submit', sendQuestion);
  $('question').addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('questionForm').requestSubmit(); }
  });
  $('question').addEventListener('input', () => { $('question').style.height = 'auto'; $('question').style.height = Math.min($('question').scrollHeight, 110) + 'px'; });
  document.querySelectorAll('.suggestion').forEach(button => button.onclick = () => {
    $('question').value = button.dataset.question; $('question').focus();
  });
  $('toggleSources').onclick = () => {
    const visible = $('sourcePanel').classList.toggle('mobile-open');
    $('toggleSources').setAttribute('aria-expanded', String(visible));
  };
  document.addEventListener('keydown', event => { if (event.key === 'Escape') { $('sourcePanel').classList.remove('mobile-open'); $('toggleSources').setAttribute('aria-expanded', 'false'); } });
  loadHistory(); renderHistory(); scrollBottom(); refreshStatus();
  setInterval(refreshStatus, 2000);
})();
