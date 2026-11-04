/* ==========================================================================
   Clip Maker — AI clipping workspace (frontend)
   Talks to the FastAPI backend in backend/ (routes/ai_clipping.py).
   Every control here is wired to a real endpoint: link inspection, chunked
   upload, the 7-stage job, all moment actions, renders, exports, scheduling.
   ========================================================================== */
(function () {
  'use strict';

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const esc = (s) => String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

  let API = '';
  const PAGE = 6;

  const state = {
    options: null,
    credits: { balance: 0, plan: 'free', unlimited: false },
    settings: {
      language: 'auto', clip_length: 'auto', count: 6, platform: 'all',
      instruction: '', clip_kind: 'normal', layout: 'Fit',
      // No colour picker any more: word highlight always renders white text
      // with a single yellow word, and the other presets use the white/black
      // 'classic' palette.
      style: { motion: 'moving', palette: 'classic', font: 'Montserrat', size: 'medium', placement: 'center', preset: 'classic' },
    },
    placeTouched: false,      // has the user picked a placement themselves?
    source: null,             // inspected URL or finished upload
    sourceMode: 'url',
    uploading: null,          // { upload_id, file, offset }
    job: null,
    video: null,
    moments: [],
    selected: new Set(),
    page: 1,
    sort: 'score',
    filters: { minScore: 0, durMin: 0, durMax: 180, speaker: '', keyword: '' },
    apiReady: false,
  };

  /* ------------------------------------------------------------- helpers */
  const fmt = {
    time(sec) {
      sec = Math.max(0, Number(sec) || 0);
      const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
      return `${m}:${String(s).padStart(2, '0')}`;
    },
    dur(sec) { return `${(Number(sec) || 0).toFixed(1)}s`; },
    bytes(n) {
      const b = Number(n) || 0;
      if (b > 1024 ** 3) return (b / 1024 ** 3).toFixed(2) + ' GB';
      if (b > 1024 ** 2) return (b / 1024 ** 2).toFixed(1) + ' MB';
      return Math.round(b / 1024) + ' KB';
    },
    when(ts) {
      const d = new Date((Number(ts) || 0) * 1000);
      return d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    },
    day(ts) {
      const d = new Date((Number(ts) || 0) * 1000);
      return d.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' });
    },
  };

  function absUrl(u) {
    if (!u) return '';
    if (/^https?:/.test(u)) return u;
    return (API || location.origin) + u;
  }

  function toast(title, msg = '', kind = '') {
    const wrap = $('#toasts');
    const el = document.createElement('div');
    el.className = 'toast ' + kind;
    el.innerHTML = `<b>${esc(title)}</b>${msg ? `<small>${esc(msg)}</small>` : ''}`;
    wrap.appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; el.style.transform = 'translateY(8px)'; }, 4200);
    setTimeout(() => el.remove(), 4700);
  }

  async function api(path, opts = {}) {
    const init = { ...opts };
    if (init.body && !(init.body instanceof FormData) && typeof init.body !== 'string') {
      init.headers = { 'Content-Type': 'application/json', ...(init.headers || {}) };
      init.body = JSON.stringify(init.body);
    }
    const res = await fetch(API + path, init);
    const type = res.headers.get('content-type') || '';
    const data = type.includes('json') ? await res.json() : await res.text();
    if (!res.ok) {
      const detail = data && data.detail !== undefined ? data.detail : data;
      const err = new Error(
        typeof detail === 'string' ? detail
          : (detail && detail.message) ? detail.message
            : `HTTP ${res.status}`);
      err.detail = detail;
      err.status = res.status;
      throw err;
    }
    return data;
  }

  function openModal({ title, body, footer = '', wide = false, onMount }) {
    const back = document.createElement('div');
    back.className = 'modal-back';
    back.innerHTML = `<div class="modal ${wide ? 'wide' : ''}" role="dialog" aria-modal="true">
        <div class="modal-head"><h3>${esc(title)}</h3>
          <button class="icon-mini" data-close aria-label="Close">✕</button></div>
        <div class="modal-body">${body}</div>
        ${footer ? `<div class="modal-foot">${footer}</div>` : ''}
      </div>`;
    $('#modalRoot').appendChild(back);
    const close = () => { back.remove(); document.removeEventListener('keydown', onKey); };
    const onKey = (e) => { if (e.key === 'Escape') close(); };
    document.addEventListener('keydown', onKey);
    back.addEventListener('click', (e) => {
      if (e.target === back || e.target.closest('[data-close]')) close();
    });
    if (onMount) onMount(back.querySelector('.modal'), close);
    return { root: back.querySelector('.modal'), close };
  }

  async function copy(text, label = 'Copied') {
    try {
      await navigator.clipboard.writeText(text);
    } catch (e) {
      const ta = document.createElement('textarea');
      ta.value = text; document.body.appendChild(ta); ta.select();
      document.execCommand('copy'); ta.remove();
    }
    toast(label, text.length > 70 ? text.slice(0, 70) + '…' : text, 'ok');
  }

  /* --------------------------------------------------------------- boot */
  async function detectApi() {
    /* SAME ORIGIN ONLY — never probe 127.0.0.1/localhost. On a public page
       those requests are what triggered Chrome's "Access other apps and
       services on this device" permission prompt, which makes visitors
       distrust the site. The Node server serves the site and the API on ONE
       origin. To use a backend hosted elsewhere, set window.CLIPOX_API
       (absolute origin, e.g. 'https://api.example.com') before this script. */
    const base = window.CLIPOX_API || '';
    try {
      const res = await fetch(base + '/api/health', { cache: 'no-store' });
      if (res.ok) { API = base; state.apiReady = true; return; }
    } catch (e) { /* no backend on this origin */ }
    state.apiReady = false;
  }

  async function boot() {
    await detectApi();
    if (!state.apiReady) {
      toast('Backend offline', 'Start the FastAPI server on port 8000 to clip videos.', 'err');
    }
    bindShell();
    bindInput();
    bindResults();
    bindLibrary();
    bindScheduler();
    await Promise.all([loadOptions(), loadCredits(), loadNotifications()]);
    route();
    setInterval(loadCredits, 45000);
  }

  function bindShell() {
    const tools = $('#toolsBtn');
    tools.addEventListener('click', (e) => {
      e.stopPropagation();
      const menu = document.createElement('div');
      menu.className = 'msg';
      menu.innerHTML = `
        <a href="clip-maker.html">🎬 Clip Maker</a>
        <a href="editor.html">✂️ AI Video Editor</a>
        <a href="studio.html#summarizer">📝 Summarizer</a>
        <a href="studio.html#v2t">💬 Video to Text</a>
        <a href="studio.html#notes">🧠 Note Taker</a>`;
      document.body.appendChild(menu);
      const rect = tools.getBoundingClientRect();
      menu.style.top = rect.bottom + 8 + 'px';
      menu.style.left = rect.left + 'px';
      menu.style.right = 'auto';
      const off = () => { menu.remove(); document.removeEventListener('click', off); };
      setTimeout(() => document.addEventListener('click', off), 0);
    });

    $('#uiLang').addEventListener('change', (e) => {
      document.documentElement.lang = e.target.value;
      localStorage.setItem('wv.uiLang', e.target.value);
      toast('Interface language set', e.target.selectedOptions[0].textContent);
    });
    const savedLang = localStorage.getItem('wv.uiLang');
    if (savedLang) { $('#uiLang').value = savedLang; document.documentElement.lang = savedLang; }

    $('#bellBtn').addEventListener('click', async () => {
      let items = [];
      try { items = (await api('/api/notifications')).items || []; } catch (e) { /* offline */ }
      openModal({
        title: 'Notifications',
        body: items.length ? items.slice(0, 20).map((n) => `
          <div class="export-row">
            <span>${n.level === 'error' ? '⛔' : n.level === 'warn' ? '⚠️' : '✅'}</span>
            <span class="grow"><b>${esc(n.title)}</b><small>${esc(n.body || '')} · ${fmt.when(n.at)}</small></span>
            ${n.link ? `<a class="btn btn-ghost btn-sm" href="${esc(n.link)}">Open</a>` : ''}
          </div>`).join('') : '<p class="muted">Nothing yet — job and export updates land here.</p>',
      });
      $('#bellDot').classList.add('hidden');
      try { await api('/api/notifications/read', { method: 'POST', body: {} }); } catch (e) { /* ignore */ }
    });

    $('#avatarBtn').addEventListener('click', () => {
      openModal({
        title: 'Account',
        body: `<div class="kv">
            <dt>Plan</dt><dd id="mPlan">—</dd>
            <dt>Credits</dt><dd id="mBalance">—</dd>
            <dt>Workspace</dt><dd>Local (this machine)</dd>
          </div>`,
        footer: `<a class="btn btn-ghost" href="editor.html#pricing">Manage plan</a>`,
        onMount(root) {
          $('#mPlan', root).textContent = state.credits.plan;
          $('#mBalance', root).textContent = state.credits.unlimited
            ? 'Unlimited (∞)' : state.credits.balance + ' credits';
        },
      });
    });
  }

  async function loadOptions() {
    try {
      const opts = await api('/api/ai-clipping/options');
      state.options = opts;
      state.credits = { balance: opts.credits.balance, plan: opts.credits.plan,
                        unlimited: !!opts.credits.unlimited };
      paintCredits();

      $('#maxSize').textContent = opts.upload.max_mb >= 1024
        ? (opts.upload.max_mb / 1024).toFixed(0) + ' GB' : opts.upload.max_mb + ' MB';
      $('#langSelect').innerHTML = opts.languages
        .map((l) => `<option value="${esc(l.key)}">${esc(l.label)}</option>`).join('');
      $('#countSelect').innerHTML = [3, 4, 5, 6, 8, 10, 12]
        .map((n) => `<option value="${n}" ${n === state.settings.count ? 'selected' : ''}>${n} clips</option>`).join('');
      $('#lenChips').innerHTML = opts.clip_lengths.map((l) =>
        `<button class="chip ${l.key === state.settings.clip_length ? 'active' : ''}" data-len="${esc(l.key)}">${esc(l.label)}</button>`).join('');
      $('#platChips').innerHTML = opts.platforms.map((p) =>
        `<button class="chip ${p.key === state.settings.platform ? 'active' : ''}" data-plat="${esc(p.key)}">${esc(p.label)}</button>`).join('');
      $('#layoutSelect').innerHTML = opts.layouts.map((l) =>
        `<option ${l === state.settings.layout ? 'selected' : ''}>${esc(l)}</option>`).join('');
      $('#motionChips').innerHTML = opts.motions.map((m) =>
        `<button class="chip ${m === state.settings.style.motion ? 'active' : ''}" data-motion="${esc(m)}">${m === 'moving' ? 'Word highlight' : 'Stationary'}</button>`).join('');
      $('#fontSelect').innerHTML = opts.fonts.map((f) =>
        `<option ${f === state.settings.style.font ? 'selected' : ''}>${esc(f)}</option>`).join('');
      $('#sizeSelect').innerHTML = opts.sizes.map((s) =>
        `<option ${s === state.settings.style.size ? 'selected' : ''}>${esc(s)}</option>`).join('');
      $('#placeChips').innerHTML = opts.placements.map((p) =>
        `<button class="chip ${p === state.settings.style.placement ? 'active' : ''}" data-place="${esc(p)}">${esc(p)}</button>`).join('');
      const presets = opts.presets || ['classic', 'hook', 'karaoke'];
      $('#presetChips').innerHTML = presets.map((p) =>
        `<button class="chip ${p === state.settings.style.preset ? 'active' : ''}" data-preset="${esc(p)}">${esc(PRESET_LABELS[p] || p)}</button>`).join('');

      bindSettingChips();
      paintStyleSummary();
      estimate();
      loadRecent();
    } catch (e) {
      toast('Could not load options', e.message, 'err');
    }
  }

  function bindSettingChips() {
    $$('#lenChips .chip').forEach((b) => b.addEventListener('click', () => {
      state.settings.clip_length = b.dataset.len;
      $$('#lenChips .chip').forEach((x) => x.classList.toggle('active', x === b));
    }));
    $$('#platChips .chip').forEach((b) => b.addEventListener('click', () => {
      state.settings.platform = b.dataset.plat;
      $$('#platChips .chip').forEach((x) => x.classList.toggle('active', x === b));
    }));
    $$('#motionChips .chip').forEach((b) => b.addEventListener('click', () => {
      state.settings.style.motion = b.dataset.motion;
      $$('#motionChips .chip').forEach((x) => x.classList.toggle('active', x === b));
      paintStyleSummary();
    }));
    $$('#placeChips .chip').forEach((b) => b.addEventListener('click', () => {
      state.placeTouched = true;
      setPlacement(b.dataset.place);
    }));
    $$('#presetChips .chip').forEach((b) => b.addEventListener('click', () => {
      applyPreset(b.dataset.preset);
    }));
  }

  function setPlacement(value) {
    state.settings.style.placement = value;
    $$('#placeChips .chip').forEach((x) => x.classList.toggle('active', x.dataset.place === value));
    paintStyleSummary();
  }

  function applyPreset(value) {
    state.settings.style.preset = value;
    $$('#presetChips .chip').forEach((x) => x.classList.toggle('active', x.dataset.preset === value));
    // Hook captions read best lower-centre; only move the placement the user
    // has not chosen themselves.
    if (!state.placeTouched) setPlacement(value === 'hook' ? 'bottom' : 'center');
    paintStyleSummary();
  }

  const PRESET_LABELS = { classic: 'Classic', hook: 'Hook style', karaoke: 'Karaoke' };

  function paintStyleSummary() {
    const s = state.settings.style;
    const cap = (v) => (v || '').charAt(0).toUpperCase() + (v || '').slice(1);
    const parts = [s.motion === 'moving' ? 'Word highlight' : 'Stationary',
                   s.font, cap(s.size), cap(s.placement)];
    if (s.preset && s.preset !== 'classic') parts.unshift(PRESET_LABELS[s.preset] || s.preset);
    $('#styleSummary').textContent = parts.join(' · ');
  }

  async function loadCredits() {
    try {
      const data = await api('/api/credits');
      state.credits = { balance: data.balance, plan: data.plan_name || data.plan,
                        unlimited: !!data.unlimited };
      paintCredits();
    } catch (e) { /* offline — the pill keeps its last value */ }
  }

  function paintCredits() {
    const unlim = !!state.credits.unlimited;
    const b = Number(state.credits.balance || 0);
    $('#creditCount').textContent = unlim ? '∞' : b.toLocaleString();
    $('#creditsPill').classList.toggle('low', !unlim && b < 20);
    const line = $('#balanceLine');
    if (line) line.textContent = unlim
      ? 'Balance: Unlimited (∞) — billing is off'
      : `Balance: ${b.toLocaleString()} credits`;
    const up = document.getElementById('upgradeBtn');
    if (up) { up.hidden = unlim; up.style.display = unlim ? 'none' : ''; }
  }

  async function loadNotifications() {
    try {
      const data = await api('/api/notifications');
      $('#bellDot').classList.toggle('hidden', !(data.unread > 0));
    } catch (e) { /* ignore */ }
  }

  /* -------------------------------------------------------------- input */
  function bindInput() {
    $$('#srcMode button').forEach((b) => b.addEventListener('click', () => {
      state.sourceMode = b.dataset.mode;
      $$('#srcMode button').forEach((x) => x.classList.toggle('active', x === b));
      $$('.pane').forEach((p) => p.classList.toggle('hidden', p.dataset.pane !== b.dataset.mode));
    }));

    let timer = null;
    $('#urlInput').addEventListener('input', () => {
      clearTimeout(timer);
      state.source = null;
      renderPreview();
      const val = $('#urlInput').value.trim();
      if (val.length < 10) return;
      timer = setTimeout(() => inspectUrl(val), 700);
    });
    $('#urlInput').addEventListener('paste', () => {
      setTimeout(() => inspectUrl($('#urlInput').value.trim()), 250);
    });

    const dz = $('#dropzone'), fi = $('#fileInput');
    dz.addEventListener('click', () => fi.click());
    dz.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') fi.click(); });
    fi.addEventListener('change', () => { if (fi.files[0]) startUpload(fi.files[0]); });
    ['dragenter', 'dragover'].forEach((ev) => dz.addEventListener(ev, (e) => {
      e.preventDefault(); dz.classList.add('drag');
    }));
    ['dragleave', 'drop'].forEach((ev) => dz.addEventListener(ev, (e) => {
      e.preventDefault(); dz.classList.remove('drag');
      if (ev === 'drop' && e.dataTransfer.files[0]) startUpload(e.dataTransfer.files[0]);
    }));
    document.addEventListener('drop', (e) => {
      if ($('#view-input').classList.contains('active') && e.dataTransfer.files[0]) {
        e.preventDefault(); startUpload(e.dataTransfer.files[0]);
      }
    });

    $('#langSelect').addEventListener('change', (e) => { state.settings.language = e.target.value; });
    $('#countSelect').addEventListener('change', (e) => { state.settings.count = Number(e.target.value); });
    $('#instruction').addEventListener('input', (e) => { state.settings.instruction = e.target.value; });
    $('#layoutSelect').addEventListener('change', (e) => { state.settings.layout = e.target.value; });
    $('#fontSelect').addEventListener('change', (e) => { state.settings.style.font = e.target.value; paintStyleSummary(); });
    $('#sizeSelect').addEventListener('change', (e) => { state.settings.style.size = e.target.value; paintStyleSummary(); });

    $$('#kindMode button').forEach((b) => b.addEventListener('click', () => {
      const kind = b.dataset.kind;
      state.settings.clip_kind = kind;
      $$('#kindMode button').forEach((x) => x.classList.toggle('active', x === b));
      $('#kindHint').textContent = kind === 'hook'
        ? 'Hook mode — the same clips plus 7 scroll-stopping hooks each (research-backed, A/B pick and a variant).'
        : 'Standard cut — scored moments with titles, a hook, caption and hashtags.';
      // Hook clips auto-select the Hook caption preset (and lower-centre
      // placement); both stay user-overridable through the chips below.
      applyPreset(kind === 'hook' ? 'hook' : 'classic');
    }));

    $('#createBtn').addEventListener('click', startJob);
    $('#cancelJob').addEventListener('click', cancelJob);
  }

  function showInputError(msg) {
    const box = $('#inputErr');
    box.textContent = msg || '';
    box.classList.toggle('hidden', !msg);
  }

  function switchToUpload() {
    state.sourceMode = 'upload';
    $$('#srcMode button').forEach((x) => x.classList.toggle('active', x.dataset.mode === 'upload'));
    $$('.pane').forEach((p) => p.classList.toggle('hidden', p.dataset.pane !== 'upload'));
  }

  /* An error the visitor can act on: the message plus the one control that
     gets them unstuck, instead of a red box with no way forward. */
  function showInputAction(msg, label, onClick) {
    const box = $('#inputErr');
    box.textContent = '';
    box.classList.remove('hidden');
    box.append(document.createTextNode(msg + ' '));
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = label;
    btn.style.cssText = 'background:none;border:0;padding:0;margin:0;font:inherit;' +
      'color:inherit;text-decoration:underline;cursor:pointer';
    btn.addEventListener('click', onClick);
    box.append(btn);
  }

  async function inspectUrl(url) {
    if (!url) return;
    showInputError('');
    try {
      const info = await api('/api/ai-clipping/inspect', { method: 'POST', body: { url } });
      state.source = { ...info, source_type: 'url' };
      renderPreview();
      estimate();
    } catch (e) {
      state.source = null;
      renderPreview();
      const message = e.message || '';
      // YouTube refuses cloud IPs outright; a retry will never succeed, so
      // point at the path that does work instead of leaving a dead end.
      if (/bot check|blocking downloads/i.test(message)) {
        showInputAction('YouTube blocks downloads from this server.', 'Upload the file instead', () => {
          switchToUpload();
          const picker = $('#fileInput');
          if (picker) picker.click();
        });
      } else {
        showInputError(message);
      }
    }
  }

  function renderPreview() {
    const box = $('#previewCard');
    const src = state.source;
    if (!src) { box.classList.remove('show'); box.innerHTML = ''; return; }
    const thumb = src.thumbnail || '';
    box.classList.add('show');
    box.innerHTML = `
      <div class="preview-thumb">
        ${thumb ? `<img src="${esc(thumb)}" alt="" loading="lazy">` : ''}
        <span class="plat-badge">${esc(src.platform_label || 'Video')}</span>
        <div class="play"><span>▶</span></div>
        <span class="dur-badge">${esc(src.duration_label || fmt.time(src.duration))}</span>
      </div>
      <div class="preview-meta">
        <strong>${esc(src.title || 'Ready to clip')}</strong>
        <small>${src.source_type === 'upload'
          ? `${src.width || 0}×${src.height || 0}${src.audio_only ? ' · audio only' : ''} · ${fmt.bytes(src.size || 0)}`
          : `${esc(src.channel || 'Unknown channel')} · ${esc(src.platform_label || '')}`}</small>
      </div>`;
    const play = $('.preview-thumb .play', box);
    play.addEventListener('click', () => {
      if (src.source_type === 'upload' && src.thumbnail) {
        toast('Preview', 'Uploaded file is ready — create the clips to watch it in 9:16.');
      } else {
        window.open(src.url, '_blank', 'noopener');
      }
    });
  }

  async function startUpload(file) {
    showInputError('');
    const opts = state.options;
    const ext = '.' + (file.name.split('.').pop() || '').toLowerCase();
    if (opts && !opts.upload.accept.includes(ext)) {
      showInputError(`${ext} is not supported. Use ${opts.upload.accept.join(', ')}.`);
      return;
    }
    if (opts && file.size > opts.upload.max_mb * 1024 * 1024) {
      showInputError(`That file is ${fmt.bytes(file.size)} — the limit is ${opts.upload.max_mb} MB.`);
      return;
    }

    const prog = $('#uploadProg');
    prog.classList.remove('hidden');
    $('#dropLabel').textContent = file.name;
    $('#upStage').textContent = 'Uploading…';
    $('#upDetail').textContent = `${fmt.bytes(file.size)} · chunked & resumable`;

    try {
      let uploadId, offset = 0, chunkSize = (opts && opts.upload.chunk_size) || 4 * 1024 * 1024;

      // resume an interrupted upload of the same file
      if (state.uploading && state.uploading.file === file.name) {
        uploadId = state.uploading.upload_id;
        try {
          const st = await api(`/api/ai-clipping/upload/${uploadId}`);
          offset = st.received; chunkSize = st.chunk_size;
        } catch (e) { uploadId = null; offset = 0; }
      }
      if (!uploadId) {
        const init = await api('/api/ai-clipping/upload/init', {
          method: 'POST', body: { filename: file.name, size: file.size },
        });
        uploadId = init.upload_id; chunkSize = init.chunk_size;
      }
      state.uploading = { upload_id: uploadId, file: file.name, offset };

      while (offset < file.size) {
        const blob = file.slice(offset, Math.min(offset + chunkSize, file.size));
        let attempt = 0, data = null;
        while (attempt < 3) {
          try {
            const res = await fetch(`${API}/api/ai-clipping/upload/${uploadId}/chunk?offset=${offset}`,
              { method: 'PUT', body: blob, headers: { 'Content-Type': 'application/octet-stream' } });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            data = await res.json();
            break;
          } catch (err) {
            attempt += 1;
            if (attempt >= 3) {
              throw new Error('Upload interrupted — click the file again to resume from where it stopped.');
            }
            await new Promise((r) => setTimeout(r, 600 * attempt));
            try {
              const st = await api(`/api/ai-clipping/upload/${uploadId}`);
              offset = st.received;
            } catch (e) { /* keep offset */ }
          }
        }
        offset = data ? data.received : offset;
        const pct = Math.min(99, Math.round((offset / file.size) * 100));
        $('#upBar').style.width = pct + '%';
        $('#upPct').textContent = pct + '%';
        $('#upDetail').textContent = `${fmt.bytes(offset)} of ${fmt.bytes(file.size)} · resumable`;
      }

      $('#upStage').textContent = 'Finishing…';
      const done = await api(`/api/ai-clipping/upload/${uploadId}/complete`, { method: 'POST' });
      state.uploading = null;
      state.source = done;
      $('#upBar').style.width = '100%';
      $('#upPct').textContent = '100%';
      $('#upDetail').textContent = 'Upload complete';
      renderPreview();
      estimate();
      toast('Upload ready', `${done.title} · ${fmt.time(done.duration)}`, 'ok');
    } catch (e) {
      showInputError(e.message);
      $('#upStage').textContent = 'Upload failed';
    }
  }

  async function estimate() {
    let duration = state.source ? Number(state.source.duration || 0) : 0;
    try {
      const data = await api('/api/ai-clipping/estimate', {
        method: 'POST', body: { duration },
      });
      const unlimited = !!data.unlimited;
      const enough = unlimited || data.balance >= data.credits;
      $('#costLine').innerHTML = `Estimated cost: <b style="color:var(--green)">${data.credits} credit${data.credits === 1 ? '' : 's'}</b>` +
        (duration ? ` · ${data.minutes} min` : '') +
        (unlimited ? ' <span class="muted">· unlimited, nothing will be deducted</span>' : '');
      state.credits.balance = data.balance;
      state.credits.unlimited = unlimited;
      paintCredits();
      const btn = $('#createBtn');
      btn.innerHTML = `Create Viral Clips <span class="cost">${data.credits} ⚡</span>`;
      btn.dataset.affordable = String(enough);
    } catch (e) {
      $('#costLine').textContent = 'Estimated cost: —';
    }
  }

  async function startJob() {
    showInputError('');
    const src = state.source;
    if (!src) {
      showInputError(state.sourceMode === 'url'
        ? 'Paste a video link and wait for the preview card.'
        : 'Choose a file to upload first.');
      return;
    }
    const btn = $('#createBtn');
    if (btn.dataset.affordable === 'false') return upgradeModal();

    btn.disabled = true;
    const original = btn.innerHTML;
    btn.innerHTML = '<span class="spinner"></span> Starting…';

    try {
      const payload = {
        source: src.source_type === 'upload' ? src.source : src.url,
        source_type: src.source_type,
        meta: {
          title: src.title, duration: src.duration, thumbnail: src.thumbnail,
          channel: src.channel, platform: src.platform,
        },
        settings: state.settings,
      };
      // Caption settings, exactly as they are about to be sent. If a chip
      // looks selected but the render disagrees, this line shows which value
      // actually left the browser.
      console.log('[clip-maker] create job payload', JSON.parse(JSON.stringify(payload)));
      const job = await api('/api/ai-clipping/jobs', { method: 'POST', body: payload });
      state.job = job;
      showJobPanel(job);
      trackJob(job.task_id);
    } catch (e) {
      if (e.status === 402) {
        upgradeModal(e.detail);
      } else {
        showInputError(e.message);
      }
    } finally {
      btn.disabled = false;
      btn.innerHTML = original;
    }
  }

  function upgradeModal(detail) {
    if (state.credits.unlimited) return;   // billing off — never block the user
    const need = detail && detail.required ? detail.required : (state.options ? 5 : 5);
    const have = detail && detail.available !== undefined ? detail.available : state.credits.balance;
    openModal({
      title: 'You need more credits',
      body: `<p style="font-size:13px;line-height:1.6;color:var(--mut)">
          This job costs <b style="color:var(--green)">${need} credits</b> and your balance is
          <b>${have}</b>. Upgrade your plan or claim reward credits to continue.</p>
        <div class="kv" style="margin-top:14px">
          <dt>Free</dt><dd>600 credits / month</dd>
          <dt>Creator</dt><dd>3,000 credits · 4K export · no watermark</dd>
          <dt>Pro</dt><dd>12,000 credits · priority queue</dd>
        </div>`,
      footer: `<a class="btn btn-ghost" href="editor.html#rewards">Claim rewards</a>
               <a class="btn btn-primary" href="editor.html#pricing">Upgrade plan</a>`,
    });
  }

  function showJobPanel(job) {
    $('#emptyPanel').classList.add('hidden');
    $('#jobPanel').classList.remove('hidden');
    $('#jobTitle').textContent = 'Processing your video';
    $('#jobStage').textContent = 'Queued';
    $('#jobPct').textContent = '0%';
    $('#jobBar').style.width = '0%';
    $('#jobSteps').innerHTML = (job.steps || []).map((s) =>
      `<li data-step="${esc(s)}"><span class="b"></span><span>${esc(s)}</span></li>`).join('');
    $('#createBtn').disabled = true;
  }

  async function trackJob(taskId) {
    let stop = false;
    const finish = () => { stop = true; $('#createBtn').disabled = false; };
    const tick = async () => {
      if (stop) return;
      let data;
      try { data = await api(`/api/ai-clipping/jobs/${taskId}`); }
      catch (e) {
        await new Promise((r) => setTimeout(r, 2000));
        return tick();
      }
      $('#jobBar').style.width = (data.progress || 0) + '%';
      $('#jobPct').textContent = Math.round(data.progress || 0) + '%';
      $('#jobStage').textContent = data.stage || data.status;
      (data.steps || []).forEach((step, i) => {
        const li = $(`#jobSteps li:nth-child(${i + 1})`);
        if (!li) return;
        li.classList.toggle('done', step.state === 'done');
        li.classList.toggle('running', step.state === 'running');
        li.querySelector('.b').textContent = step.state === 'done' ? '✓' : step.state === 'running' ? '•' : '';
      });

      if (data.status === 'done') {
        finish();
        const videoId = (data.result || {}).video_id;
        toast('Clips ready', `${(data.result || {}).moments} moments found`, 'ok');
        await loadCredits();
        if (videoId) { location.hash = 'results/' + videoId; }
        return;
      }
      if (data.status === 'error' || data.status === 'cancelled') {
        finish();
        showInputError(data.error || 'The job failed.');
        $('#jobStage').textContent = data.status === 'cancelled' ? 'Cancelled' : 'Failed';
        toast('Clipping failed', data.error || 'Unknown error', 'err');
        await loadCredits();   // credits are refunded server-side
        return;
      }
      setTimeout(tick, 1200);
    };
    tick();
  }

  async function cancelJob() {
    if (!state.job) return;
    try {
      await api(`/api/ai-clipping/jobs/${state.job.task_id}/cancel`, { method: 'POST' });
      toast('Cancelling…', 'The job stops after the current step.');
    } catch (e) { toast('Could not cancel', e.message, 'err'); }
  }

  /* ------------------------------------------------------------ results */
  function bindResults() {
    $('#resBack').addEventListener('click', () => { location.hash = 'input'; });
    $('#sortSelect').addEventListener('change', (e) => { state.sort = e.target.value; state.page = 1; renderGrid(); });
    $('#filterBtn').addEventListener('click', () => $('#filters').classList.toggle('open'));
    $('#fMinScore').addEventListener('input', (e) => {
      state.filters.minScore = Number(e.target.value);
      $('#fMinScoreVal').textContent = e.target.value;
      state.page = 1; renderGrid();
    });
    $('#fDurMin').addEventListener('input', (e) => {
      state.filters.durMin = Number(e.target.value) || 0;
      state.page = 1; renderGrid();
    });
    $('#fDurMax').addEventListener('input', (e) => {
      state.filters.durMax = Number(e.target.value) || 0;
      state.page = 1; renderGrid();
    });
    $('#fSpeaker').addEventListener('change', (e) => { state.filters.speaker = e.target.value; state.page = 1; renderGrid(); });
    $('#fKeyword').addEventListener('input', (e) => {
      state.filters.keyword = e.target.value.toLowerCase();
      state.page = 1; renderGrid();
    });
    $('#selectAll').addEventListener('change', (e) => {
      const visible = filtered();
      if (e.target.checked) visible.forEach((m) => state.selected.add(m.id));
      else visible.forEach((m) => state.selected.delete(m.id));
      renderGrid();
    });
    $('#loadMore').addEventListener('click', () => { state.page += 1; renderGrid(); });

    $('#bulkEdit').addEventListener('click', bulkEditModal);
    $('#bulkSchedule').addEventListener('click', () => bulkScheduleModal());
    $('#bulkExport').addEventListener('click', bulkExportModal);
    $('#bulkMontage').addEventListener('click', montageModal);

    $('#tabViral').addEventListener('click', () => {
      $('#tabViral').classList.add('active'); $('#tabFind').classList.remove('active');
      $('#findPanel').classList.remove('open');
    });
    $('#tabFind').addEventListener('click', () => {
      $('#tabFind').classList.add('active'); $('#tabViral').classList.remove('active');
      $('#findPanel').classList.add('open');
      $('#findQuery').focus();
    });
    $('#findBtn').addEventListener('click', runFind);
    $('#findQuery').addEventListener('keydown', (e) => { if (e.key === 'Enter') runFind(); });
    $$('#findPanel .find-chips button').forEach((b) => b.addEventListener('click', () => {
      $('#findQuery').value = b.dataset.q;
      runFind();
    }));

    document.addEventListener('click', (e) => {
      if (!e.target.closest('.more-menu') && !e.target.closest('[data-act="more"]')) {
        $$('.more-menu.open').forEach((m) => m.classList.remove('open'));
      }
    });
    document.addEventListener('keydown', (e) => {
      if (e.target.matches('input,textarea,select')) return;
      if (e.key === 'Escape') $$('.more-menu.open').forEach((m) => m.classList.remove('open'));
    });
  }

  async function openVideo(id) {
    $('#view-results').classList.add('active');
    $('#clipGrid').innerHTML = '<div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div>';
    try {
      const doc = await api(`/api/ai-clipping/videos/${id}`);
      state.video = doc;
      state.moments = doc.moments || [];
      state.selected.clear();
      state.page = 1;
      state.sort = 'score';
      $('#sortSelect').value = 'score';
      $('#resTitle').textContent =
        `${doc.title} · ${fmt.time(doc.duration)} · ${doc.moments.length} moments` +
        (doc.selector === 'llm' ? ' · AI selection' : ' · local scorer');
      const speakers = [...new Set(state.moments.map((m) => m.speaker).filter(Boolean))].sort();
      $('#fSpeaker').innerHTML = '<option value="">All</option>' +
        speakers.map((s) => `<option>${esc(s)}</option>`).join('');
      renderGrid();
    } catch (e) {
      $('#clipGrid').innerHTML = `<div class="err">${esc(e.message)}</div>`;
    }
  }

  function filtered() {
    const f = state.filters;
    let list = state.moments.filter((m) => {
      if (m.score < f.minScore) return false;
      if (m.duration < f.durMin) return false;
      if (f.durMax && m.duration > f.durMax) return false;
      if (f.speaker && m.speaker !== f.speaker) return false;
      if (f.keyword) {
        const hay = (m.title + ' ' + (m.transcript || '') + ' ' + (m.hook_text || '')).toLowerCase();
        if (!hay.includes(f.keyword)) return false;
      }
      return true;
    });
    if (state.sort === 'score') list.sort((a, b) => b.score - a.score);
    if (state.sort === 'chrono') list.sort((a, b) => a.start - b.start);
    if (state.sort === 'shortest') list.sort((a, b) => a.duration - b.duration);
    if (state.sort === 'longest') list.sort((a, b) => b.duration - a.duration);
    return list;
  }

  function renderGrid() {
    const grid = $('#clipGrid');
    const list = filtered();
    const visible = list.slice(0, state.page * PAGE);
    grid.innerHTML = visible.map(cardHtml).join('');
    if (!visible.length) {
      grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1"><span class="big">🔎</span>
        <p>No clips match these filters. Widen the score or length range.</p></div>`;
    }
    $$('.clip-card', grid).forEach((card) => bindCard(card));

    $('#viralCount').textContent = state.moments.length;
    $('#selectAllLabel').textContent = `Select all ${list.length} clip${list.length === 1 ? '' : 's'}`;
    $('#selectAll').checked = list.length > 0 && list.every((m) => state.selected.has(m.id));
    $('#shownCount').textContent = `Showing ${visible.length} of ${list.length}`;
    $('#loadMore').classList.toggle('hidden', visible.length >= list.length);
    updateBulkBar();
  }

  function cardHtml(m) {
    const breakdown = m.score_breakdown || {};
    const tip = Object.entries(breakdown)
      .map(([k, v]) => `${k}: ${v}`).join(' · ') + '\n\nClick to see what to improve.';
    const tags = (m.hashtags || []).slice(0, 4).map((h) => `<span class="tag">${esc(h)}</span>`).join('');
    return `
    <article class="clip-card ${state.selected.has(m.id) ? 'selected' : ''}" data-mid="${esc(m.id)}">
      <div class="clip-media">
        <video src="${esc(m.preview_url || '')}#t=0.1" poster="${esc(m.poster_url || '')}"
               preload="metadata" playsinline></video>
        <div class="clip-top">
          <span class="rank">#<b>${m.rank}</b></span>
          <button class="check ${state.selected.has(m.id) ? 'on' : ''}" data-act="select"
                  aria-label="Select clip">${state.selected.has(m.id) ? '✓' : ''}</button>
        </div>
        <div class="play"><span>▶</span></div>
        <span class="range-badge">${esc(m.start_label)} – ${esc(m.end_label)}</span>
        <span class="score-badge" title="${esc(tip)}">${m.score}/100</span>
      </div>
      <div class="clip-body">
        <div class="clip-title">
          <h4 data-title>${esc(m.title || 'Untitled clip')}</h4>
          <button class="icon-mini" data-act="regen" title="Regenerate this moment">⟳</button>
        </div>
        <p class="why">${esc(m.why_it_works || '')}</p>
        <div class="tagline">${tags}</div>
        <div class="actions">
          <button class="act" data-act="publish" title="Publish / Schedule">➤</button>
          <button class="act" data-act="edit" title="Edit in the AI video editor">✂</button>
          <button class="act" data-act="reframe" title="Reframe">⛶</button>
          <button class="act" data-act="download" title="Download MP4">⬇</button>
          <button class="act" data-act="share" title="Copy share link">⧉</button>
          <button class="act" data-act="more" title="More">⋯</button>
          <div class="more-menu">
            <button data-more="rename">Rename</button>
            <button data-more="trim">Adjust start / end</button>
            <button data-more="duplicate">Duplicate</button>
            <button data-more="hooks">Alternative hooks</button>
            <button data-more="copy-caption">Copy caption</button>
            <button data-more="copy-tags">Copy hashtags</button>
            <button data-more="transcript">View transcript</button>
            <button data-more="delete" class="danger">Delete clip</button>
          </div>
        </div>
      </div>
    </article>`;
  }

  function bindCard(card) {
    const id = card.dataset.mid;
    const moment = state.moments.find((m) => m.id === id);
    if (!moment) return;
    const video = $('video', card);
    const play = $('.play', card);

    const startPlay = () => {
      play.style.display = 'none';
      video.setAttribute('controls', 'controls');
      video.play();
    };
    play.addEventListener('click', startPlay);
    // clicking the poster (but not a badge/button) also starts playback
    $('.clip-media', card).addEventListener('click', (e) => {
      if (e.target.closest('button, .score-badge, .check')) return;
      if (play.style.display === 'none') return;
      startPlay();
    });
    video.addEventListener('pause', () => { video.removeAttribute('controls'); play.style.display = ''; });
    video.addEventListener('ended', () => { video.removeAttribute('controls'); play.style.display = ''; });

    $('[data-act="select"]', card).addEventListener('click', (e) => {
      e.stopPropagation();
      if (state.selected.has(id)) state.selected.delete(id); else state.selected.add(id);
      renderGrid();
    });
    $('.score-badge', card).addEventListener('click', () => scoreModal(moment));
    $('.clip-title h4', card).addEventListener('click', () => renameModal(moment));

    card.addEventListener('click', async (e) => {
      const moreBtn = e.target.closest('[data-more]');
      if (moreBtn) { e.stopPropagation(); $('.more-menu', card).classList.remove('open'); return moreAction(moreBtn.dataset.more, moment); }
      const act = e.target.closest('[data-act]');
      if (!act) return;
      e.stopPropagation();
      const kind = act.dataset.act;
      if (kind === 'more') {
        const menu = $('.more-menu', card);
        const wasOpen = menu.classList.contains('open');
        $$('.more-menu.open').forEach((m) => m.classList.remove('open'));
        if (!wasOpen) menu.classList.add('open');
        return;
      }
      if (kind === 'select') return;
      if (kind === 'publish') return publishModal([moment]);
      if (kind === 'edit') return openInEditor(moment);
      if (kind === 'reframe') return reframeModal(moment);
      if (kind === 'download') return downloadClip(moment);
      if (kind === 'share') return shareClip(moment);
      if (kind === 'regen') return regenerate(moment);
    });
  }

  /* ------------------------------------------------------- moment actions */
  function scoreModal(m) {
    const b = m.score_breakdown || {};
    const max = { hook: 25, pacing: 20, emotion: 20, clarity: 15, shareability: 10, trend: 10 };
    const rows = Object.keys(max).map((k) => {
      const pct = Math.round(((b[k] || 0) / max[k]) * 100);
      return `<div style="margin-bottom:9px">
          <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px">
            <span style="text-transform:capitalize">${k}</span>
            <span class="muted">${(b[k] || 0).toFixed(1)} / ${max[k]}</span>
          </div>
          <div class="bar" style="height:5px"><i style="width:${pct}%"></i></div>
        </div>`;
    }).join('');
    const weak = Object.keys(max).sort((a, c) => (b[a] / max[a]) - (b[c] / max[c]))[0];
    const tips = {
      hook: 'Start on the strongest line — trim the first seconds so the opening stings.',
      pacing: 'Cut the dead air: remove long pauses or shorten the clip to its densest part.',
      emotion: 'Pick the moment right after the peak reaction so the payoff lands inside the clip.',
      clarity: 'Start a sentence later so the clip stands alone without prior context.',
      shareability: 'Add an on-screen hook that names the audience ("if you are broke right now…").',
      trend: 'Retitle around the current topic angle to match what the platform is pushing.',
    };
    openModal({
      title: `Viral score — ${m.score}/100`,
      body: `${rows}
        <p class="hint" style="margin-top:14px">Biggest opportunity: <b style="color:var(--green)">${weak}</b>.
          ${esc(tips[weak] || '')}</p>
        <div class="kv" style="margin-top:14px">
          <dt>Clip</dt><dd>${esc(m.start_label)} → ${esc(m.end_label)} (${fmt.dur(m.duration)})</dd>
          <dt>Speaker</dt><dd>${esc(m.speaker || '—')}</dd>
          <dt>Risk flags</dt><dd>${(m.risk_flags || []).join(', ') || 'none'}</dd>
          <dt>Best platforms</dt><dd>${(m.best_platforms || []).join(', ') || '—'}</dd>
        </div>`,
      footer: `<button class="btn btn-ghost" data-close>Close</button>`,
    });
  }

  async function renameModal(m) {
    openModal({
      title: 'Rename clip',
      body: `<label class="field-label">Title (max 70 characters)</label>
        <input class="input" id="renameInput" maxlength="70" value="${esc(m.title || '')}">`,
      footer: `<button class="btn btn-ghost" data-close>Cancel</button>
               <button class="btn btn-primary" id="renameSave">Save</button>`,
      onMount(root, close) {
        const input = $('#renameInput', root);
        input.focus(); input.select();
        $('#renameSave', root).addEventListener('click', async () => {
          try {
            const res = await api(`/api/ai-clipping/moments/${m.id}?video_id=${state.video.id}`,
              { method: 'PATCH', body: { title: input.value } });
            Object.assign(m, res.moment);
            close(); renderGrid(); toast('Renamed', '', 'ok');
          } catch (e) { toast('Rename failed', e.message, 'err'); }
        });
      },
    });
  }

  function trimModal(m) {
    openModal({
      title: 'Adjust clip boundaries',
      body: `<div class="trim-row">
          <div><label class="field-label">Start (seconds)</label>
            <input class="input" id="trimStart" type="number" step="0.1" min="0" max="${state.video.duration}" value="${m.start}"></div>
          <div><label class="field-label">End (seconds)</label>
            <input class="input" id="trimEnd" type="number" step="0.1" min="0" max="${state.video.duration}" value="${m.end}"></div>
        </div>
        <p class="hint">Boundaries snap to whole words and re-render the preview. Source length: ${fmt.time(state.video.duration)}.</p>
        <video src="${esc(m.preview_url || '')}" controls style="width:100%;margin-top:12px;border-radius:10px;background:#04090b"></video>`,
      footer: `<button class="btn btn-ghost" data-close>Cancel</button>
               <button class="btn btn-primary" id="trimSave">Apply &amp; re-render</button>`,
      onMount(root, close) {
        $('#trimSave', root).addEventListener('click', async (e) => {
          const btn = e.currentTarget;
          btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Rendering…';
          try {
            const res = await api(`/api/ai-clipping/moments/${m.id}?video_id=${state.video.id}`, {
              method: 'PATCH',
              body: { start: Number($('#trimStart', root).value), end: Number($('#trimEnd', root).value) },
            });
            Object.assign(m, res.moment);
            close(); renderGrid(); toast('Clip updated', 'Preview re-rendered', 'ok');
          } catch (err) {
            btn.disabled = false; btn.textContent = 'Apply & re-render';
            toast('Could not update', err.message, 'err');
          }
        });
      },
    });
  }

  async function regenerate(m) {
    toast('Regenerating…', 'Looking for a stronger cut around this moment');
    try {
      const res = await api(`/api/ai-clipping/moments/${m.id}/regenerate?video_id=${state.video.id}`,
        { method: 'POST' });
      Object.assign(m, res.moment);
      renderGrid(); await loadCredits();
      toast('New cut', `Score ${res.moment.score}/100`, 'ok');
    } catch (e) {
      toast('Keeping this cut', e.message, 'err');
    }
  }

  function reframeModal(m) {
    const layouts = (state.options && state.options.layouts) || ['Full', 'Fit', 'Split', 'Trio', 'Grid', 'PiP'];
    openModal({
      title: 'Reframe',
      body: `<p class="hint" style="margin-bottom:12px">9:16 output. Pick the layout for this clip — the preview re-renders immediately.</p>
        <div class="opt-grid" id="layoutOpts">
          ${layouts.map((l) => `<button class="opt ${l === m.layout ? 'active' : ''}" data-layout="${esc(l)}">${esc(l)}</button>`).join('')}
        </div>`,
      onMount(root, close) {
        $$('[data-layout]', root).forEach((b) => b.addEventListener('click', async () => {
          $$('[data-layout]', root).forEach((x) => x.classList.toggle('active', x === b));
          try {
            const res = await api(`/api/ai-clipping/moments/${m.id}?video_id=${state.video.id}`,
              { method: 'PATCH', body: { layout: b.dataset.layout } });
            Object.assign(m, res.moment);
            renderGrid();
            toast('Layout applied', b.dataset.layout, 'ok');
            close();
          } catch (e) { toast('Reframe failed', e.message, 'err'); }
        }));
      },
    });
  }

  function openInEditor(m) {
    if (!m.preview_url) return toast('Not ready', 'Render this clip first.', 'err');
    const src = encodeURIComponent(absUrl(m.preview_url));
    window.open(`editor.html?src=${src}&name=${encodeURIComponent(m.title || 'Clip')}`, '_blank', 'noopener');
  }

  async function downloadClip(m) {
    toast('Rendering 1080p', 'Your download starts when it finishes');
    try {
      const res = await api('/api/ai-clipping/export', {
        method: 'POST',
        body: {
          video_id: state.video.id, moment_ids: [m.id], resolution: '1080p',
          options: { captions: true, hook: false, watermark: false }, zip: false,
        },
      });
      await pollJob(res.task_id, 'Rendering clip');
      const record = await exportRecordFor(res.task_id);
      const file = (record.files || [])[0];
      if (file) {
        const a = document.createElement('a');
        a.href = absUrl(file.url);
        a.download = file.file || 'clip.mp4';
        document.body.appendChild(a); a.click(); a.remove();
        toast('Download ready', file.file, 'ok');
      }
    } catch (e) { toast('Export failed', e.message, 'err'); }
  }

  async function shareClip(m) {
    try {
      const res = await api('/api/ai-clipping/share', {
        method: 'POST', body: { video_id: state.video.id, moment_id: m.id },
      });
      copy(absUrl(res.url), 'Share link copied');
    } catch (e) { toast('Could not share', e.message, 'err'); }
  }

  function transcriptModal(m) {
    openModal({
      title: 'Clip transcript',
      wide: true,
      body: `<div class="transcript-box">${esc(m.transcript || 'No transcript for this clip.')}</div>
        <p class="hint">${esc(m.start_label)} → ${esc(m.end_label)} · ${fmt.dur(m.duration)}</p>`,
      footer: `<button class="btn btn-ghost" id="copyTranscript">Copy transcript</button>
               <button class="btn btn-ghost" data-close>Close</button>`,
      onMount(root) {
        $('#copyTranscript', root).addEventListener('click', () => copy(m.transcript || '', 'Transcript copied'));
      },
    });
  }

  async function hooksModal(m) {
    openModal({
      title: 'Alternative hooks',
      body: '<p class="hint">Five A/B variants. Click one to use it on this clip.</p><div id="hookList"><span class="spinner"></span></div>',
      onMount(root, close) {
        api(`/api/ai-clipping/moments/${m.id}/hooks?video_id=${state.video.id}`, { method: 'POST' })
          .then((data) => {
            $('#hookList', root).innerHTML = (data.hooks || []).map((h) => `
              <button class="export-row" data-hook="${esc(h.text)}" style="width:100%;text-align:left">
                <span class="grow"><b>${esc(h.text)}</b><small>${esc(h.pattern)} — ${esc(h.why || '')}</small></span>
              </button>`).join('');
            $$('[data-hook]', root).forEach((b) => b.addEventListener('click', async () => {
              try {
                const res = await api(`/api/ai-clipping/moments/${m.id}?video_id=${state.video.id}`,
                  { method: 'PATCH', body: { hook_text: b.dataset.hook } });
                Object.assign(m, res.moment);
                close(); renderGrid(); toast('Hook applied', b.dataset.hook, 'ok');
              } catch (e) { toast('Could not apply', e.message, 'err'); }
            }));
          })
          .catch((e) => { $('#hookList', root).innerHTML = `<div class="err">${esc(e.message)}</div>`; });
      },
    });
  }

  async function moreAction(kind, m) {
    if (kind === 'rename') return renameModal(m);
    if (kind === 'trim') return trimModal(m);
    if (kind === 'duplicate') {
      try {
        const res = await api(`/api/ai-clipping/moments/${m.id}/duplicate?video_id=${state.video.id}`, { method: 'POST' });
        state.moments.push(res.moment); renderGrid(); toast('Duplicated', '', 'ok');
      } catch (e) { toast('Duplicate failed', e.message, 'err'); }
      return;
    }
    if (kind === 'hooks') return hooksModal(m);
    if (kind === 'copy-caption') return copy(m.suggested_caption || m.title || '', 'Caption copied');
    if (kind === 'copy-tags') return copy((m.hashtags || []).join(' '), 'Hashtags copied');
    if (kind === 'transcript') return transcriptModal(m);
    if (kind === 'delete') {
      openModal({
        title: 'Delete this clip?',
        body: `<p style="font-size:13px;color:var(--mut)">"${esc(m.title)}" and its preview are removed from this project.</p>`,
        footer: `<button class="btn btn-ghost" data-close>Cancel</button>
                 <button class="btn btn-danger" id="delYes">Delete clip</button>`,
        onMount(root, close) {
          $('#delYes', root).addEventListener('click', async () => {
            try {
              await api(`/api/ai-clipping/moments/${m.id}?video_id=${state.video.id}`, { method: 'DELETE' });
              state.moments = state.moments.filter((x) => x.id !== m.id);
              state.selected.delete(m.id);
              close(); renderGrid(); toast('Deleted', '', 'ok');
            } catch (e) { toast('Delete failed', e.message, 'err'); }
          });
        },
      });
    }
  }

  /* ---------------------------------------------------------- selection */
  function selectedMoments() {
    return state.moments.filter((m) => state.selected.has(m.id));
  }

  function updateBulkBar() {
    const n = state.selected.size;
    $('#selCount').innerHTML = n
      ? `<b>${n}</b> clip${n === 1 ? '' : 's'} selected`
      : 'Select clips to enable bulk actions';
    ['bulkEdit', 'bulkSchedule', 'bulkExport', 'bulkMontage'].forEach((id) => {
      $('#' + id).disabled = n === 0;
    });
    $('#bulkMontage').disabled = n < 2;
  }

  /* ------------------------------------------------------ bulk actions */
  function bulkEditModal() {
    const list = selectedMoments();
    const layouts = (state.options && state.options.layouts) || ['Full', 'Fit', 'Split', 'Trio', 'Grid', 'PiP'];
    openModal({
      title: `Bulk edit ${list.length} clips`,
      body: `<label class="field-label">Reframe layout</label>
        <div class="opt-grid" id="bLayouts">
          <button class="opt active" data-blayout="">Keep current</button>
          ${layouts.map((l) => `<button class="opt" data-blayout="${esc(l)}">${esc(l)}</button>`).join('')}
        </div>
        <label class="field-label" style="margin-top:14px">Hook text (optional)</label>
        <input class="input" id="bHook" maxlength="120" placeholder="Leave empty to keep each clip's own hook">
        <p class="hint">Layout changes re-render each preview. Brand kit items live in the AI editor's Brand Kit panel.</p>`,
      footer: `<button class="btn btn-ghost" data-close>Cancel</button>
               <button class="btn btn-primary" id="bApply">Apply to ${list.length}</button>`,
      onMount(root, close) {
        let layout = '';
        $$('[data-blayout]', root).forEach((b) => b.addEventListener('click', () => {
          layout = b.dataset.blayout;
          $$('[data-blayout]', root).forEach((x) => x.classList.toggle('active', x === b));
        }));
        $('#bApply', root).addEventListener('click', async (e) => {
          const btn = e.currentTarget;
          btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Applying…';
          const hook = $('#bHook', root).value.trim();
          let done = 0;
          for (const m of list) {
            const body = {};
            if (layout) body.layout = layout;
            if (hook) body.hook_text = hook;
            if (!Object.keys(body).length) continue;
            try {
              const res = await api(`/api/ai-clipping/moments/${m.id}?video_id=${state.video.id}`,
                { method: 'PATCH', body });
              Object.assign(m, res.moment);
              done += 1;
            } catch (err) { toast('One clip failed', err.message, 'err'); }
          }
          close();
          toast('Bulk edit applied', `${done} clip(s) updated`, 'ok');
          renderGrid();
        });
      },
    });
  }

  function bulkExportModal() {
    const list = selectedMoments();
    openModal({
      title: `Export ${list.length} clip${list.length === 1 ? '' : 's'}`,
      body: `<label class="field-label">Resolution</label>
        <div class="opt-grid" id="expRes">
          <button class="opt" data-res="720p">720p</button>
          <button class="opt active" data-res="1080p">1080p</button>
          <button class="opt" data-res="4K">4K</button>
        </div>
        <label class="field-label" style="margin-top:14px">Include</label>
        <div class="chips" id="expOpts">
          <button class="chip active" data-opt="captions">Burned captions</button>
          <button class="chip" data-opt="watermark">Watermark</button>
          <button class="chip" data-opt="cold_open">Cold-open teaser</button>
        </div>
        <p class="hint" style="margin-top:12px">Multiple clips are packaged as a ZIP.</p>`,
      footer: `<button class="btn btn-ghost" data-close>Cancel</button>
               <button class="btn btn-primary" id="expGo">Render &amp; download</button>`,
      onMount(root, close) {
        let resolution = '1080p';
        const opts = { captions: true, hook: false, watermark: false, cold_open: false };
        $$('[data-res]', root).forEach((b) => b.addEventListener('click', () => {
          resolution = b.dataset.res;
          $$('[data-res]', root).forEach((x) => x.classList.toggle('active', x === b));
        }));
        $$('[data-opt]', root).forEach((b) => b.addEventListener('click', () => {
          const key = b.dataset.opt;
          opts[key] = !opts[key];
          b.classList.toggle('active', opts[key]);
        }));
        $('#expGo', root).addEventListener('click', async (e) => {
          const btn = e.currentTarget;
          btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Rendering…';
          try {
            const res = await api('/api/ai-clipping/export', {
              method: 'POST',
              body: {
                video_id: state.video.id, moment_ids: list.map((m) => m.id),
                resolution, options: opts, zip: list.length > 1,
              },
            });
            close();
            await pollJob(res.task_id, 'Rendering export');
            const record = await exportRecordFor(res.task_id);
            exportDoneModal(record);
          } catch (err) { toast('Export failed', err.message, 'err'); }
        });
      },
    });
  }

  async function exportRecordFor(taskId) {
    // the export task result is the record itself; re-fetch the task to read it
    const task = await api(`/api/ai-clipping/jobs/${taskId}`).catch(() => null);
    if (task && task.result) return task.result;
    const data = await api('/api/ai-clipping/exports');
    return (data.items || [])[0] || { files: [] };
  }

  function exportDoneModal(record) {
    const files = record.files || [];
    openModal({
      title: 'Export ready',
      body: `${record.zip_url ? `<a class="btn btn-primary btn-block" href="${absUrl(record.zip_url)}" download>Download ZIP (${files.length} clips)</a>` : ''}
        <div style="margin-top:12px">
          ${files.map((f) => `<div class="export-row">
              <span class="grow"><b>${esc(f.file)}</b><small>${esc(f.resolution)} · ${fmt.bytes(f.size)}</small></span>
              <a class="btn btn-ghost btn-sm" href="${absUrl(f.url)}" download="${esc(f.file)}">Download</a>
            </div>`).join('')}
        </div>`,
      footer: `<button class="btn btn-ghost" id="openLib">Open Library</button>
               <button class="btn btn-primary" data-close>Done</button>`,
      onMount(root, close) {
        $('#openLib', root).addEventListener('click', () => { close(); location.hash = 'library'; });
      },
    });
  }

  function montageModal() {
    const list = selectedMoments().slice().sort((a, b) => a.start - b.start);
    openModal({
      title: 'Create montage',
      body: `<p class="hint">${list.length} clips joined in chronological order with transitions and burned captions.</p>
        <label class="field-label" style="margin-top:12px">Transition</label>
        <div class="opt-grid" id="monTrans">
          ${['fade', 'dissolve', 'wipeleft', 'slideleft', 'circleopen'].map((t) => `
            <button class="opt ${t === 'fade' ? 'active' : ''}" data-trans="${t}">${t}</button>`).join('')}
        </div>
        <label class="field-label" style="margin-top:14px">Quality</label>
        <div class="opt-grid" id="monRes">
          <button class="opt" data-res="720p">720p</button>
          <button class="opt active" data-res="1080p">1080p</button>
          <button class="opt" data-res="4K">4K</button>
        </div>
        <label class="field-label" style="margin-top:14px">Order</label>
        <div class="transcript-box" style="max-height:140px">
          ${list.map((m, i) => `${i + 1}. ${esc(m.title)} (${fmt.dur(m.duration)})`).join('<br>')}
        </div>`,
      footer: `<button class="btn btn-ghost" data-close>Cancel</button>
               <button class="btn btn-primary" id="monGo">Create montage</button>`,
      onMount(root, close) {
        let transition = 'fade', resolution = '1080p';
        $$('[data-trans]', root).forEach((b) => b.addEventListener('click', () => {
          transition = b.dataset.trans;
          $$('[data-trans]', root).forEach((x) => x.classList.toggle('active', x === b));
        }));
        $$('[data-res]', root).forEach((b) => b.addEventListener('click', () => {
          resolution = b.dataset.res;
          $$('[data-res]', root).forEach((x) => x.classList.toggle('active', x === b));
        }));
        $('#monGo', root).addEventListener('click', async (e) => {
          const btn = e.currentTarget;
          btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Joining…';
          try {
            const res = await api('/api/ai-clipping/montage', {
              method: 'POST',
              body: {
                video_id: state.video.id, moment_ids: list.map((m) => m.id),
                resolution, transition, captions: true,
              },
            });
            close();
            await pollJob(res.task_id, 'Creating montage');
            const task = await api(`/api/ai-clipping/jobs/${res.task_id}`);
            const rec = task.result || {};
            openModal({
              title: 'Montage ready',
              body: `<video src="${esc(absUrl(rec.url))}" controls style="width:100%;border-radius:12px;background:#04090b"></video>
                <p class="hint">${esc(rec.title || '')} · ${rec.clips || list.length} clips</p>`,
              footer: `<a class="btn btn-ghost" href="${esc(absUrl(rec.url))}" download>Download</a>
                       <button class="btn btn-primary" data-close>Done</button>`,
            });
          } catch (err) { toast('Montage failed', err.message, 'err'); }
        });
      },
    });
  }

  /* ----------------------------------------------------------- schedule */
  async function accounts() {
    try { return (await api('/api/ai-clipping/accounts')).platforms || []; }
    catch (e) { return []; }
  }

  async function publishModal(list) {
    const plats = await accounts();
    const first = list[0] || {};
    const postBody = (moment) => `
      <div class="export-row" style="align-items:flex-start">
        <img src="${esc(moment.poster_url || '')}" alt="" style="width:56px;height:78px;object-fit:cover;border-radius:7px">
        <span class="grow"><b>${esc(moment.title)}</b>
          <small>${esc(moment.start_label)} – ${esc(moment.end_label)} · score ${moment.score}</small></span>
      </div>`;

    openModal({
      title: list.length === 1 ? 'Publish clip' : `Schedule ${list.length} clips`,
      wide: true,
      body: `<div id="pubList">${list.slice(0, 4).map(postBody).join('')}</div>
        <label class="field-label" style="margin-top:14px">Platforms</label>
        <div class="chips" id="pubPlatforms">
          ${plats.map((p) => `<button class="chip ${p.connected ? '' : ''}" data-plat="${esc(p.platform)}"
              title="${p.configured ? 'Credentials configured' : 'No OAuth credentials on this server'}">
              ${esc(p.label)} ${p.connected ? '✓' : p.configured ? '·' : '⛔'}</button>`).join('')}
        </div>
        <p class="hint" id="platHint">Platforms marked ⛔ have no OAuth credentials configured on this server
          (set WV_&lt;PLATFORM&gt;_CLIENT_ID / _SECRET). You can still queue drafts and see the exact error.</p>

        <label class="field-label" style="margin-top:14px">Title</label>
        <div style="display:flex;gap:8px">
          <input class="input" id="pubTitle" maxlength="90" value="${esc(first.title || '')}">
          <button class="btn btn-ghost btn-sm" id="aiTitle">AI</button>
        </div>
        <label class="field-label" style="margin-top:12px">Description / caption</label>
        <textarea class="input" id="pubDesc">${esc(first.suggested_caption || '')}</textarea>
        <div style="display:flex;gap:8px;margin-top:8px">
          <button class="btn btn-ghost btn-sm" id="aiDesc">AI caption &amp; hashtags</button>
        </div>
        <label class="field-label" style="margin-top:12px">Hashtags</label>
        <input class="input" id="pubTags" value="${esc((first.hashtags || []).join(' '))}">

        <div class="row2" style="margin-top:14px">
          <div><label class="field-label">When</label>
            <input class="input" id="pubWhen" type="datetime-local"></div>
          <div><label class="field-label">Spacing between clips (minutes)</label>
            <input class="input" id="pubSpacing" type="number" min="0" value="30"></div>
        </div>`,
      footer: `<button class="btn btn-ghost" id="pubDraft">Save as scheduled</button>
               <button class="btn btn-primary" id="pubNow">Publish now</button>`,
      onMount(root, close) {
        const chosen = new Set();
        $$('[data-plat]', root).forEach((b) => b.addEventListener('click', () => {
          const key = b.dataset.plat;
          if (chosen.has(key)) chosen.delete(key); else chosen.add(key);
          b.classList.toggle('active', chosen.has(key));
        }));
        const when = $('#pubWhen', root);
        const d = new Date(Date.now() + 60 * 60 * 1000);
        when.value = new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 16);

        const fillFromAI = async () => {
          try {
            const copy = await api(`/api/ai-clipping/moments/${first.id}/platform?video_id=${state.video.id}`,
              { method: 'POST' });
            const titles = copy.titles || {};
            $('#pubTitle', root).value = titles.youtube || titles.reels || first.title;
            $('#pubDesc', root).value = copy.youtube_description || first.suggested_caption || '';
            const tags = ((copy.hashtags || {}).tiktok || (copy.hashtags || {}).instagram || first.hashtags || []);
            $('#pubTags', root).value = tags.join(' ');
            toast('AI copy ready', 'Title, description and hashtags filled', 'ok');
          } catch (e) { toast('AI copy failed', e.message, 'err'); }
        };
        $('#aiTitle', root).addEventListener('click', fillFromAI);
        $('#aiDesc', root).addEventListener('click', fillFromAI);

        const submit = async (publishNow) => {
          if (!chosen.size) return toast('Pick a platform', '', 'err');
          const ts = publishNow ? Math.floor(Date.now() / 1000) : Math.floor(new Date(when.value).getTime() / 1000);
          try {
            const res = await api('/api/ai-clipping/schedule', {
              method: 'POST',
              body: {
                video_id: state.video.id, moment_ids: list.map((m) => m.id),
                platforms: [...chosen], start_at: ts,
                spacing_minutes: Number($('#pubSpacing', root).value) || 0,
                title: $('#pubTitle', root).value,
                description: $('#pubDesc', root).value,
                hashtags: $('#pubTags', root).value.split(/[\s,]+/).filter(Boolean),
              },
            });
            if (!publishNow) {
              close();
              toast('Scheduled', `${res.count} clip(s) queued`, 'ok');
              return;
            }
            let published = 0, problem = '';
            for (const post of res.items) {
              try { await api('/api/ai-clipping/publish', { method: 'POST', body: { post_id: post.id } }); published += 1; }
              catch (e) { problem = e.message; }
            }
            if (published) {
              close();
              toast('Published', `${published} post(s)`, 'ok');
            } else {
              toast('Publish blocked', problem, 'err');
              const hint = $('#platHint', root);
              if (hint) hint.innerHTML = `<b style="color:#ff9c9c">${esc(problem)}</b>`;
            }
          } catch (e) { toast('Could not schedule', e.message, 'err'); }
        };
        $('#pubDraft', root).addEventListener('click', () => submit(false));
        $('#pubNow', root).addEventListener('click', () => submit(true));
      },
    });
  }

  function bulkScheduleModal() {
    publishModal(selectedMoments());
  }

  /* ----------------------------------------------------------- find/clips */
  async function runFind() {
    const q = $('#findQuery').value.trim();
    if (!q) return toast('Type a search', 'Describe the moment you want', 'err');
    const btn = $('#findBtn');
    btn.disabled = true; btn.innerHTML = '<span class="spinner"></span>';
    try {
      const res = await api('/api/ai-clipping/find', {
        method: 'POST', body: { video_id: state.video.id, query: q },
      });
      if (!res.added) {
        $('#findHint').textContent = `No new moments matched “${q}”.`;
      } else {
        state.moments = state.moments.concat(res.moments);
        $('#tabViral').click();
        $('#filters').classList.remove('open');
        renderGrid();
        toast(`${res.added} new moment(s)`, `Matched “${q}” (${res.source === 'llm' ? 'AI' : 'transcript scorer'})`, 'ok');
      }
    } catch (e) {
      toast('Search failed', e.message, 'err');
    } finally {
      btn.disabled = false; btn.textContent = 'Search';
    }
  }

  async function pollJob(taskId, label) {
    for (;;) {
      const data = await api(`/api/ai-clipping/jobs/${taskId}`);
      if (data.status === 'done') return data.result;
      if (data.status === 'error' || data.status === 'cancelled') {
        throw new Error(data.error || `${label} failed`);
      }
      await new Promise((r) => setTimeout(r, 1500));
    }
  }

  /* ------------------------------------------------------------- library */
  async function loadRecent() {
    try {
      const data = await api('/api/ai-clipping/videos');
      const items = (data.items || []).slice(0, 4);
      $('#recentList').innerHTML = items.length ? items.map((v) => `
        <a class="export-row" href="#results/${esc(v.id)}" style="text-decoration:none">
          <span class="grow"><b>${esc(v.title)}</b>
            <small>${v.moment_count} clip(s) · ${fmt.time(v.duration)} · ${esc(v.platform || '')}</small></span>
          <span style="color:var(--green)">→</span>
        </a>`).join('') : '<p class="hint">No projects yet — clip your first video above.</p>';
    } catch (e) { /* offline */ }
  }

  function bindLibrary() {
    // nothing to bind until the view opens
  }

  async function renderLibrary() {
    $('#libGrid').innerHTML = '<div class="skeleton" style="height:150px"></div><div class="skeleton" style="height:150px"></div>';
    try {
      const [videos, exports] = await Promise.all([
        api('/api/ai-clipping/videos'),
        api('/api/ai-clipping/exports'),
      ]);
      const items = videos.items || [];
      $('#libGrid').innerHTML = items.length ? items.map((v) => `
        <div class="lib-card" data-open="${esc(v.id)}">
          <div class="thumb" style="background-image:url('${esc(v.thumb || '')}')"></div>
          <div class="meta">
            <strong>${esc(v.title)}</strong>
            <small>${v.moment_count} clips · ${fmt.time(v.duration)} · ${esc(v.source_type || '')}</small>
            <div style="display:flex;gap:6px;margin-top:8px">
              <button class="btn btn-ghost btn-sm" data-open="${esc(v.id)}">Open</button>
              <button class="btn btn-danger btn-sm" data-del="${esc(v.id)}">Delete</button>
            </div>
          </div>
        </div>`).join('') : '<div class="empty-state" style="grid-column:1/-1"><span class="big">▤</span><p>No clipping projects yet.</p></div>';

      $$('#libGrid [data-open]').forEach((el) => el.addEventListener('click', (e) => {
        if (e.target.closest('[data-del]')) return;
        location.hash = 'results/' + el.dataset.open;
      }));
      $$('#libGrid [data-del]').forEach((b) => b.addEventListener('click', async (e) => {
        e.stopPropagation();
        if (!confirm('Delete this project and its renders?')) return;
        try {
          await api(`/api/ai-clipping/videos/${b.dataset.del}`, { method: 'DELETE' });
          renderLibrary(); loadRecent();
          toast('Project deleted', '', 'ok');
        } catch (err) { toast('Delete failed', err.message, 'err'); }
      }));

      const ex = exports.items || [];
      $('#exportList').innerHTML = ex.length ? ex.map((item) => `
        <div class="export-row">
          <span class="grow"><b>${esc(item.moment_ids ? (item.files || [])[0]?.file || 'Export' : 'Montage')}</b>
            <small>${esc(item.video_title || item.title || '')} · ${esc(item.resolution || '')} ·
              ${fmt.when(item.created_at)}</small></span>
          <a class="btn btn-ghost btn-sm" href="${esc(absUrl(item.zip_url || item.url))}" download>Download</a>
        </div>`).join('') : '<p class="hint">No exports yet.</p>';
    } catch (e) {
      $('#libGrid').innerHTML = `<div class="err">${esc(e.message)}</div>`;
    }
  }

  /* ----------------------------------------------------------- scheduler */
  let calStart = (() => {
    const d = new Date(); d.setHours(0, 0, 0, 0);
    return d;
  })();

  function bindScheduler() {
    $('#calPrev').addEventListener('click', () => { calStart = new Date(calStart.getTime() - 7 * 864e5); renderScheduler(); });
    $('#calNext').addEventListener('click', () => { calStart = new Date(calStart.getTime() + 7 * 864e5); renderScheduler(); });
  }

  async function renderScheduler() {
    $('#calLabel').textContent = `Week of ${calStart.toLocaleDateString([], { month: 'short', day: 'numeric' })}`;
    let items = [];
    try { items = (await api('/api/ai-clipping/schedule')).items || []; }
    catch (e) { toast('Could not load schedule', e.message, 'err'); }
    const counts = {};
    items.forEach((p) => { counts[p.status] = (counts[p.status] || 0) + 1; });
    $('#calCounts').textContent = Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(' · ') || 'nothing scheduled';

    const days = Array.from({ length: 7 }, (_, i) => new Date(calStart.getTime() + i * 864e5));
    $('#calGrid').innerHTML = days.map((day) => {
      const next = new Date(day.getTime() + 864e5);
      const dayPosts = items.filter((p) => p.when * 1000 >= day.getTime() && p.when * 1000 < next.getTime());
      const today = new Date().toDateString() === day.toDateString();
      return `<div class="cal-day ${today ? 'today' : ''}" data-ts="${day.getTime() / 1000}">
          <h5><span>${day.toLocaleDateString([], { weekday: 'short', day: 'numeric' })}</span>
            <span>${dayPosts.length || ''}</span></h5>
          ${dayPosts.map((p) => `
            <div class="post ${esc(p.status)}" draggable="true" data-post="${esc(p.id)}" title="${esc(p.title)}">
              <b>${esc(p.title)}</b>
              <small>${new Date(p.when * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                · ${(p.platforms || []).join(', ')}</small>
            </div>`).join('')}
        </div>`;
    }).join('');

    $$('.post').forEach((el) => {
      el.addEventListener('dragstart', (e) => {
        e.dataTransfer.setData('text/plain', el.dataset.post);
        e.dataTransfer.effectAllowed = 'move';
      });
      el.addEventListener('click', () => {
        const post = items.find((p) => p.id === el.dataset.post);
        if (post) postModal(post);
      });
    });
    $$('.cal-day').forEach((cell) => {
      cell.addEventListener('dragover', (e) => { e.preventDefault(); cell.classList.add('over'); });
      cell.addEventListener('dragleave', () => cell.classList.remove('over'));
      cell.addEventListener('drop', async (e) => {
        e.preventDefault(); cell.classList.remove('over');
        const postId = e.dataTransfer.getData('text/plain');
        if (!postId) return;
        const base = Number(cell.dataset.ts);
        const post = items.find((p) => p.id === postId);
        const hours = post ? new Date(post.when * 1000).getHours() : 10;
        const minutes = post ? new Date(post.when * 1000).getMinutes() : 0;
        const when = base + hours * 3600 + minutes * 60;
        try {
          await api(`/api/ai-clipping/schedule/${postId}`, { method: 'PATCH', body: { when } });
          renderScheduler(); toast('Rescheduled', fmt.when(when), 'ok');
        } catch (err) { toast('Could not reschedule', err.message, 'err'); }
      });
    });
  }

  function postModal(post) {
    openModal({
      title: post.title || 'Scheduled clip',
      body: `<div class="kv">
          <dt>Platforms</dt><dd>${(post.platforms || []).join(', ')}</dd>
          <dt>When</dt><dd>${fmt.when(post.when)}</dd>
          <dt>Status</dt><dd>${esc(post.status)}</dd>
          <dt>Description</dt><dd>${esc(post.description || '—')}</dd>
          <dt>Hashtags</dt><dd>${esc((post.hashtags || []).join(' '))}</dd>
        </div>
        ${post.video ? `<video src="${esc(absUrl(post.video))}" controls style="width:100%;margin-top:12px;border-radius:10px"></video>` : ''}`,
      footer: `<button class="btn btn-danger" id="postDel">Delete</button>
               <button class="btn btn-ghost" id="postNow">Publish now</button>
               <button class="btn btn-primary" data-close>Close</button>`,
      onMount(root, close) {
        $('#postNow', root).addEventListener('click', async () => {
          try {
            await api('/api/ai-clipping/publish', { method: 'POST', body: { post_id: post.id } });
            close(); renderScheduler(); toast('Published', '', 'ok');
          } catch (e) { toast('Publish failed', e.message, 'err'); }
        });
        $('#postDel', root).addEventListener('click', async () => {
          try {
            await api(`/api/ai-clipping/schedule/${post.id}`, { method: 'DELETE' });
            close(); renderScheduler(); toast('Removed', '', 'ok');
          } catch (e) { toast('Could not remove', e.message, 'err'); }
        });
      },
    });
  }

  /* ------------------------------------------------------------- routing */
  function showView(name) {
    $$('.cm-view').forEach((v) => v.classList.toggle('active', v.id === 'view-' + name));
    /* the shared sidebar (assets/js/sidebar.js) keeps "AI Video Clipping" lit:
       this whole page is one destination, whatever view is open. */
  }

  function route() {
    const hash = location.hash.replace(/^#/, '') || 'input';
    const [page, arg] = hash.split('/');
    if (page === 'results' && arg) { showView('results'); openVideo(arg); return; }
    if (page === 'library') { showView('library'); renderLibrary(); return; }
    if (page === 'scheduler') { showView('scheduler'); renderScheduler(); return; }
    showView('input');
    loadRecent();
  }

  window.addEventListener('hashchange', route);
  document.addEventListener('DOMContentLoaded', boot);
})();
