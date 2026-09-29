(() => {
  'use strict';

  const MAX_UPLOAD_MB = Number(document.body.dataset.maxUploadMb) || 25;

  const state = {
    documents: [],
    activeDocId: null,
    isUploading: false,
    isAsking: false,
    theme: safeStorageGet('documind-theme') || 'dark'
  };

  const elements = {
    sidebar: document.getElementById('sidebar'),
    sidebarOverlay: document.getElementById('sidebarOverlay'),
    uploadZone: document.getElementById('uploadZone'),
    uploadStatus: document.getElementById('uploadStatus'),
    fileInput: document.getElementById('fileInput'),
    docItems: document.getElementById('docItems'),
    emptyState: document.getElementById('emptyState'),
    docCount: document.getElementById('docCount'),
    clearAllBtn: document.getElementById('clearAllBtn'),
    activeDocLabel: document.getElementById('activeDocLabel'),
    welcomeScreen: document.getElementById('welcomeScreen'),
    suggestions: document.getElementById('suggestions'),
    chatArea: document.getElementById('chatArea'),
    messages: document.getElementById('messages'),
    composer: document.getElementById('composer'),
    queryInput: document.getElementById('queryInput'),
    sendBtn: document.getElementById('sendBtn'),
    themeToggle: document.getElementById('themeToggle'),
    mobileSidebarToggle: document.getElementById('mobileSidebarToggle'),
    toastContainer: document.getElementById('toastContainer')
  };

  const ICONS = {
    assistant: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 3.5 14.2 9.8 20.5 12l-6.3 2.2L12 20.5l-2.2-6.3L3.5 12l6.3-2.2L12 3.5Z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/><path d="m19 3 .65 1.85L21.5 5.5l-1.85.65L19 8l-.65-1.85L16.5 5.5l1.85-.65L19 3Z" fill="currentColor"/></svg>',
    user: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="12" cy="8" r="3.25" stroke="currentColor" stroke-width="1.5"/><path d="M5.5 20c.45-3.2 2.8-5 6.5-5s6.05 1.8 6.5 5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
    source: '<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M6.5 3.75h6l4 4v8.5a1.5 1.5 0 0 1-1.5 1.5h-8a1.5 1.5 0 0 1-1.5-1.5v-11a1.5 1.5 0 0 1 1.5-1.5Z" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><path d="M12.5 4v4h4M8 11h5m-5 3h5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>',
    trash: '<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M3.5 5.5h13M8 5.5V3.75h4V5.5m3.5 0-.7 10.2a1.5 1.5 0 0 1-1.5 1.4H6.7a1.5 1.5 0 0 1-1.5-1.4L4.5 5.5m3.25 3.25v5m4.5-5v5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>'
  };

  function safeStorageGet(key) {
    try { return window.localStorage.getItem(key); } catch (_error) { return null; }
  }

  function safeStorageSet(key, value) {
    try { window.localStorage.setItem(key, value); } catch (_error) { /* Storage is optional. */ }
  }

  function init() {
    applyTheme(state.theme);
    bindEvents();
    updateComposerState();
    loadDocuments();
    requestAnimationFrame(() => document.body.classList.add('is-ready'));
  }

  function applyTheme(theme) {
    state.theme = theme === 'light' ? 'light' : 'dark';
    document.documentElement.dataset.theme = state.theme;
    elements.themeToggle.setAttribute(
      'aria-label',
      state.theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'
    );
    safeStorageSet('documind-theme', state.theme);
  }

  function bindEvents() {
    elements.themeToggle.addEventListener('click', () => {
      applyTheme(state.theme === 'dark' ? 'light' : 'dark');
    });

    elements.mobileSidebarToggle.addEventListener('click', openSidebar);
    elements.sidebarOverlay.addEventListener('click', closeSidebar);
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') closeSidebar();
      if (event.key === '/' && !isTypingTarget(event.target) && state.documents.length) {
        event.preventDefault();
        elements.queryInput.focus();
      }
    });
    window.addEventListener('resize', () => {
      if (window.innerWidth > 760) closeSidebar();
    });

    elements.uploadZone.addEventListener('click', (event) => {
      if (!event.target.closest('input') && !state.isUploading) elements.fileInput.click();
    });
    elements.uploadZone.addEventListener('keydown', (event) => {
      if ((event.key === 'Enter' || event.key === ' ') && !state.isUploading) {
        event.preventDefault();
        elements.fileInput.click();
      }
    });
    elements.uploadZone.addEventListener('dragover', (event) => {
      event.preventDefault();
      if (!state.isUploading) elements.uploadZone.classList.add('is-dragging');
    });
    elements.uploadZone.addEventListener('dragleave', (event) => {
      if (!elements.uploadZone.contains(event.relatedTarget)) elements.uploadZone.classList.remove('is-dragging');
    });
    elements.uploadZone.addEventListener('drop', (event) => {
      event.preventDefault();
      elements.uploadZone.classList.remove('is-dragging');
      uploadFiles(Array.from(event.dataTransfer.files || []));
    });
    elements.fileInput.addEventListener('change', () => {
      uploadFiles(Array.from(elements.fileInput.files || []));
      elements.fileInput.value = '';
    });

    elements.docItems.addEventListener('click', (event) => {
      const deleteButton = event.target.closest('[data-delete-id]');
      if (deleteButton) {
        event.stopPropagation();
        deleteDocument(deleteButton.dataset.deleteId);
        return;
      }
      const selectButton = event.target.closest('[data-select-id]');
      if (selectButton) selectDocument(selectButton.dataset.selectId);
    });

    elements.clearAllBtn.addEventListener('click', clearLibrary);
    elements.composer.addEventListener('submit', (event) => {
      event.preventDefault();
      askQuestion(elements.queryInput.value);
    });
    elements.queryInput.addEventListener('input', () => {
      elements.queryInput.style.height = 'auto';
      elements.queryInput.style.height = `${Math.min(elements.queryInput.scrollHeight, 150)}px`;
      updateComposerState();
    });
    elements.queryInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        elements.composer.requestSubmit();
      }
    });
    elements.suggestions.addEventListener('click', (event) => {
      const suggestion = event.target.closest('[data-question]');
      if (!suggestion || !state.documents.length) return;
      elements.queryInput.value = suggestion.dataset.question;
      updateComposerState();
      askQuestion(elements.queryInput.value);
    });
  }

  function isTypingTarget(target) {
    return target instanceof HTMLElement && (
      target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)
    );
  }

  function openSidebar() {
    if (window.innerWidth > 760) return;
    elements.sidebar.classList.add('is-open');
    elements.sidebarOverlay.classList.add('is-visible');
    elements.mobileSidebarToggle.setAttribute('aria-expanded', 'true');
  }

  function closeSidebar() {
    elements.sidebar.classList.remove('is-open');
    elements.sidebarOverlay.classList.remove('is-visible');
    elements.mobileSidebarToggle.setAttribute('aria-expanded', 'false');
  }

  async function requestJson(url, options = {}) {
    const response = await fetch(url, options);
    let data;
    try { data = await response.json(); } catch (_error) { data = {}; }
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
    return data;
  }

  async function loadDocuments() {
    try {
      const data = await requestJson('/api/documents');
      state.documents = Array.isArray(data.documents) ? data.documents : [];
      if (!state.documents.some((document) => document.doc_id === state.activeDocId)) {
        state.activeDocId = null;
      }
      renderDocuments();
      updateActiveDocumentLabel();
      updateComposerState();
    } catch (error) {
      showToast(error.message || 'Could not load your document library.', 'error');
    }
  }

  function renderDocuments() {
    elements.docItems.replaceChildren();
    elements.docCount.textContent = String(state.documents.length);
    elements.clearAllBtn.disabled = !state.documents.length || state.isUploading;

    if (!state.documents.length) {
      elements.docItems.appendChild(elements.emptyState);
      return;
    }

    for (const doc of state.documents) {
      const card = document.createElement('article');
      card.className = `document-card${doc.doc_id === state.activeDocId ? ' is-active' : ''}`;
      card.setAttribute('role', 'listitem');

      const selectButton = document.createElement('button');
      selectButton.className = 'document-card__select';
      selectButton.type = 'button';
      selectButton.dataset.selectId = doc.doc_id;
      selectButton.setAttribute('aria-pressed', String(doc.doc_id === state.activeDocId));
      selectButton.setAttribute('aria-label', `Search ${doc.filename}`);

      const fileIcon = document.createElement('span');
      fileIcon.className = `file-icon file-icon--${fileExtension(doc.filename)}`;
      fileIcon.textContent = fileExtension(doc.filename);

      const copy = document.createElement('span');
      copy.className = 'document-card__copy';
      const filename = document.createElement('span');
      filename.className = 'document-card__name';
      filename.textContent = doc.filename;
      filename.title = doc.filename;
      const meta = document.createElement('span');
      meta.className = 'document-card__meta';
      meta.textContent = `${Number(doc.chunks) || 0} passages · ${formatDate(doc.created_at)}`;
      copy.append(filename, meta);
      selectButton.append(fileIcon, copy);

      const deleteButton = document.createElement('button');
      deleteButton.type = 'button';
      deleteButton.className = 'document-card__delete';
      deleteButton.dataset.deleteId = doc.doc_id;
      deleteButton.setAttribute('aria-label', `Remove ${doc.filename}`);
      deleteButton.title = 'Remove document';
      deleteButton.innerHTML = ICONS.trash;
      card.append(selectButton, deleteButton);
      elements.docItems.appendChild(card);
    }
  }

  function fileExtension(filename) {
    const extension = String(filename).split('.').pop().toLowerCase();
    return extension === 'markdown' ? 'md' : (extension || 'file');
  }

  function formatDate(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return 'just now';
    return new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric' }).format(date);
  }

  function selectDocument(docId) {
    state.activeDocId = state.activeDocId === docId ? null : docId;
    renderDocuments();
    updateActiveDocumentLabel();
    closeSidebar();
    if (state.documents.length && !elements.messages.children.length) elements.queryInput.focus();
  }

  function updateActiveDocumentLabel() {
    const activeDocument = state.documents.find((document) => document.doc_id === state.activeDocId);
    elements.activeDocLabel.textContent = activeDocument
      ? `Focused on ${activeDocument.filename} · ask across your library by deselecting it`
      : 'Search your library or choose a document to focus your answer.';
  }

  async function uploadFiles(files) {
    if (state.isUploading || !files.length) return;
    const accepted = [];
    const rejected = [];
    for (const file of files) {
      const extension = file.name.split('.').pop().toLowerCase();
      if (!['pdf', 'txt', 'md', 'markdown'].includes(extension)) rejected.push(`${file.name}: unsupported format`);
      else if (file.size > MAX_UPLOAD_MB * 1024 * 1024) rejected.push(`${file.name}: larger than ${MAX_UPLOAD_MB} MB`);
      else accepted.push(file);
    }
    rejected.forEach((message) => showToast(message, 'error'));
    if (!accepted.length) return;

    state.isUploading = true;
    elements.uploadZone.classList.add('is-processing');
    elements.uploadZone.setAttribute('aria-disabled', 'true');
    elements.fileInput.disabled = true;
    elements.uploadStatus.hidden = false;
    elements.clearAllBtn.disabled = true;

    let lastUploadedId = null;
    for (let position = 0; position < accepted.length; position += 1) {
      const file = accepted[position];
      elements.uploadStatus.textContent = `Reading and indexing ${file.name} locally…`;
      const form = new FormData();
      form.append('file', file);
      try {
        const data = await requestJson('/api/upload', { method: 'POST', body: form });
        lastUploadedId = data.document.doc_id;
        const suffix = data.duplicate ? 'Already in your library' : `${data.document.chunks} passages indexed`;
        showToast(`${file.name} · ${suffix}`, 'success');
      } catch (error) {
        showToast(`${file.name} · ${error.message}`, 'error');
      }
      if (position < accepted.length - 1) {
        elements.uploadStatus.textContent = `Indexed ${position + 1} of ${accepted.length} documents…`;
      }
    }

    state.isUploading = false;
    elements.uploadZone.classList.remove('is-processing');
    elements.uploadZone.removeAttribute('aria-disabled');
    elements.fileInput.disabled = false;
    elements.uploadStatus.hidden = true;
    await loadDocuments();
    if (lastUploadedId && state.documents.some((document) => document.doc_id === lastUploadedId)) {
      state.activeDocId = lastUploadedId;
      renderDocuments();
      updateActiveDocumentLabel();
    }
    updateComposerState();
  }

  async function deleteDocument(docId) {
    const document = state.documents.find((item) => item.doc_id === docId);
    if (!document) return;
    if (!window.confirm(`Remove “${document.filename}” from your library?`)) return;
    try {
      await requestJson(`/api/delete/${encodeURIComponent(docId)}`, { method: 'DELETE' });
      if (state.activeDocId === docId) state.activeDocId = null;
      await loadDocuments();
      showToast('Document removed from your library.', 'success');
    } catch (error) {
      showToast(error.message, 'error');
    }
  }

  async function clearLibrary() {
    if (!state.documents.length || state.isUploading) return;
    if (!window.confirm(`Remove all ${state.documents.length} documents from your library?`)) return;
    const documents = [...state.documents];
    try {
      const results = await Promise.allSettled(
        documents.map((document) => requestJson(`/api/delete/${encodeURIComponent(document.doc_id)}`, { method: 'DELETE' }))
      );
      const failed = results.filter((result) => result.status === 'rejected').length;
      state.activeDocId = null;
      elements.messages.replaceChildren();
      elements.messages.hidden = true;
      elements.welcomeScreen.hidden = false;
      await loadDocuments();
      showToast(failed ? `Some documents could not be removed (${failed}).` : 'Your library is clear.', failed ? 'error' : 'success');
    } catch (error) {
      showToast(error.message, 'error');
    }
  }

  function updateComposerState() {
    const hasDocuments = state.documents.length > 0;
    elements.queryInput.disabled = !hasDocuments || state.isAsking;
    elements.queryInput.placeholder = hasDocuments
      ? 'Ask a question grounded in your documents…'
      : 'Upload a document to start asking…';
    elements.sendBtn.disabled = !hasDocuments || state.isAsking || !elements.queryInput.value.trim();
    elements.clearAllBtn.disabled = !hasDocuments || state.isUploading;
    elements.suggestions.querySelectorAll('button').forEach((button) => {
      button.disabled = !hasDocuments || state.isAsking;
    });
  }

  async function askQuestion(rawQuery) {
    const query = String(rawQuery || '').trim();
    if (!query || !state.documents.length || state.isAsking) return;

    state.isAsking = true;
    elements.welcomeScreen.hidden = true;
    elements.messages.hidden = false;
    appendMessage('user', query);
    elements.queryInput.value = '';
    elements.queryInput.style.height = 'auto';
    updateComposerState();

    const pending = appendMessage('assistant', 'Searching your local passages…', null, true);
    try {
      const data = await requestJson('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, doc_id: state.activeDocId })
      });
      pending.text.textContent = data.answer;
      pending.message.classList.remove('message--pending');
      if (data.sources && data.sources.length) appendSources(pending.content, data.sources);
    } catch (error) {
      pending.text.textContent = error.message;
      pending.message.classList.remove('message--pending');
      pending.message.classList.add('message--error');
    } finally {
      state.isAsking = false;
      updateComposerState();
      if (state.documents.length) elements.queryInput.focus();
      scrollToLatest();
    }
  }

  function appendMessage(role, text, sources = null, pending = false) {
    const message = document.createElement('article');
    message.className = `message message--${role}${pending ? ' message--pending' : ''}`;

    const avatar = document.createElement('span');
    avatar.className = `message__avatar message__avatar--${role}`;
    avatar.innerHTML = role === 'user' ? ICONS.user : ICONS.assistant;

    const content = document.createElement('div');
    content.className = 'message__content';
    const heading = document.createElement('div');
    heading.className = 'message__heading';
    const author = document.createElement('strong');
    author.textContent = role === 'user' ? 'You' : 'DocuMind';
    const timestamp = document.createElement('time');
    timestamp.dateTime = new Date().toISOString();
    timestamp.textContent = new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' }).format(new Date());
    heading.append(author, timestamp);

    const textElement = document.createElement('p');
    textElement.className = 'message__text';
    textElement.textContent = text;
    content.append(heading, textElement);
    if (sources && sources.length) appendSources(content, sources);
    message.append(avatar, content);
    elements.messages.appendChild(message);
    scrollToLatest();
    return { message, content, text: textElement };
  }

  function appendSources(parent, sources) {
    const details = document.createElement('details');
    details.className = 'sources';
    const summary = document.createElement('summary');
    summary.innerHTML = `${ICONS.source}<span>Evidence</span><span class="sources__count">${sources.length} ${sources.length === 1 ? 'passage' : 'passages'}</span>`;
    const list = document.createElement('div');
    list.className = 'sources__list';
    for (const source of sources) {
      const item = document.createElement('div');
      item.className = 'source-card';
      const top = document.createElement('div');
      top.className = 'source-card__top';
      const filename = document.createElement('strong');
      filename.textContent = source.filename || 'Document';
      const match = document.createElement('span');
      match.className = 'source-card__match';
      match.textContent = `${Math.round((Number(source.score) || 0) * 100)}% match`;
      top.append(filename, match);
      const location = document.createElement('span');
      location.className = 'source-card__location';
      const position = source.page ? `Page ${source.page}` : `Passage ${source.chunk_index}`;
      location.textContent = `${position} · Passage ${source.chunk_index}`;
      const excerpt = document.createElement('p');
      excerpt.textContent = source.text || '';
      item.append(top, location, excerpt);
      list.appendChild(item);
    }
    details.append(summary, list);
    parent.appendChild(details);
  }

  function scrollToLatest() {
    elements.chatArea.scrollTo({ top: elements.chatArea.scrollHeight, behavior: 'smooth' });
  }

  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast--${type}`;
    const marker = document.createElement('span');
    marker.className = 'toast__marker';
    marker.setAttribute('aria-hidden', 'true');
    const text = document.createElement('span');
    text.textContent = message;
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'toast__close';
    close.setAttribute('aria-label', 'Dismiss notification');
    close.textContent = '×';
    close.addEventListener('click', () => toast.remove());
    toast.append(marker, text, close);
    elements.toastContainer.appendChild(toast);
    window.setTimeout(() => {
      toast.classList.add('toast--leaving');
      window.setTimeout(() => toast.remove(), 220);
    }, 4200);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
