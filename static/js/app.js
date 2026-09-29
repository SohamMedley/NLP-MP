/* DocuRAG frontend — vanilla JavaScript, no frameworks. */
(() => {
  "use strict";

  const MAX_MB = Number(document.body.dataset.maxUploadMb) || 50;
  const HISTORY_LIMIT = 6; // messages sent for follow-up context (3 Q/A pairs)

  const $ = (id) => document.getElementById(id);
  const el = {
    dropzone: $("dropzone"), fileInput: $("fileInput"), progress: $("progress"),
    progressStage: $("progressStage"), progressFile: $("progressFile"), progressBar: $("progressBar"),
    uploadMessages: $("uploadMessages"), docList: $("docList"), emptyDocs: $("emptyDocs"),
    docCount: $("docCount"), statDocs: $("statDocs"), statPages: $("statPages"), statChunks: $("statChunks"),
    indexStatus: $("indexStatus"), rebuildBtn: $("rebuildBtn"), clearDocsBtn: $("clearDocsBtn"),
    messages: $("messages"), welcome: $("welcome"), welcomeText: $("welcomeText"), suggestions: $("suggestions"),
    composer: $("composer"), input: $("questionInput"), sendBtn: $("sendBtn"), clearChatBtn: $("clearChatBtn"),
    debugToggle: $("debugToggle"), statusPill: $("statusPill"), statusText: $("statusText"), toast: $("toast"),
  };

  const state = { history: [], indexReady: false, busyUpload: false, busyAsk: false };

  /* ---------------- helpers ---------------- */
  function h(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    }
    for (const child of children.flat()) {
      if (child == null || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  const escapeHtml = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");

  async function api(url, options = {}) {
    let res;
    try {
      res = await fetch(url, options);
    } catch {
      throw new Error("Network error — is the server running?");
    }
    let data = {};
    try { data = await res.json(); } catch { /* non-JSON response */ }
    if (!res.ok || data.success === false) {
      const err = new Error(data.error || `Request failed (${res.status}).`);
      err.data = data;
      throw err;
    }
    return data;
  }

  let toastTimer;
  function toast(message, isError = false) {
    el.toast.textContent = message;
    el.toast.classList.toggle("err", isError);
    el.toast.classList.remove("hidden");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.toast.classList.add("hidden"), 4000);
  }

  const scrollToBottom = () => { el.messages.scrollTop = el.messages.scrollHeight; };

  /* ---------------- safe Markdown ----------------
     All text is HTML-escaped FIRST, then a small set of Markdown patterns is
     converted to tags. Raw HTML from the LLM can therefore never execute. */
  function inline(text) {
    return text
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>")
      .replace(/\[Source (\d+)\]/g, '<span class="cite">Source $1</span>');
  }

  function renderMarkdown(src) {
    const escaped = escapeHtml(src.replace(/\r\n/g, "\n"));
    const codeBlocks = [];
    const withoutCode = escaped.replace(/```[\w-]*\n?([\s\S]*?)```/g, (_, code) => {
      codeBlocks.push(`<pre><code>${code.replace(/\n$/, "")}</code></pre>`);
      return `\u0000${codeBlocks.length - 1}\u0000`;
    });

    const out = [];
    let list = null; // {type, items}
    let para = [];
    const flushPara = () => { if (para.length) { out.push(`<p>${inline(para.join("<br>"))}</p>`); para = []; } };
    const flushList = () => {
      if (list) { out.push(`<${list.type}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.type}>`); list = null; }
    };

    for (const line of withoutCode.split("\n")) {
      const trimmed = line.trim();
      const code = trimmed.match(/^\u0000(\d+)\u0000$/);
      const bullet = trimmed.match(/^[-*•]\s+(.*)$/);
      const numbered = trimmed.match(/^\d+[.)]\s+(.*)$/);
      const heading = trimmed.match(/^#{1,4}\s+(.*)$/);
      if (code) { flushPara(); flushList(); out.push(codeBlocks[Number(code[1])]); }
      else if (!trimmed) { flushPara(); flushList(); }
      else if (heading) { flushPara(); flushList(); out.push(`<h4>${inline(heading[1])}</h4>`); }
      else if (bullet || numbered) {
        flushPara();
        const type = bullet ? "ul" : "ol";
        if (!list || list.type !== type) { flushList(); list = { type, items: [] }; }
        list.items.push((bullet || numbered)[1]);
      } else if (list && /^\s{2,}/.test(line)) {
        list.items[list.items.length - 1] += " " + trimmed;
      } else { flushList(); para.push(trimmed); }
    }
    flushPara(); flushList();
    return out.join("").replace(/\u0000(\d+)\u0000/g, (_, i) => codeBlocks[Number(i)]);
  }

  /* ---------------- documents ---------------- */
  function renderDocuments(documents, stats) {
    el.docList.replaceChildren(...documents.map((doc) => h("li", { class: "doc-item" },
      h("div", { class: "doc-icon", "aria-hidden": "true" }, "PDF"),
      h("div", { class: "doc-meta" },
        h("div", { class: "doc-name", title: doc.filename }, doc.filename),
        h("div", { class: "doc-sub" }, `${doc.pages} pages · ${doc.chunks} chunks`)),
      h("button", {
        class: "icon-btn", type: "button", title: `Remove ${doc.filename}`,
        "aria-label": `Remove ${doc.filename}`, onclick: () => removeDocument(doc),
      }, "✕"))));

    el.emptyDocs.classList.toggle("hidden", documents.length > 0);
    el.docCount.textContent = stats.documents;
    el.statDocs.textContent = stats.documents;
    el.statPages.textContent = stats.pages;
    el.statChunks.textContent = stats.chunks;
    el.indexStatus.textContent = stats.index_ready
      ? `Index ready · ${stats.documents} document${stats.documents === 1 ? "" : "s"} indexed • ${stats.chunks} chunks`
      : "Index empty";
    el.indexStatus.classList.toggle("ready", stats.index_ready);

    state.indexReady = stats.index_ready;
    el.rebuildBtn.disabled = !stats.index_ready || state.busyUpload;
    el.clearDocsBtn.disabled = !documents.length || state.busyUpload;
    el.suggestions.classList.toggle("hidden", !stats.index_ready);
    el.welcomeText.textContent = stats.index_ready
      ? "Your documents are indexed. Ask anything — answers are generated only from retrieved passages, with citations."
      : "Upload one or more PDFs on the left. DocuRAG retrieves the most relevant passages and answers only from them — with page-level citations.";
    updateComposer();
  }

  async function loadDocuments() {
    try {
      const data = await api("/api/documents");
      renderDocuments(data.documents, data.stats);
    } catch (err) { toast(err.message, true); }
  }

  async function removeDocument(doc) {
    if (!confirm(`Remove "${doc.filename}" and its embeddings from the index?`)) return;
    try {
      await api(`/api/documents/${encodeURIComponent(doc.id)}`, { method: "DELETE" });
      toast(`Removed ${doc.filename}`);
      loadDocuments();
    } catch (err) { toast(err.message, true); }
  }

  el.clearDocsBtn.addEventListener("click", async () => {
    if (!confirm("Remove all documents and clear the semantic index?")) return;
    try {
      await api("/api/clear", { method: "POST" });
      el.uploadMessages.replaceChildren();
      toast("All documents cleared");
      loadDocuments();
    } catch (err) { toast(err.message, true); }
  });

  el.rebuildBtn.addEventListener("click", async () => {
    el.rebuildBtn.disabled = true;
    el.rebuildBtn.textContent = "Rebuilding…";
    try {
      const data = await api("/api/rebuild", { method: "POST" });
      toast(`Index rebuilt · ${data.chunks} chunks re-embedded`);
    } catch (err) { toast(err.message, true); }
    el.rebuildBtn.textContent = "Rebuild index";
    loadDocuments();
  });

  /* ---------------- upload ---------------- */
  function addUploadMessage(kind, text) {
    el.uploadMessages.append(h("div", { class: `msg-line ${kind}` }, text));
  }

  function setUploading(on) {
    state.busyUpload = on;
    el.dropzone.classList.toggle("disabled", on);
    el.progress.classList.toggle("hidden", !on);
    el.rebuildBtn.disabled = on || !state.indexReady;
    el.clearDocsBtn.disabled = on;
  }

  function uploadWithProgress(formData) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/upload");
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          const pct = Math.round((e.loaded / e.total) * 100);
          el.progressStage.textContent = `Uploading… ${pct}%`;
          el.progressBar.style.width = `${pct * 0.2}%`;
        }
      };
      xhr.onload = () => {
        let data = {};
        try { data = JSON.parse(xhr.responseText); } catch { /* ignore */ }
        if (xhr.status >= 200 && xhr.status < 300 && data.success) resolve(data);
        else {
          const err = new Error(data.error || `Upload failed (${xhr.status}).`);
          err.data = data;
          reject(err);
        }
      };
      xhr.onerror = () => reject(new Error("Network error during upload."));
      xhr.send(formData);
    });
  }

  const STAGES = ["Reading PDF...", "Extracting text...", "Creating chunks...", "Generating embeddings...", "Building semantic index..."];

  async function pollJob(jobId) {
    for (;;) {
      await new Promise((r) => setTimeout(r, 600));
      const job = await api(`/api/jobs/${jobId}`);
      const stageIdx = Math.max(0, STAGES.indexOf(job.stage));
      const perFile = 80 / job.total_files;
      const done = (job.file_index - 1) * perFile + ((stageIdx + 1) / STAGES.length) * perFile;
      el.progressStage.textContent = job.done ? "Ready" : job.stage;
      el.progressFile.textContent = job.current_file
        ? `${job.current_file} (${job.file_index} of ${job.total_files})` : "";
      el.progressBar.style.width = `${job.done ? 100 : 20 + Math.max(0, done)}%`;
      if (job.done) return job;
    }
  }

  async function handleFiles(fileList) {
    if (state.busyUpload) return;
    const files = Array.from(fileList);
    el.uploadMessages.replaceChildren();
    const valid = [];
    let totalBytes = 0;
    for (const file of files) {
      const isPdf = file.name.toLowerCase().endsWith(".pdf");
      if (!isPdf) { addUploadMessage("err", `${file.name}: Only PDF files are supported.`); continue; }
      if (file.size === 0) { addUploadMessage("err", `${file.name}: File is empty.`); continue; }
      totalBytes += file.size;
      valid.push(file);
    }
    if (!valid.length) return;
    if (totalBytes > MAX_MB * 1024 * 1024) {
      addUploadMessage("err", `Selected files exceed the ${MAX_MB} MB upload limit. Upload fewer or smaller PDFs.`);
      return;
    }

    const form = new FormData();
    valid.forEach((f) => form.append("files", f));
    setUploading(true);
    el.progressBar.style.width = "0%";
    el.progressFile.textContent = valid.length === 1 ? valid[0].name : `${valid.length} files`;
    try {
      const { job_id: jobId, rejected } = await uploadWithProgress(form);
      (rejected || []).forEach((r) => addUploadMessage("err", `${r.filename}: ${r.message}`));
      const job = await pollJob(jobId);
      for (const r of job.results) {
        if (r.status === "indexed") addUploadMessage("ok", `${r.filename}: ${r.pages} pages · ${r.chunks} chunks indexed`);
        else if (r.status === "duplicate") addUploadMessage("warn", `${r.filename}: ${r.message}`);
        else addUploadMessage("err", `${r.filename}: ${r.message}`);
      }
      const s = job.stats;
      if (s && s.index_ready) toast(`${s.documents} document${s.documents === 1 ? "" : "s"} indexed • ${s.chunks} chunks`);
    } catch (err) {
      addUploadMessage("err", err.message);
      (err.data?.results || []).slice(1).forEach((r) => addUploadMessage("err", `${r.filename}: ${r.message}`));
    } finally {
      setUploading(false);
      el.fileInput.value = "";
      loadDocuments();
    }
  }

  el.dropzone.addEventListener("click", () => el.fileInput.click());
  el.dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.fileInput.click(); }
  });
  el.fileInput.addEventListener("change", () => handleFiles(el.fileInput.files));
  ["dragenter", "dragover"].forEach((ev) => el.dropzone.addEventListener(ev, (e) => {
    e.preventDefault(); el.dropzone.classList.add("dragover");
  }));
  ["dragleave", "drop"].forEach((ev) => el.dropzone.addEventListener(ev, (e) => {
    e.preventDefault(); el.dropzone.classList.remove("dragover");
  }));
  el.dropzone.addEventListener("drop", (e) => handleFiles(e.dataTransfer.files));
  // Prevent the browser from opening a PDF dropped outside the zone.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  /* ---------------- chat ---------------- */
  function updateComposer() {
    const enabled = state.indexReady && !state.busyAsk;
    el.input.disabled = !state.indexReady;
    el.input.placeholder = state.indexReady
      ? "Ask a question about your documents…" : "Upload a PDF to start asking questions…";
    el.sendBtn.disabled = !enabled || !el.input.value.trim();
    el.clearChatBtn.disabled = state.history.length === 0 && !el.messages.querySelector(".msg");
  }

  function scoreBadge(score) {
    const cls = score >= 0.5 ? "high" : score < 0.3 ? "low" : "";
    return h("span", { class: `score ${cls}`, title: "Cosine similarity between question and chunk" }, score.toFixed(2));
  }

  function renderSources(sources) {
    return h("div", { class: "sources" },
      h("div", { class: "sources-title" }, "Sources"),
      sources.map((s) => h("details", { class: "source" },
        h("summary", {},
          h("span", { class: "src-num" }, s.number),
          h("span", { class: "src-name", title: s.document }, s.document),
          h("span", { class: "src-page" }, `— Page ${s.page}`),
          scoreBadge(s.score)),
        h("div", { class: "excerpt" }, s.excerpt))));
  }

  function renderDebug(retrieval, searchQuery) {
    const box = h("div", { class: "debug debug-panel" },
      h("div", { class: "sources-title" }, "Retrieved context (debug)"),
      h("div", { class: "debug-query" }, "Search query: ", h("code", {}, searchQuery || "")),
      retrieval.length ? retrieval.map((r, i) => h("details", { class: `source ${r.used ? "" : "unused"}` },
        h("summary", {},
          h("span", { class: "src-num" }, i + 1),
          h("span", { class: "src-name" }, r.document),
          h("span", { class: "src-page" }, `— Page ${r.page}`),
          h("span", { class: `tag ${r.used ? "used" : "dropped"}` }, r.used ? "sent to LLM" : "below threshold"),
          scoreBadge(r.score)),
        h("div", { class: "excerpt" }, r.text))) : h("div", { class: "debug-query" }, "No chunks retrieved."));
    box.classList.toggle("hidden", !el.debugToggle.checked);
    return box;
  }

  function addUserMessage(text) {
    el.welcome.classList.add("hidden");
    el.messages.append(h("div", { class: "msg user" }, h("div", { class: "bubble-user" }, text)));
    scrollToBottom();
  }

  function addPending() {
    const node = h("div", { class: "msg ai" },
      h("div", { class: "avatar", "aria-hidden": "true" }, "AI"),
      h("div", { class: "ai-body" }, h("div", { class: "answer-card" },
        h("span", { class: "typing" }, h("span", { class: "dots" }, h("span"), h("span"), h("span")),
          "Retrieving relevant passages and generating answer…"))));
    el.messages.append(node);
    scrollToBottom();
    return node;
  }

  function fillAnswer(node, data) {
    const body = node.querySelector(".ai-body");
    const card = h("div", { class: `answer-card ${data.grounded ? "" : "not-found"}` });
    if (!data.grounded) card.append(h("div", { class: "answer-tag" }, "⚠ Not found in documents"));
    const md = h("div", { class: "markdown" });
    md.innerHTML = renderMarkdown(data.answer); // safe: escaped before formatting
    card.append(md);
    body.replaceChildren(card);
    if (data.sources?.length) body.append(renderSources(data.sources));
    if (data.retrieval) body.append(renderDebug(data.retrieval, data.search_query));
  }

  function fillError(node, err) {
    const body = node.querySelector(".ai-body");
    body.replaceChildren(h("div", { class: "answer-card error" }, err.message));
    if (err.data?.retrieval) body.append(renderDebug(err.data.retrieval, ""));
  }

  async function ask(question) {
    if (!question || state.busyAsk || !state.indexReady) return;
    state.busyAsk = true;
    el.input.value = "";
    autoresize();
    addUserMessage(question);
    const pending = addPending();
    updateComposer();
    try {
      const data = await api("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history: state.history.slice(-HISTORY_LIMIT) }),
      });
      fillAnswer(pending, data);
      state.history.push({ role: "user", content: question }, { role: "assistant", content: data.answer });
    } catch (err) {
      fillError(pending, err);
    } finally {
      state.busyAsk = false;
      updateComposer();
      scrollToBottom();
      el.input.focus();
    }
  }

  function autoresize() {
    el.input.style.height = "auto";
    el.input.style.height = `${Math.min(el.input.scrollHeight, 180)}px`;
  }

  el.composer.addEventListener("submit", (e) => { e.preventDefault(); ask(el.input.value.trim()); });
  el.input.addEventListener("input", () => { autoresize(); updateComposer(); });
  el.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); ask(el.input.value.trim()); }
  });
  el.suggestions.addEventListener("click", (e) => {
    if (e.target.matches(".chip")) ask(e.target.textContent.trim());
  });
  el.clearChatBtn.addEventListener("click", () => {
    state.history = [];
    el.messages.querySelectorAll(".msg").forEach((m) => m.remove());
    el.welcome.classList.remove("hidden");
    updateComposer();
  });
  el.debugToggle.addEventListener("change", () => {
    document.querySelectorAll(".debug-panel").forEach((p) => p.classList.toggle("hidden", !el.debugToggle.checked));
  });

  /* ---------------- health ---------------- */
  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      if (data.groq_configured) {
        el.statusPill.className = "status-pill ok";
        el.statusText.textContent = "Online";
        el.statusPill.title = `LLM: ${data.model}`;
      } else {
        el.statusPill.className = "status-pill warn";
        el.statusText.textContent = "GROQ_API_KEY missing";
        el.statusPill.title = "Retrieval works, but answers need GROQ_API_KEY on the server.";
      }
    } catch {
      el.statusPill.className = "status-pill err";
      el.statusText.textContent = "Offline";
    }
  }

  checkHealth();
  loadDocuments();
})();
