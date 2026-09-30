/* DocuRAG frontend — vanilla JavaScript, no frameworks. */
(() => {
  "use strict";

  const MAX_MB = Number(document.body.dataset.maxUploadMb) || 50;
  const HISTORY_LIMIT = 6; // messages sent for follow-up context (3 Q/A pairs)
  const WIDE = window.matchMedia("(min-width: 1281px)");
  const MOBILE = window.matchMedia("(max-width: 860px)");

  const $ = (id) => document.getElementById(id);
  const el = {
    dropzone: $("dropzone"), fileInput: $("fileInput"), progress: $("progress"), progressStage: $("progressStage"),
    progressFile: $("progressFile"), progressBar: $("progressBar"), progressPct: $("progressPct"), steps: $("steps"),
    uploadMessages: $("uploadMessages"), docList: $("docList"), emptyDocs: $("emptyDocs"),
    statDocs: $("statDocs"), statPages: $("statPages"), statChunks: $("statChunks"), indexStatus: $("indexStatus"),
    rebuildBtn: $("rebuildBtn"), clearDocsBtn: $("clearDocsBtn"), segCount: $("segCount"), scopeLabel: $("scopeLabel"),
    messages: $("messages"), welcome: $("welcome"), welcomeText: $("welcomeText"), welcomeUpload: $("welcomeUpload"),
    suggestions: $("suggestions"), composer: $("composer"), input: $("questionInput"), sendBtn: $("sendBtn"),
    clearChatBtn: $("clearChatBtn"), debugToggle: $("debugToggle"), statusPill: $("statusPill"), statusText: $("statusText"),
    modelChip: $("modelChip"), evidenceBody: $("evidenceBody"), evidenceSub: $("evidenceSub"),
    evidenceClose: $("evidenceClose"), scrim: $("scrim"), toast: $("toast"),
  };

  const state = { history: [], turns: [], selectedTurn: null, indexReady: false, busyUpload: false, busyAsk: false,
    mode: localStorage.getItem("docurag-mode") === "local" ? "local" : "llm", groqReady: true };

  /* ================= helpers ================= */
  function h(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
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

  const SVG_NS = "http://www.w3.org/2000/svg";
  function icon(paths, size = 16, stroke = 2) {
    const svg = document.createElementNS(SVG_NS, "svg");
    for (const [k, v] of Object.entries({ viewBox: "0 0 24 24", width: size, height: size, fill: "none", stroke: "currentColor",
      "stroke-width": stroke, "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true" })) svg.setAttribute(k, v);
    for (const d of paths) { const p = document.createElementNS(SVG_NS, "path"); p.setAttribute("d", d); svg.append(p); }
    return svg;
  }
  const ICONS = {
    spark: ["M12 3l1.9 5.8L20 10l-6.1 1.2L12 17l-1.9-5.8L4 10l6.1-1.2z"],
    close: ["M18 6 6 18", "m6 6 12 12"],
    copy: ["M8 8h11v13H8z", "M5 16V3h11"],
    check: ["M20 6 9 17l-5-5"],
    doc: ["M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z", "M14 2v6h6"],
  };

  // Stable tint per document so each file is recognisable everywhere.
  const TINTS = ["#6d5ef5", "#e5484d", "#12a594", "#f76b15", "#3e63dd", "#d6409f", "#8e4ec6", "#2b9a66"];
  function tintFor(name) {
    let hash = 0;
    for (const ch of name) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
    return TINTS[hash % TINTS.length];
  }

  const escapeHtml = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

  const WAKING_MSG = "The server is waking up or restarting (free hosting sleeps when idle). Please try again in a few seconds.";
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // Retries once or twice when the hosting proxy (not our app) returns 502/503/504,
  // which happens during cold starts and redeploys on Render's free plan.
  async function api(url, options = {}, retries = 2) {
    let res;
    try { res = await fetch(url, options); } catch {
      if (retries > 0) { await sleep(3000); return api(url, options, retries - 1); }
      throw new Error("Network error. Check your connection and try again.");
    }
    let data = null;
    try { data = await res.json(); } catch { /* non-JSON: came from the proxy, not the app */ }
    if (!data && [502, 503, 504].includes(res.status)) {
      if (retries > 0) { await sleep(4000); return api(url, options, retries - 1); }
      throw new Error(WAKING_MSG);
    }
    data = data || {};
    if (!res.ok || data.success === false) {
      const err = new Error(data.error || `Request failed (${res.status}). Please try again.`);
      err.data = data;
      err.status = res.status;
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
    toastTimer = setTimeout(() => el.toast.classList.add("hidden"), 3800);
  }

  const scrollToBottom = () => el.messages.scrollTo({ top: el.messages.scrollHeight, behavior: "smooth" });

  /* ================= safe Markdown =================
     Text is HTML-escaped FIRST; only a small set of Markdown patterns is then
     turned into tags, so raw HTML from the LLM can never execute. */
  function inline(text) {
    return text
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>")
      // [Source 1], [Source 1, Source 3], [Sources 1, 2], 【Source 1】
      .replace(/[\[【]\s*Sources?\s*([\d\s,;&and]+?)\s*[\]】]/gi, (match, list) => {
        const nums = list.match(/\d+/g);
        return nums ? nums.map((n) => `<button type="button" class="cite" data-n="${n}">${n}</button>`).join("") : match;
      });
  }

  function renderMarkdown(src) {
    const escaped = escapeHtml(src.replace(/\r\n/g, "\n"));
    const codeBlocks = [];
    const body = escaped.replace(/```[\w-]*\n?([\s\S]*?)```/g, (_, code) => {
      codeBlocks.push(`<pre><code>${code.replace(/\n$/, "")}</code></pre>`);
      return `\u0000${codeBlocks.length - 1}\u0000`;
    });
    const out = [];
    let list = null;
    let para = [];
    const flushPara = () => { if (para.length) { out.push(`<p>${inline(para.join("<br>"))}</p>`); para = []; } };
    const flushList = () => {
      if (list) { out.push(`<${list.type}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.type}>`); list = null; }
    };
    for (const line of body.split("\n")) {
      const t = line.trim();
      const code = t.match(/^\u0000(\d+)\u0000$/);
      const bullet = t.match(/^[-*•]\s+(.*)$/);
      const numbered = t.match(/^\d+[.)]\s+(.*)$/);
      const heading = t.match(/^#{1,4}\s+(.*)$/);
      if (code) { flushPara(); flushList(); out.push(codeBlocks[Number(code[1])]); }
      else if (!t) { flushPara(); flushList(); }
      else if (heading) { flushPara(); flushList(); out.push(`<h4>${inline(heading[1])}</h4>`); }
      else if (bullet || numbered) {
        flushPara();
        const type = bullet ? "ul" : "ol";
        if (!list || list.type !== type) { flushList(); list = { type, items: [] }; }
        list.items.push((bullet || numbered)[1]);
      } else if (list && /^\s{2,}/.test(line)) { list.items[list.items.length - 1] += " " + t; }
      else { flushList(); para.push(t); }
    }
    flushPara(); flushList();
    return out.join("").replace(/\u0000(\d+)\u0000/g, (_, i) => codeBlocks[Number(i)]);
  }

  /* ================= views (mobile) & evidence drawer ================= */
  function showView(view) {
    document.body.dataset.view = view;
    document.querySelectorAll(".seg").forEach((s) => {
      const on = s.dataset.view === view;
      s.classList.toggle("active", on);
      s.setAttribute("aria-selected", on);
    });
    if (view === "chat") setTimeout(scrollToBottom, 50);
  }
  document.querySelectorAll(".seg").forEach((s) => s.addEventListener("click", () => showView(s.dataset.view)));

  function openEvidence() {
    if (WIDE.matches) return; // always visible on wide screens
    document.body.classList.add("evidence-open");
    el.scrim.hidden = false;
  }
  function closeEvidence() {
    document.body.classList.remove("evidence-open");
    el.scrim.hidden = true;
  }
  el.evidenceClose.addEventListener("click", closeEvidence);
  el.scrim.addEventListener("click", closeEvidence);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeEvidence(); });

  /* ================= documents ================= */
  function renderDocuments(documents, stats) {
    el.docList.replaceChildren(...documents.map((doc) => h("li", { class: "doc", style: `--doc-tint:${tintFor(doc.filename)}` },
      h("div", { class: `doc-thumb ${doc.type === "txt" ? "txt" : ""}`, "aria-hidden": "true" }),
      h("div", { class: "doc-meta" },
        h("div", { class: "doc-name", title: doc.filename }, doc.filename),
        h("div", { class: "doc-sub" }, `${(doc.type || "pdf").toUpperCase()} · ${plural(doc.pages, (doc.unit || "Page").toLowerCase())} · ${plural(doc.chunks, "chunk")}`)),
      h("button", { class: "icon-btn", type: "button", title: `Remove ${doc.filename}`,
        "aria-label": `Remove ${doc.filename}`, onclick: () => removeDocument(doc) }, icon(ICONS.close, 15)))));

    el.emptyDocs.classList.toggle("hidden", documents.length > 0);
    el.statDocs.textContent = stats.documents;
    el.statPages.textContent = stats.pages;
    el.statChunks.textContent = stats.chunks;
    el.segCount.textContent = stats.documents;
    el.indexStatus.classList.toggle("ready", stats.index_ready);
    el.indexStatus.lastChild.textContent = stats.index_ready ? "Index ready" : "Empty";
    el.scopeLabel.textContent = stats.index_ready
      ? `Searching ${plural(stats.documents, "document")} · ${plural(stats.chunks, "chunk")}` : "No sources yet";

    state.indexReady = stats.index_ready;
    el.rebuildBtn.disabled = !stats.index_ready || state.busyUpload;
    el.clearDocsBtn.disabled = !documents.length || state.busyUpload;
    el.suggestions.classList.toggle("hidden", !stats.index_ready);
    el.welcomeUpload.classList.toggle("hidden", stats.index_ready);
    el.welcomeText.textContent = stats.index_ready
      ? `${plural(stats.documents, "document")} indexed and ready. Pick a starter question or ask your own. Every answer links to the exact passages it used.`
      : "Add PDF or TXT files to your sources. DocuRAG retrieves the most relevant passages and answers only from them, with page-level citations you can inspect.";
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
    try {
      const data = await api("/api/rebuild", { method: "POST" });
      toast(`Index rebuilt · ${plural(data.chunks, "chunk")} re-embedded`);
    } catch (err) { toast(err.message, true); }
    loadDocuments();
  });

  /* ================= upload ================= */
  const STAGES = ["Reading file...", "Extracting text...", "Creating chunks...", "Generating embeddings...", "Building semantic index..."];

  function note(kind, text) { el.uploadMessages.append(h("div", { class: `note ${kind}` }, text)); }

  function setStep(active) {
    el.steps.querySelectorAll("li").forEach((li) => {
      const i = Number(li.dataset.step);
      li.classList.toggle("done", i < active);
      li.classList.toggle("active", i === active);
    });
  }
  function setProgress(pct) {
    const v = Math.max(0, Math.min(100, Math.round(pct)));
    el.progressBar.style.width = `${v}%`;
    el.progressPct.textContent = `${v}%`;
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
        if (!e.lengthComputable) return;
        const pct = (e.loaded / e.total) * 100;
        el.progressStage.textContent = `Uploading… ${Math.round(pct)}%`;
        setProgress(pct * 0.2);
      };
      xhr.onload = () => {
        let data = {};
        try { data = JSON.parse(xhr.responseText); } catch { /* ignore */ }
        if (xhr.status >= 200 && xhr.status < 300 && data.success) resolve(data);
        else {
          const msg = data.error || ([502, 503, 504].includes(xhr.status) ? WAKING_MSG : `Upload failed (${xhr.status}). Please try again.`);
          const err = new Error(msg); err.data = data; reject(err);
        }
      };
      xhr.onerror = () => reject(new Error("Network error during upload."));
      xhr.send(formData);
    });
  }

  async function pollJob(jobId) {
    for (;;) {
      await new Promise((r) => setTimeout(r, 500));
      let job;
      try { job = await api(`/api/jobs/${jobId}`); } catch (err) {
        if (err.status === 404) throw new Error("The server restarted while processing. Please upload the file again.");
        throw err;
      }
      const stageIdx = STAGES.indexOf(job.stage);
      const perFile = 80 / job.total_files;
      const fileBase = Math.max(0, job.file_index - 1) * perFile;
      const within = stageIdx >= 0 ? ((stageIdx + 1) / STAGES.length) * perFile : 0;
      setStep(job.done ? 6 : stageIdx >= 0 ? stageIdx + 1 : 1);
      setProgress(job.done ? 100 : 20 + fileBase + within);
      el.progressStage.textContent = job.done ? "Ready" : (job.stage || "Queued").replace("...", "…");
      if (job.current_file) el.progressFile.textContent = job.total_files > 1
        ? `${job.current_file} · ${job.file_index}/${job.total_files}` : job.current_file;
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
      if (!/\.(pdf|txt)$/i.test(file.name)) { note("err", `${file.name}: only PDF and TXT files are supported.`); continue; }
      if (file.size === 0) { note("err", `${file.name}: file is empty.`); continue; }
      totalBytes += file.size;
      valid.push(file);
    }
    if (!valid.length) return;
    if (totalBytes > MAX_MB * 1024 * 1024) { note("err", `Selected files exceed the ${MAX_MB} MB limit. Upload fewer or smaller files.`); return; }

    const form = new FormData();
    valid.forEach((f) => form.append("files", f));
    setUploading(true);
    setStep(0); setProgress(0);
    el.progressFile.textContent = valid.length === 1 ? valid[0].name : `${valid.length} files`;
    el.progressStage.textContent = "Uploading…";
    try {
      const { job_id: jobId, rejected } = await uploadWithProgress(form);
      (rejected || []).forEach((r) => note("err", `${r.filename}: ${r.message}`));
      const job = await pollJob(jobId);
      for (const r of job.results) {
        if (r.status === "indexed") note("ok", `${r.filename} · ${plural(r.pages, r.filename.toLowerCase().endsWith(".txt") ? "section" : "page")}, ${plural(r.chunks, "chunk")}`);
        else if (r.status === "duplicate") note("warn", `${r.filename}: already indexed, skipped.`);
        else note("err", `${r.filename}: ${r.message}`);
      }
      const s = job.stats;
      if (s && s.index_ready) toast(`${plural(s.documents, "document")} indexed · ${plural(s.chunks, "chunk")}`);
      if (job.results.some((r) => r.status === "indexed") && MOBILE.matches) setTimeout(() => showView("chat"), 900);
    } catch (err) {
      note("err", err.message);
      (err.data?.results || []).slice(1).forEach((r) => note("err", `${r.filename}: ${r.message}`));
    } finally {
      await new Promise((r) => setTimeout(r, 400));
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
  ["dragenter", "dragover"].forEach((ev) => el.dropzone.addEventListener(ev, (e) => { e.preventDefault(); el.dropzone.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) => el.dropzone.addEventListener(ev, (e) => { e.preventDefault(); el.dropzone.classList.remove("dragover"); }));
  el.dropzone.addEventListener("drop", (e) => handleFiles(e.dataTransfer.files));
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());
  el.welcomeUpload.addEventListener("click", () => { if (MOBILE.matches) showView("library"); el.fileInput.click(); });

  /* ================= evidence ================= */
  function meter(score) {
    const pct = Math.max(4, Math.min(100, Math.round(score * 100)));
    const cls = score >= 0.5 ? "hi" : score < 0.3 ? "lo" : "";
    return h("div", { class: "ev-score" },
      h("div", { class: "ev-score-val", title: "Cosine similarity to the question" }, score > 0 ? score.toFixed(2) : "—"),
      h("div", { class: "ev-meter" }, h("i", { class: cls, style: `width:${score > 0 ? pct : 0}%` })));
  }

  function evidenceCard({ n, document: docName, page, unit = "Page", score, text, tag }) {
    const card = h("article", { class: `ev-card ${tag === "dropped" ? "unused" : ""}`, "data-n": n ?? "" },
      h("div", { class: "ev-top" },
        h("span", { class: "ev-n", style: tag === "dropped" ? null : `background:${tintFor(docName)}` }, n ?? "·"),
        h("div", { class: "ev-title" },
          h("div", { class: "ev-doc", title: docName }, docName),
          h("div", { class: "ev-page" }, `${unit} ${page}`)),
        meter(score)),
      h("div", { class: "ev-text" }, text));
    const more = h("button", { type: "button", class: "ev-more" }, "Show full passage");
    more.addEventListener("click", () => {
      const open = card.classList.toggle("expanded");
      more.textContent = open ? "Show less" : "Show full passage";
    });
    const usedLabel = state.selectedTurn?.mode === "local" ? "Used for answer" : "Sent to LLM";
    const tags = { used: usedLabel, dropped: "Below threshold", overview: "Overview context" };
    card.append(h("div", { class: "ev-foot" }, more, tag ? h("span", { class: `tag ${tag}` }, tags[tag]) : null));
    return card;
  }

  function renderEvidence(turn, focusN = null) {
    state.selectedTurn = turn;
    document.querySelectorAll(".turn-ai").forEach((t) => t.classList.toggle("selected", t === turn.node));
    const body = el.evidenceBody;
    body.replaceChildren();
    const debug = el.debugToggle.checked;
    el.evidenceSub.textContent = `For: “${turn.question.length > 60 ? turn.question.slice(0, 57) + "…" : turn.question}”`;

    const sources = turn.sources || [];
    const retrieval = turn.retrieval || [];
    body.append(h("div", { class: "ev-summary" },
      h("span", {}, h("b", {}, sources.length || turn.checked.length),
        sources.length ? (sources.length === 1 ? " source used" : " sources used") : " checked"),
      h("span", {}, "·"),
      h("span", {}, h("b", {}, retrieval.length), " retrieved")));

    if (debug && turn.searchQuery) {
      body.append(h("div", { class: "ev-section" }, "Search query"), h("div", { class: "ev-query" }, turn.searchQuery));
    }

    if (sources.length) {
      body.append(h("div", { class: "ev-section" }, "Cited passages"));
      sources.forEach((s) => body.append(evidenceCard({ n: s.number, document: s.document, page: s.page, unit: s.unit, score: s.score,
        text: s.excerpt, tag: debug ? (s.score > 0 ? "used" : "overview") : null })));
    } else if (turn.checked.length && !debug) {
      body.append(h("p", { class: "muted", style: "font-size:12.5px;padding:0 2px" },
        "These passages were checked, but they don't contain the answer."),
        h("div", { class: "ev-section" }, "Checked passages"));
      turn.checked.forEach((s) => body.append(evidenceCard({ n: s.number, document: s.document, page: s.page, unit: s.unit,
        score: s.score, text: s.excerpt, tag: "dropped" })));
    } else if (!debug) {
      body.append(h("div", { class: "evidence-empty" },
        h("p", {}, "No passage was similar enough to your question, so the LLM was not called. Try rephrasing, or turn on Retrieval debug to inspect scores.")));
    }

    if (debug) {
      const dropped = retrieval.filter((r) => !r.used);
      body.append(h("div", { class: "ev-section" }, `All retrieved chunks (${retrieval.length})`));
      if (!retrieval.length) body.append(h("div", { class: "evidence-empty" }, h("p", {}, "No chunks retrieved.")));
      retrieval.forEach((r, i) => body.append(evidenceCard({ n: i + 1, document: r.document, page: r.page, unit: r.unit, score: r.score,
        text: r.text, tag: r.used ? "used" : "dropped" })));
      if (dropped.length) body.append(h("p", { class: "muted", style: "font-size:12px;padding:0 2px" },
        `${plural(dropped.length, "chunk")} scored below the similarity threshold and were not sent to the LLM.`));
    }

    focusSource(focusN);
  }

  function focusSource(n) {
    document.querySelectorAll(".cite.on, .src-chip.on").forEach((c) => c.classList.remove("on"));
    el.evidenceBody.querySelectorAll(".ev-card.on").forEach((c) => c.classList.remove("on"));
    if (n == null || !state.selectedTurn) return;
    state.selectedTurn.node.querySelectorAll(`.cite[data-n="${n}"], .src-chip[data-n="${n}"]`).forEach((c) => c.classList.add("on"));
    const card = el.evidenceBody.querySelector(`.ev-card[data-n="${n}"]`);
    if (card) {
      card.classList.add("on");
      card.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  function showEvidence(turn, n = null) {
    renderEvidence(turn, n);
    openEvidence();
  }

  el.debugToggle.addEventListener("change", () => { if (state.selectedTurn) renderEvidence(state.selectedTurn); });

  /* ================= chat ================= */
  function updateComposer() {
    el.input.disabled = !state.indexReady;
    el.input.placeholder = state.indexReady ? "Ask anything about your documents…" : "Add a PDF or TXT file to start asking…";
    el.sendBtn.disabled = !state.indexReady || state.busyAsk || !el.input.value.trim();
    el.clearChatBtn.disabled = state.turns.length === 0 && !el.messages.querySelector(".turn-user");
  }

  function addUserTurn(text) {
    el.welcome.classList.add("hidden");
    el.messages.append(h("div", { class: "turn-user" }, h("div", { class: "q-bubble" }, text)));
    scrollToBottom();
  }

  function addPendingTurn() {
    const node = h("div", { class: "turn-ai" },
      h("div", { class: "ai-mark", "aria-hidden": "true" }, icon(ICONS.spark, 15, 2.2)),
      h("div", { class: "ai-main" },
        h("div", { class: "thinking" },
          h("div", { class: "think-line" }, h("span", { class: "spin" }), state.mode === "local"
            ? "Ranking sentences with TF-IDF and embeddings…" : "Searching your documents and composing a grounded answer…"),
          h("div", { class: "skeleton", style: "width:92%" }),
          h("div", { class: "skeleton", style: "width:78%" }),
          h("div", { class: "skeleton", style: "width:64%" }))));
    el.messages.append(node);
    scrollToBottom();
    return node;
  }

  function copyButton(text) {
    const btn = h("button", { type: "button", class: "tool" }, icon(ICONS.copy, 14), "Copy");
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(text);
        btn.replaceChildren(icon(ICONS.check, 14), "Copied");
        setTimeout(() => btn.replaceChildren(icon(ICONS.copy, 14), "Copy"), 1500);
      } catch { toast("Could not copy to clipboard", true); }
    });
    return btn;
  }

  function fillAnswer(node, question, data) {
    const turn = { node, question, mode: data.mode, sources: data.sources || [], checked: data.checked || [],
      retrieval: data.retrieval || [], searchQuery: data.search_query };
    state.turns.push(turn);
    const main = node.querySelector(".ai-main");
    const status = data.status || (data.grounded ? "grounded" : "not_found");
    const grounded = status !== "not_found";

    const local = data.mode === "local";
    const head = h("div", { class: "ai-head" },
      h("strong", {}, "DocuRAG"),
      h("span", { class: `badge ${local ? "mode-local" : "mode-llm"}`,
        title: local ? "Extractive answer built in Python: TF-IDF, stemming, embeddings, MMR/TextRank" : "Generative answer written by an LLM on Groq" },
        local ? "Local NLP" : "Groq LLM"),
      status === "grounded"
        ? h("span", { class: "badge grounded" }, icon(ICONS.check, 12, 2.6), `Grounded in ${plural(turn.sources.length, "source")}`)
        : status === "partial"
          ? h("span", { class: "badge partial" }, `Partially answered · ${plural(turn.sources.length, "source")}`)
          : h("span", { class: "badge notfound" }, "Not found in documents"));

    const answer = h("div", { class: `answer md ${grounded ? "" : "notfound"}` });
    answer.innerHTML = renderMarkdown(data.answer); // safe: escaped before formatting
    answer.addEventListener("click", (e) => {
      const cite = e.target.closest(".cite");
      if (cite) showEvidence(turn, Number(cite.dataset.n));
    });

    const parts = [head, answer];
    if (turn.sources.length) {
      parts.push(h("div", { class: "src-row" }, turn.sources.map((s) =>
        h("button", { type: "button", class: "src-chip", "data-n": s.number, title: `${s.document}, ${(s.unit || "Page").toLowerCase()} ${s.page}`,
          onclick: () => showEvidence(turn, s.number) },
          h("span", { class: "src-n", style: `background:${tintFor(s.document)};color:#fff` }, s.number),
          h("span", { class: "src-doc" }, s.document),
          h("span", { class: "src-pg" }, `${s.unit === "Section" ? "§" : "p."}${s.page}`)))));
    }
    parts.push(h("div", { class: "ai-tools" },
      copyButton(data.answer),
      h("button", { type: "button", class: "tool", onclick: () => showEvidence(turn) }, icon(ICONS.doc, 14), "View evidence")));
    main.replaceChildren(...parts);

    renderEvidence(turn); // keep the evidence panel in sync with the latest answer
  }

  function fillError(node, question, err) {
    const main = node.querySelector(".ai-main");
    const turn = { node, question, sources: [], retrieval: err.data?.retrieval || [], searchQuery: "" };
    main.replaceChildren(
      h("div", { class: "ai-head" }, h("strong", {}, "DocuRAG"), h("span", { class: "badge error" }, "Error")),
      h("div", { class: "answer error" }, err.message),
      turn.retrieval.length ? h("div", { class: "ai-tools" },
        h("button", { type: "button", class: "tool", onclick: () => { el.debugToggle.checked = true; showEvidence(turn); } },
          icon(ICONS.doc, 14), "Inspect retrieval")) : null);
  }

  async function ask(question) {
    if (!question || state.busyAsk || !state.indexReady) return;
    state.busyAsk = true;
    el.input.value = "";
    autoresize();
    addUserTurn(question);
    const pending = addPendingTurn();
    updateComposer();
    try {
      const data = await api("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, mode: state.mode, history: state.history.slice(-HISTORY_LIMIT) }),
      });
      fillAnswer(pending, question, data);
      state.history.push({ role: "user", content: question }, { role: "assistant", content: data.answer });
    } catch (err) {
      fillError(pending, question, err);
    } finally {
      state.busyAsk = false;
      updateComposer();
      scrollToBottom();
      if (!MOBILE.matches) el.input.focus();
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
    const btn = e.target.closest(".suggest");
    if (btn) ask(btn.textContent.replace(/^\W+/, "").trim());
  });
  el.clearChatBtn.addEventListener("click", () => {
    state.history = [];
    state.turns = [];
    state.selectedTurn = null;
    el.messages.querySelectorAll(".turn-user, .turn-ai").forEach((m) => m.remove());
    el.welcome.classList.remove("hidden");
    el.evidenceSub.textContent = "Passages behind the selected answer.";
    el.evidenceBody.replaceChildren(h("div", { class: "evidence-empty" },
      h("p", {}, "Ask a question to see the exact passages, pages and similarity scores used to ground the answer.")));
    closeEvidence();
    updateComposer();
  });

  /* ================= answer engine ================= */
  const HINTS = {
    llm: "Generative answer via Groq · <kbd>Enter</kbd> to send",
    local: "Extractive NLP in Python, no API · <kbd>Enter</kbd> to send",
  };
  function setMode(mode, announce = false) {
    state.mode = mode;
    localStorage.setItem("docurag-mode", mode);
    document.querySelectorAll(".mode-opt").forEach((b) => {
      const on = b.dataset.mode === mode;
      b.classList.toggle("active", on);
      b.setAttribute("aria-checked", on);
    });
    $("modeHint").innerHTML = HINTS[mode]; // static strings only
    if (announce) toast(mode === "local" ? "Local NLP: answers extracted in Python, no API call" : "Groq LLM: answers generated by the language model");
  }
  document.querySelectorAll(".mode-opt").forEach((b) => b.addEventListener("click", () => {
    if (b.dataset.mode === "llm" && !state.groqReady) { toast("GROQ_API_KEY is not set on the server. Use Local NLP.", true); return; }
    setMode(b.dataset.mode, true);
  }));
  setMode(state.mode);

  /* ================= health ================= */
  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      if (!res.ok) throw new Error("unhealthy");
      const data = await res.json();
      el.modelChip.textContent = `Groq · ${String(data.model).replace(/^openai\//, "")}`;
      el.modelChip.title = `LLM served by Groq: ${data.model} (open-weight model)`;
      el.modelChip.hidden = false;
      if (data.groq_configured) {
        el.statusPill.className = "status ok";
        el.statusText.textContent = "Online";
        el.statusPill.title = `LLM: ${data.model} on Groq · Embeddings: ${data.embedding_model} (local)`;
      } else {
        el.statusPill.className = "status warn";
        el.statusText.textContent = "Local NLP only";
        state.groqReady = false;
        setMode("local");
        el.statusPill.title = "Retrieval works, but answers need GROQ_API_KEY on the server.";
      }
    } catch {
      el.statusPill.className = "status err";
      el.statusText.textContent = "Waking up…";
      setTimeout(() => { checkHealth(); loadDocuments(); }, 5000);
    }
  }

  checkHealth();
  loadDocuments();
})();
