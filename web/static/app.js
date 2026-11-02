"use strict";

const POLL_INTERVAL_MS = 2000;
const STORAGE_KEY = "vepa-chat";

const els = {
  messages: document.getElementById("messages"),
  empty: document.getElementById("empty"),
  form: document.getElementById("ask-form"),
  input: document.getElementById("message"),
  askBtn: document.getElementById("ask-btn"),
  clearBtn: document.getElementById("clear-btn"),
  newBtn: document.getElementById("new-btn"),
  exportBtn: document.getElementById("export-btn"),
  sessionLabel: document.getElementById("session-label"),
  modelLabel: document.getElementById("model-label"),
};

// history holds {role, content, html?}; html is the server-rendered (sanitized) answer.
let state = { sessionId: "", history: [] };
let busy = false;

function loadState() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "null");
    if (saved && Array.isArray(saved.history)) state = saved;
  } catch (_) { /* storage unavailable; start fresh */ }
}

function saveState() {
  try { sessionStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch (_) { /* ignore */ }
}

function redirectIfSignedOut(response) {
  if (response.status === 401) {
    window.location.href = "/login";
    return true;
  }
  return false;
}

function addMessage(role, content, html) {
  els.empty.hidden = true;
  const wrap = document.createElement("div");
  wrap.className = `msg msg-${role}`;
  const label = document.createElement("div");
  label.className = "msg-label";
  label.textContent = role === "user" ? "You" : "Agent";
  const body = document.createElement("div");
  body.className = "msg-body";
  if (html) {
    body.innerHTML = html; // rendered server-side with raw HTML disabled
  } else {
    body.textContent = content;
  }
  wrap.append(label, body);
  els.messages.append(wrap);
  wrap.scrollIntoView({ behavior: "smooth", block: "start" });
  return body;
}

function renderAll() {
  els.messages.querySelectorAll(".msg").forEach((n) => n.remove());
  els.empty.hidden = state.history.length > 0;
  for (const m of state.history) addMessage(m.role, m.content, m.html);
  els.sessionLabel.textContent = state.sessionId ? `Session: ${state.sessionId}` : "";
}

function setBusy(value) {
  busy = value;
  els.askBtn.disabled = value;
  els.input.disabled = value;
  els.clearBtn.disabled = value;
  els.newBtn.disabled = value;
  els.askBtn.textContent = value ? "Working…" : "Ask Agent";
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Error pages from the server or the tunnel are not JSON; turn them into a readable error.
async function readJson(res) {
  try {
    return await res.json();
  } catch (_) {
    return { error: `Server error (${res.status}). Please try again.` };
  }
}

async function pollJob(jobId, statusEl) {
  const started = Date.now();
  for (;;) {
    const secs = Math.round((Date.now() - started) / 1000);
    statusEl.textContent = `Searching evidence and writing an answer… ${secs}s`;
    await sleep(POLL_INTERVAL_MS);
    let res;
    try {
      res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, { cache: "no-store" });
    } catch (_) {
      continue; // transient network error; keep polling
    }
    if (redirectIfSignedOut(res)) throw new Error("Signed out.");
    const data = await readJson(res);
    if (!res.ok) throw new Error(data.error || `Request failed (${res.status}).`);
    if (data.status === "done") return data;
    if (data.status === "error") throw new Error(data.error || "The agent failed.");
  }
}

async function ask(message) {
  setBusy(true);
  state.history.push({ role: "user", content: message });
  saveState();
  addMessage("user", message);
  const statusEl = addMessage("assistant", "");
  statusEl.classList.add("pending");

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: state.sessionId }),
    });
    if (redirectIfSignedOut(res)) return;
    const data = await readJson(res);
    if (!res.ok) throw new Error(data.error || `Request failed (${res.status}).`);
    state.sessionId = data.session_id;
    els.sessionLabel.textContent = `Session: ${state.sessionId}`;

    const result = await pollJob(data.job_id, statusEl);
    statusEl.classList.remove("pending");
    statusEl.innerHTML = result.answer_html;
    state.history.push({ role: "assistant", content: result.answer, html: result.answer_html });
  } catch (err) {
    statusEl.classList.remove("pending");
    statusEl.classList.add("failed");
    statusEl.textContent = `Error: ${err.message}`;
    state.history.push({ role: "assistant", content: `Error: ${err.message}` });
  } finally {
    saveState();
    setBusy(false);
    els.input.focus();
  }
}

async function exportPdf() {
  if (state.history.length === 0) {
    alert("There is no conversation to export yet.");
    return;
  }
  els.exportBtn.disabled = true;
  try {
    const res = await fetch("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        history: state.history.map(({ role, content }) => ({ role, content })),
      }),
    });
    if (redirectIfSignedOut(res)) return;
    if (!res.ok) throw new Error(`Export failed (${res.status}).`);
    const blob = await res.blob();
    const disposition = res.headers.get("Content-Disposition") || "";
    const match = disposition.match(/filename="?([^";]+)"?/);
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = match ? match[1] : "conversation.pdf";
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  } catch (err) {
    alert(err.message);
  } finally {
    els.exportBtn.disabled = false;
  }
}

els.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = els.input.value.trim();
  if (!message || busy) return;
  els.input.value = "";
  ask(message);
});

els.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    els.form.requestSubmit();
  }
});

els.clearBtn.addEventListener("click", () => {
  state = { sessionId: "", history: [] };
  saveState();
  renderAll();
});

els.newBtn.addEventListener("click", () => {
  // Keep the visible conversation; the next question starts a fresh agent session.
  state.sessionId = "";
  saveState();
  els.sessionLabel.textContent = "New session starts with your next question";
});

els.exportBtn.addEventListener("click", exportPdf);

fetch("/api/info", { cache: "no-store" })
  .then((res) => (redirectIfSignedOut(res) ? null : res.json()))
  .then((data) => { if (data && data.model_label) els.modelLabel.textContent = data.model_label; })
  .catch(() => {});

loadState();
renderAll();
