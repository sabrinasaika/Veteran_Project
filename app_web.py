#!/usr/bin/env python3
"""
Veterans Employment Pathways Agent — Web UI
-------------------------------------------
FastAPI server for the chat interface, powered by the Agno agent (pathways_agent.py).

The server listens on localhost only; run_public.sh exposes it through a
Cloudflare tunnel. Questions run as background jobs that the page polls, so no
single HTTP request stays open long enough to hit proxy timeouts.
"""

from __future__ import annotations

import html
import os
import re
import secrets
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from markdown_it import MarkdownIt
from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from pathways_agent import GROQ_MODEL, OLLAMA_MODEL, OPENAI_MODEL, _active_provider_name, run_agent


# ── Configuration ─────────────────────────────────────────────────────────────

APP_USERNAME = os.getenv("APP_USERNAME", "").strip()
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()
APP_PORT = int(os.getenv("APP_PORT", "7860"))

WEB_DIR = Path(__file__).parent / "web"
EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)

SESSION_COOKIE = "vepa_auth"
AUTH_TTL_SECONDS = 12 * 60 * 60
JOB_TTL_SECONDS = 60 * 60
MAX_MESSAGE_CHARS = 4000
MAX_LOGIN_FAILURES = 5
LOGIN_FAILURE_WINDOW_SECONDS = 10 * 60
SESSION_ID_RE = re.compile(r"^session-[0-9a-f]{8}$")

CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)

if _active_provider_name() == "ollama":
    MODEL_LABEL = f"Local {OLLAMA_MODEL}"
elif _active_provider_name() == "openai":
    MODEL_LABEL = f"OpenAI {OPENAI_MODEL}"
else:
    MODEL_LABEL = f"Groq {GROQ_MODEL.split('/')[-1]}"


# ── Markdown rendering ────────────────────────────────────────────────────────
# Raw HTML is disabled so model or tool output cannot inject markup or scripts;
# markdown-it also rejects javascript:/data: link targets by default.

_markdown = MarkdownIt("commonmark", {"html": False, "breaks": True}).enable("table")


def _render_link_open(self, tokens, idx, options, env):
    tokens[idx].attrSet("target", "_blank")
    tokens[idx].attrSet("rel", "noopener noreferrer")
    return self.renderToken(tokens, idx, options, env)


_markdown.add_render_rule("link_open", _render_link_open)


_ESCAPED_BR_RE = re.compile(r"&lt;br\s*/?&gt;", re.IGNORECASE)


def render_markdown(text: str) -> str:
    try:
        # Raw HTML is escaped, but the model often uses <br> for line breaks inside
        # table cells; restore just that one tag.
        return _ESCAPED_BR_RE.sub("<br>", _markdown.render(text or ""))
    except Exception:
        # Never lose a finished answer to a formatting bug; show it as escaped plain text.
        return f"<pre>{html.escape(text or '')}</pre>"


# ── Authentication ────────────────────────────────────────────────────────────

_auth_tokens: dict[str, float] = {}
_login_failures: dict[str, list[float]] = {}
_auth_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "unknown")


def _is_authenticated(request: Request) -> bool:
    token = request.cookies.get(SESSION_COOKIE, "")
    if not token:
        return False
    with _auth_lock:
        expires = _auth_tokens.get(token)
        if expires is None:
            return False
        if expires < time.time():
            _auth_tokens.pop(token, None)
            return False
    return True


def _require_auth(request: Request) -> str:
    if not _is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not signed in.")
    return request.cookies[SESSION_COOKIE]


def _too_many_failures(ip: str) -> bool:
    cutoff = time.time() - LOGIN_FAILURE_WINDOW_SECONDS
    with _auth_lock:
        recent = [t for t in _login_failures.get(ip, []) if t > cutoff]
        _login_failures[ip] = recent
        return len(recent) >= MAX_LOGIN_FAILURES


def _record_failure(ip: str) -> None:
    with _auth_lock:
        _login_failures.setdefault(ip, []).append(time.time())


# ── Background jobs ───────────────────────────────────────────────────────────

@dataclass
class Job:
    owner: str
    session_id: str
    status: str = "running"
    answer: str = ""
    error: str = ""
    created: float = field(default_factory=time.time)


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=4)


def _purge_old_jobs() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    with _jobs_lock:
        for job_id in [k for k, j in _jobs.items() if j.created < cutoff]:
            _jobs.pop(job_id, None)


def _run_job(job_id: str, message: str, session_id: str) -> None:
    try:
        answer = run_agent(message, session_id=session_id)
        with _jobs_lock:
            _jobs[job_id].answer = str(answer or "No response from agent.")
            _jobs[job_id].status = "done"
    except Exception as exc:  # surface agent failures to the page instead of hanging
        with _jobs_lock:
            _jobs[job_id].error = f"{type(exc).__name__}: {exc}"
            _jobs[job_id].status = "error"


def _new_session_id() -> str:
    return f"session-{uuid.uuid4().hex[:8]}"


# ── PDF export ────────────────────────────────────────────────────────────────

def _wrap_pdf_text(text: str, font_name: str, font_size: int, max_width: float) -> list[str]:
    wrapped: list[str] = []
    for paragraph in (text or "").splitlines():
        paragraph = paragraph.rstrip()
        if not paragraph:
            wrapped.append("")
            continue
        words = paragraph.split()
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            if stringWidth(trial, font_name, font_size) <= max_width:
                current = trial
            else:
                wrapped.append(current)
                current = word
        wrapped.append(current)
    return wrapped


def export_chat(history: list[dict], session_id: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_session = session_id if SESSION_ID_RE.match(session_id or "") else "session"
    out_path = EXPORT_DIR / f"{safe_session}-{stamp}.pdf"

    pdf = canvas.Canvas(str(out_path), pagesize=letter)
    width, height = letter
    left = 54
    right = width - 54
    top = height - 54
    bottom = 54
    line_height = 14

    def new_page() -> float:
        pdf.showPage()
        return top

    y = top
    pdf.setTitle("Veterans Employment Pathways Agent Conversation")
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(left, y, "Veterans Employment Pathways Agent Conversation")
    y -= 24

    pdf.setFont("Helvetica", 10)
    meta_lines = [
        f"Session ID: {session_id or 'unspecified'}",
        f"Exported: {datetime.now().isoformat(timespec='seconds')}",
    ]
    for line in meta_lines:
        pdf.drawString(left, y, line)
        y -= line_height
    y -= 8

    for item in history:
        role = str(item.get("role") or "unknown").strip().lower()
        label = "User" if role == "user" else "Assistant"
        content = str(item.get("content") or "").strip() or "No content."

        if y < bottom + 48:
            y = new_page()

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(left, y, label)
        y -= 18

        pdf.setFont("Helvetica", 10)
        for line in _wrap_pdf_text(content, "Helvetica", 10, right - left):
            if y < bottom + 20:
                y = new_page()
                pdf.setFont("Helvetica", 10)
            pdf.drawString(left, y, line)
            y -= line_height
        y -= 10

    pdf.save()
    return out_path


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = CSP
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


class AskRequest(BaseModel):
    message: str = Field(max_length=MAX_MESSAGE_CHARS)
    session_id: str = ""


class ChatMessage(BaseModel):
    role: str
    content: str


class ExportRequest(BaseModel):
    session_id: str = ""
    history: list[ChatMessage] = Field(max_length=200)


@app.get("/")
def index(request: Request):
    if not _is_authenticated(request):
        return RedirectResponse("/login", status_code=303)
    return FileResponse(WEB_DIR / "index.html")


@app.get("/login")
def login_page(request: Request):
    if _is_authenticated(request):
        return RedirectResponse("/", status_code=303)
    return FileResponse(WEB_DIR / "login.html")


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    ip = _client_ip(request)
    if _too_many_failures(ip):
        return RedirectResponse("/login?error=locked", status_code=303)

    valid_user = secrets.compare_digest(username.encode(), APP_USERNAME.encode())
    valid_pass = secrets.compare_digest(password.encode(), APP_PASSWORD.encode())
    if not (valid_user and valid_pass):
        _record_failure(ip)
        return RedirectResponse("/login?error=invalid", status_code=303)

    token = secrets.token_urlsafe(32)
    with _auth_lock:
        _auth_tokens[token] = time.time() + AUTH_TTL_SECONDS
        _login_failures.pop(ip, None)

    response = RedirectResponse("/", status_code=303)
    secure = request.headers.get("x-forwarded-proto") == "https" or request.url.scheme == "https"
    response.set_cookie(
        SESSION_COOKIE, token, max_age=AUTH_TTL_SECONDS,
        httponly=True, samesite="strict", secure=secure,
    )
    return response


@app.post("/logout")
def logout(request: Request):
    token = request.cookies.get(SESSION_COOKIE, "")
    with _auth_lock:
        _auth_tokens.pop(token, None)
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE)
    return response


@app.get("/api/info")
def info(request: Request):
    _require_auth(request)
    return {"model_label": MODEL_LABEL}


@app.post("/api/ask")
def ask(request: Request, body: AskRequest):
    owner = _require_auth(request)
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty.")
    session_id = body.session_id if SESSION_ID_RE.match(body.session_id) else _new_session_id()

    _purge_old_jobs()
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = Job(owner=owner, session_id=session_id)
    _executor.submit(_run_job, job_id, message, session_id)
    return {"job_id": job_id, "session_id": session_id}


@app.get("/api/jobs/{job_id}")
def job_status(request: Request, job_id: str):
    owner = _require_auth(request)
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None or job.owner != owner:
            raise HTTPException(status_code=404, detail="Job not found.")
        status, answer, error = job.status, job.answer, job.error

    if status == "running":
        return {"status": "running"}
    if status == "error":
        return {"status": "error", "error": error}
    return {"status": "done", "answer": answer, "answer_html": render_markdown(answer)}


@app.post("/api/export")
def export(request: Request, body: ExportRequest):
    _require_auth(request)
    if not body.history:
        raise HTTPException(status_code=400, detail="Nothing to export.")
    out_path = export_chat([m.model_dump() for m in body.history], body.session_id)
    return FileResponse(out_path, media_type="application/pdf", filename=out_path.name)


@app.exception_handler(HTTPException)
def http_error(request: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")


if __name__ == "__main__":
    # The app is exposed publicly through a Cloudflare tunnel, so refuse to start without a login.
    if not APP_USERNAME or not APP_PASSWORD:
        sys.exit("APP_USERNAME and APP_PASSWORD must be set in .env before launching the app.")

    # Listen on localhost only; cloudflared forwards public traffic to this port.
    print(f"Running on local URL:  http://127.0.0.1:{APP_PORT}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=APP_PORT, server_header=False, log_level="warning")
