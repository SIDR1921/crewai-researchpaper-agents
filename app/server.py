#!/usr/bin/env python
"""Local web app for the literature-review assistant.

    python app/server.py            # then open http://127.0.0.1:8000

Deliberately built on the standard library: no framework, no build step, nothing
to install beyond what the pipeline already needs. It is a local tool bound to
the loopback interface, and it reads the project's own artefacts through
:mod:`litreview.service`, so every number it shows is the same number the report
quotes.

Writes are limited to starting pipeline commands that the project already
exposes as scripts, and only those on a fixed allow-list.
"""
from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from litreview import service as S           # noqa: E402
from litreview.config import paths           # noqa: E402

WEB = Path(__file__).resolve().parent / "web"
PY = str(REPO / ".venv" / "bin" / "python") if (REPO / ".venv" / "bin" / "python").exists() \
    else sys.executable

#: Only these may be started from the browser, and only with validated arguments.
COMMANDS: dict[str, dict] = {
    "flow": {
        "label": "Run the full Flow on a topic",
        "argv": lambda a: [PY, str(REPO / "scripts" / "run_flow.py"),
                           "--topic", a["topic"], "--n-papers", str(int(a.get("n_papers", 12))),
                           "--auto-approve"],
        "fields": [{"name": "topic", "label": "Topic", "type": "text",
                    "placeholder": "faithfulness evaluation in abstractive summarisation"},
                   {"name": "n_papers", "label": "Papers", "type": "number", "value": 12}],
    },
    "demo": {
        "label": "Demo: regenerate one benchmark paper's related work",
        "argv": lambda a: [PY, str(REPO / "scripts" / "run_demo.py"),
                           "--split", "test", "--max-cited",
                           str(int(a.get("max_cited", 12)))]
                          + (["--arxiv-id", a["arxiv_id"]] if a.get("arxiv_id") else []),
        "fields": [{"name": "arxiv_id", "label": "arXiv id", "type": "text",
                    "placeholder": "blank = the smallest test example"},
                   {"name": "max_cited", "label": "Max cited", "type": "number", "value": 12}],
    },
    "extract": {
        "label": "Extract and verify claims",
        "argv": lambda a: [PY, str(REPO / "scripts" / "run_stage3.py"), "extract",
                           "--n", str(int(a.get("n", 20)))],
        "fields": [{"name": "n", "label": "Papers", "type": "number", "value": 20}],
    },
    "figures": {
        "label": "Regenerate figures",
        "argv": lambda a: [PY, str(REPO / "scripts" / "make_figures.py")],
        "fields": [],
    },
    "report": {
        "label": "Rebuild the report",
        "argv": lambda a: [PY, str(REPO / "scripts" / "build_report.py")],
        "fields": [],
    },
}

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _start(cmd: str, args: dict) -> dict:
    spec = COMMANDS[cmd]
    argv = spec["argv"](args)
    job_id = uuid.uuid4().hex[:8]
    log = paths().results / "_jobs" / f"{cmd}_{job_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("w")
    proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO),
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})
    with _LOCK:
        _JOBS[job_id] = {"id": job_id, "cmd": cmd, "started": time.time(),
                         "log": str(log), "proc": proc, "argv": argv[1:]}
    return {"id": job_id, "cmd": cmd, "label": spec["label"]}


def _job_state(job_id: str) -> dict:
    with _LOCK:
        job = _JOBS.get(job_id)
    if not job:
        return {"error": "unknown job"}
    proc = job["proc"]
    code = proc.poll()
    text = ""
    p = Path(job["log"])
    if p.exists():
        raw = p.read_text(errors="ignore")
        lines = [l for l in raw.replace("\r", "\n").split("\n") if l.strip()]
        text = "\n".join(lines[-200:])
    return {"id": job_id, "cmd": job["cmd"], "running": code is None,
            "exit_code": code, "elapsed": round(time.time() - job["started"], 1),
            "log": text}


ROUTES = {
    "/api/status": lambda q: S.pipeline_status(),
    "/api/dataset": lambda q: S.dataset_summary(),
    "/api/benchmark": lambda q: S.benchmark_examples(
        limit=int(q.get("limit", ["24"])[0]), offset=int(q.get("offset", ["0"])[0]),
        split=(q.get("split", [None])[0] or None)),
    "/api/example": lambda q: S.benchmark_example(q.get("id", [""])[0]) or {"error": "not found"},
    "/api/labels": lambda q: S.label_summary(),
    "/api/review": lambda q: S.review(q.get("run", [None])[0]) or {"empty": True},
    "/api/runs": lambda q: {"runs": S.runs()},
    "/api/usage": lambda q: S.usage(),
    "/api/references": lambda q: S.references(),
    "/api/report": lambda q: S.report_status(),
    "/api/commands": lambda q: {"commands": [
        {"id": k, "label": v["label"], "fields": v["fields"]} for k, v in COMMANDS.items()]},
    "/api/job": lambda q: _job_state(q.get("id", [""])[0]),
}


class Handler(BaseHTTPRequestHandler):
    server_version = "litreview"

    def log_message(self, fmt, *a):             # quiet: the console is for the pipeline
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, default=str).encode(), "application/json")

    def do_GET(self) -> None:                    # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ROUTES:
            try:
                return self._json(ROUTES[u.path](q))
            except Exception as e:               # a broken panel must not kill the app
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        if u.path == "/api/graph":
            return self._graph(q)
        return self._static(u.path)

    def do_POST(self) -> None:                   # noqa: N802
        u = urlparse(self.path)
        if u.path != "/api/run":
            return self._json({"error": "not found"}, 404)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad JSON"}, 400)
        cmd = body.get("cmd")
        if cmd not in COMMANDS:
            return self._json({"error": "unknown command"}, 400)
        args = body.get("args") or {}
        if cmd == "flow" and not (args.get("topic") or "").strip():
            return self._json({"error": "a topic is required"}, 400)
        try:
            return self._json(_start(cmd, args))
        except Exception as e:
            return self._json({"error": str(e)}, 500)

    # ---------------------------------------------------------------- helpers

    def _graph(self, q) -> None:
        run = q.get("run", [None])[0]
        dirs = sorted((paths().results / "flow_runs").glob("*")) \
            if (paths().results / "flow_runs").exists() else []
        d = next((x for x in dirs if x.name == run), dirs[-1] if dirs else None)
        p = (d / "citation_graph.html") if d else None
        if not p or not p.exists():
            return self._send(404, b"<p style='font:14px system-ui;padding:20px;color:#8aa'>"
                                   b"No citation graph yet. Run the Flow first.</p>",
                              "text/html; charset=utf-8")
        return self._send(200, p.read_bytes(), "text/html; charset=utf-8")

    def _static(self, path: str) -> None:
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (WEB / rel).resolve()
        if not str(target).startswith(str(WEB.resolve())) or not target.is_file():
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype == "application/javascript":
            ctype += "; charset=utf-8"
        body = target.read_bytes()
        if rel == "index.html":
            # Stamp the asset links with their modification time. Browsers cache
            # these hard even against no-store, and a stale stylesheet in a tool
            # you are actively editing is a waste of everyone's afternoon.
            text = body.decode()
            for asset in ("app.css", "app.js"):
                f = WEB / asset
                if f.exists():
                    text = text.replace(f"/{asset}", f"/{asset}?v={int(f.stat().st_mtime)}")
            body = text.encode()
        return self._send(200, body, ctype)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"litreview console on {url}   (project root: {paths().root})")
    print("Ctrl-C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
