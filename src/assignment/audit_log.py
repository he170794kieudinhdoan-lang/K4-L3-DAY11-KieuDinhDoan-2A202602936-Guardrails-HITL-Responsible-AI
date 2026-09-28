"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._pending: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input + start timestamp keyed by request_id."""
        rid = request_id or f"{user_id}:{len(self.logs) + len(self._pending)}"
        self._open[rid] = time.perf_counter()
        self._pending[rid] = {
            "request_id": rid,
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
        }
        return rid

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision, latency; append to self.logs."""
        rid = request_id or user_id
        pending = self._pending.pop(rid, {})
        started = self._open.pop(rid, None)
        latency_ms = None
        if started is not None:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
        self.logs.append({
            "request_id": rid,
            "user_id": pending.get("user_id", user_id),
            "input": pending.get("input", ""),
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "started_at": pending.get("started_at"),
            "finished_at": utc_now_iso(),
            "latency_ms": latency_ms,
        })

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
