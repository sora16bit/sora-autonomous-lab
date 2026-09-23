"""Optional Jev decision adapter.

Jev is deliberately advisory in this project.  This module sends only a
compact, non-secret snapshot, records usage without recording credentials,
and returns ``None`` for every failure so the local policy remains authoritative.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JevDecision:
    response: dict[str, Any]
    input_tokens: int
    latency_ms: int
    request_hash: str


class JevController:
    """Small fail-open adapter for the hosted Jev API."""

    def __init__(self, store: Any, *, endpoint: str | None = None,
                 model: str | None = None, timeout: float | None = None) -> None:
        self.store = store
        self.endpoint = endpoint or os.environ.get("JEV_API_URL", "https://api.typesafe.ai/v1/systemone")
        self.model = model or os.environ.get("JEV_MODEL", "jev-latest")
        self.api_key = os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
        self.timeout = float(timeout or os.environ.get("JEV_TIMEOUT_SECONDS", "3"))
        self.daily_limit = int(os.environ.get("JEV_DAILY_INPUT_TOKEN_LIMIT", "1250000"))

    @property
    def available(self) -> bool:
        # The key itself is never returned or logged.
        return bool(self.api_key and self.model and self.endpoint)

    def _reserve_budget(self, tokens: int) -> bool:
        now = time.time()
        usage = self.store.get_meta("jev_usage", {}) or {}
        if not isinstance(usage, dict) or now - float(usage.get("window_start", 0)) >= 86400:
            usage = {"window_start": now, "input_tokens": 0, "requests": 0}
        used = int(usage.get("input_tokens", 0))
        if used + tokens > self.daily_limit:
            return False
        usage.update(input_tokens=used + tokens, requests=int(usage.get("requests", 0)) + 1)
        self.store.set_meta("jev_usage", usage)
        return True

    @staticmethod
    def _payload(snapshot: dict[str, Any], model: str) -> dict[str, Any]:
        return {
            "model": model,
            "state": snapshot,
            "questions": {
                "next_focus": {
                    "type": "choice",
                    "instructions": "Choose the next marginal compute focus from the supplied state.",
                    "criteria": {
                        "continue_search": "Current exploration remains productive.",
                        "diversify_search": "Search is concentrated or stale and broader exploration is useful.",
                        "evolve_attack": "Attack discovery is the most informative next use of compute.",
                        "defense_trial": "Actionable attack evidence supports a bounded defense trial.",
                        "deepen_audit": "Evidence is weak or conflicting and needs independent audit.",
                        "local_advisor": "A local hypothesis is useful but frontier escalation is unnecessary.",
                        "astra": "A consequential design question warrants expensive reasoning.",
                    },
                },
                "plateau": {"type": "noul", "instructions": "Does useful progress appear materially stalled?"},
                "audit_uncertain": {"type": "noul", "instructions": "Is the audit evidence weak or conflicting?"},
                "astra_worthy": {"type": "noul", "instructions": "Would expensive external reasoning add value now?"},
            },
        }

    @staticmethod
    def _response_diagnostic(raw: bytes, content_type: str | None) -> dict[str, Any]:
        """Return response-shape metadata only; never persist response contents."""
        diagnostic: dict[str, Any] = {
            "content_type": (content_type or "")[:120],
            "body_bytes": len(raw),
            "body_sha256": hashlib.sha256(raw).hexdigest(),
        }
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            diagnostic["json_valid"] = False
            return diagnostic

        diagnostic["json_valid"] = True
        diagnostic["root_type"] = type(parsed).__name__
        if isinstance(parsed, dict):
            keys = sorted(str(key) for key in parsed.keys())
            diagnostic["top_level_keys"] = keys[:32]
            answers = parsed.get("answers")
            diagnostic["answers_type"] = type(answers).__name__ if "answers" in parsed else "missing"
            if isinstance(answers, dict):
                diagnostic["answer_keys"] = sorted(str(key) for key in answers.keys())[:32]
            diagnostic["model_type"] = type(parsed.get("model")).__name__ if "model" in parsed else "missing"
            usage = parsed.get("usage")
            diagnostic["usage_type"] = type(usage).__name__ if "usage" in parsed else "missing"
            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens")
                if isinstance(input_tokens, int) and input_tokens >= 0:
                    diagnostic["reported_input_tokens"] = input_tokens
            # Preserve only validation locations and machine codes. Server error
            # messages can echo submitted values, so do not persist them.
            detail = parsed.get("detail")
            if isinstance(detail, list):
                diagnostic["validation_errors"] = [
                    {
                        "loc": item.get("loc") if isinstance(item, dict) else None,
                        "type": item.get("type") if isinstance(item, dict) else None,
                    }
                    for item in detail[:8]
                ]
        return diagnostic

    def decide(self, snapshot: dict[str, Any]) -> JevDecision | None:
        if not self.available:
            return None
        payload = self._payload(snapshot, self.model)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        estimated_tokens = max(1, len(encoded) // 4)
        digest = hashlib.sha256(encoded).hexdigest()
        if not self._reserve_budget(estimated_tokens):
            self.store.event("jev_fallback", {"reason": "daily_input_budget", "request_hash": digest,
                                                "estimated_tokens": estimated_tokens})
            return None
        request = urllib.request.Request(
            self.endpoint,
            data=encoded,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        response_headers: Any = None
        try:
            raw = b""
            for attempt in range(2):
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout) as response:
                        raw = response.read()
                        response_headers = response.headers
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code not in (429, 529) or attempt:
                        raise
                    retry_after = exc.headers.get("Retry-After", "1")
                    try:
                        delay = min(2.0, max(0.1, float(retry_after)))
                    except (TypeError, ValueError):
                        delay = 1.0
                    time.sleep(delay)
            parsed = json.loads(raw.decode("utf-8"))
            if not isinstance(parsed, dict) or not isinstance(parsed.get("answers"), dict):
                diagnostic = self._response_diagnostic(
                    raw, response_headers.get("Content-Type") if response_headers else None
                )
                self.store.event("jev_fallback", {
                    "reason": "unexpected_response_shape",
                    "request_hash": digest,
                    "estimated_tokens": estimated_tokens,
                    "http_status": 200,
                    "diagnostic": diagnostic,
                })
                return None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            diagnostic: dict[str, Any] = {"error_class": type(exc).__name__}
            if isinstance(exc, urllib.error.HTTPError):
                diagnostic["http_status"] = int(exc.code)
                diagnostic.update(self._response_diagnostic(
                    exc.read(), exc.headers.get("Content-Type") if exc.headers else None
                ))
            elif raw:
                diagnostic.update(self._response_diagnostic(
                    raw, response_headers.get("Content-Type") if response_headers else None
                ))
            self.store.event("jev_fallback", {"reason": type(exc).__name__, "request_hash": digest,
                                                "estimated_tokens": estimated_tokens,
                                                "diagnostic": diagnostic})
            return None
        latency_ms = round((time.monotonic() - started) * 1000)
        usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else {}
        actual_tokens = int(usage.get("input_tokens", estimated_tokens) or estimated_tokens)
        return JevDecision(response=parsed, input_tokens=actual_tokens,
                           latency_ms=latency_ms, request_hash=digest)


def build_report(store: Any, *, since_hours: float | None = None) -> dict[str, Any]:
    """Summarize Jev shadow traffic without exposing credentials or raw state."""
    if since_hours is not None and since_hours < 0:
        raise ValueError("since_hours must be non-negative")
    cutoff = None if since_hours is None else time.time() - since_hours * 3600.0
    query = "SELECT created_at, kind, payload_json FROM events WHERE kind LIKE 'jev_%'"
    params: tuple[Any, ...] = ()
    if cutoff is not None:
        query += " AND created_at >= ?"
        params = (cutoff,)
    query += " ORDER BY event_id"
    rows = store.db.execute(query, params).fetchall()
    counts: dict[str, int] = {}
    fallback_reasons: dict[str, int] = {}
    choices: dict[str, int] = {}
    latencies: list[int] = []
    tokens = 0
    cost = 0.0
    diagnostic_events: list[dict[str, Any]] = []
    for row in rows:
        kind = str(row["kind"])
        counts[kind] = counts.get(kind, 0) + 1
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, json.JSONDecodeError):
            payload = {}
        if kind == "jev_fallback":
            reason = str(payload.get("reason", "unknown"))
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
            diagnostic = payload.get("diagnostic")
            estimated = payload.get("estimated_tokens")
            try:
                failed_tokens = int(estimated)
                if failed_tokens >= 0:
                    tokens += failed_tokens
                    cost += failed_tokens * 0.042 / 1_000_000
            except (TypeError, ValueError):
                pass
            if isinstance(diagnostic, dict):
                reported = diagnostic.get("reported_input_tokens")
                try:
                    reported_tokens = int(reported)
                    if reported_tokens >= 0 and isinstance(estimated, int):
                        tokens += reported_tokens - int(estimated)
                        cost += (reported_tokens - int(estimated)) * 0.042 / 1_000_000
                except (TypeError, ValueError):
                    pass
                diagnostic_events.append({
                    "created_at": row["created_at"],
                    "reason": reason,
                    "http_status": payload.get("http_status", diagnostic.get("http_status")),
                    "response": diagnostic,
                })
        elif kind == "jev_decision_shadow":
            try:
                latency = int(payload.get("latency_ms"))
                if latency >= 0:
                    latencies.append(latency)
            except (TypeError, ValueError):
                pass
            try:
                request_tokens = int(payload.get("input_tokens"))
                if request_tokens >= 0:
                    tokens += request_tokens
            except (TypeError, ValueError):
                pass
            try:
                cost += max(0.0, float(payload.get("estimated_cost_usd", 0.0)))
            except (TypeError, ValueError):
                pass
            response = payload.get("response") if isinstance(payload, dict) else {}
            answers = response.get("answers", {}) if isinstance(response, dict) else {}
            next_focus = answers.get("next_focus", {}) if isinstance(answers, dict) else {}
            choice = next_focus.get("choice") if isinstance(next_focus, dict) else None
            if choice is not None:
                key = str(choice)
                choices[key] = choices.get(key, 0) + 1
    latencies.sort()
    def percentile(values: list[int], fraction: float) -> int | None:
        if not values:
            return None
        index = min(len(values) - 1, int(round((len(values) - 1) * fraction)))
        return values[index]
    attempts = counts.get("jev_decision_shadow", 0) + counts.get("jev_fallback", 0)
    return {
        "configured": JevController(store).available,
        "window_hours": since_hours,
        "counts": counts,
        "attempts": attempts,
        "valid_response_rate": (counts.get("jev_decision_shadow", 0) / attempts if attempts else None),
        "fallback_reasons": fallback_reasons,
        "recent_failure_diagnostics": diagnostic_events[-5:],
        "next_focus_distribution": choices,
        "latency_ms": {"count": len(latencies), "p50": percentile(latencies, 0.50), "p95": percentile(latencies, 0.95)},
        "input_tokens": tokens,
        "estimated_cost_usd": round(cost, 8),
        "remaining_daily_input_tokens": max(0, JevController(store).daily_limit - int((store.get_meta("jev_usage", {}) or {}).get("input_tokens", 0))),
        "mode": "shadow",
        "execution_effect": "none",
    }
