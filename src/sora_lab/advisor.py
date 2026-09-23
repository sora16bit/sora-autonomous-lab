from __future__ import annotations

import json
import os
import subprocess
import time
import shlex
from dataclasses import dataclass
from typing import Any, Callable

from .store import LabStore


@dataclass(frozen=True)
class AdvicePolicy:
    interval_seconds: int = 5 * 60 * 60
    max_input_tokens: int = 2000
    max_output_tokens: int = 500
    provider: str = "astra"
    model: str = "astra-low"


def window_id(accumulated_seconds: float, policy: AdvicePolicy) -> str:
    return str(int(accumulated_seconds // policy.interval_seconds))


def should_consult(store: LabStore, *, accumulated_seconds: float, policy: AdvicePolicy, needs_decision: bool) -> bool:
    if not needs_decision or accumulated_seconds < policy.interval_seconds:
        return False
    wid = window_id(accumulated_seconds, policy)
    return not store.advice_exists(wid, policy.provider, policy.model)


def build_packet(summary: dict[str, Any], *, max_chars: int = 8000) -> dict[str, Any]:
    """Build a shareable report; callers must pass aggregate results only."""
    packet = {
        "purpose": "短い研究方針判断。コード変更や実験の直接実行は求めない。",
        "summary": summary,
        "allowed_response": ["decision", "reason", "next_experiment", "uncertainty"],
    }
    encoded = json.dumps(packet, ensure_ascii=False, sort_keys=True)
    if len(encoded) > max_chars:
        packet["summary"] = {"truncated": True, "note": "report exceeded local character budget"}
    return packet


def record_skipped(store: LabStore, *, accumulated_seconds: float, policy: AdvicePolicy, reason: str) -> str:
    wid = window_id(accumulated_seconds, policy)
    return store.add_advice(window_id=wid, provider=policy.provider, model=policy.model, request={"reason": reason}, response=None, status="skipped")


def consult_command(store: LabStore, *, accumulated_seconds: float, policy: AdvicePolicy, packet: dict[str, Any], command: list[str] | None = None, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run, incident_id: str | None = None) -> dict[str, Any]:
    """Optional adapter for a preconfigured cloud CLI; disabled unless explicitly supplied."""
    if incident_id is None and not should_consult(store, accumulated_seconds=accumulated_seconds, policy=policy, needs_decision=True):
        return {"status": "not_due"}
    wid = incident_id or window_id(accumulated_seconds, policy)
    # Reserve before any external call.  A timeout or crashed caller therefore
    # cannot send the same paid window again automatically.
    if not store.reserve_advice(window_id=wid, provider=policy.provider, model=policy.model, request=packet):
        return {"status": "already_reserved", "window_id": wid}
    command = command or (shlex.split(os.environ["SORA_ASTRA_COMMAND"]) if os.environ.get("SORA_ASTRA_COMMAND") else None)
    if not command:
        store.finish_advice(window_id=wid, provider=policy.provider, model=policy.model, response=None, status="manual_required")
        return {"status": "manual_required", "window_id": wid, "packet": packet}
    started = time.monotonic()
    try:
        proc = runner(command, input=json.dumps(packet, ensure_ascii=False), text=True, capture_output=True, timeout=180, check=False)
        raw = proc.stdout or ""
        # Keep the cloud result bounded even if the configured adapter ignores
        # its own output limit.  A rough token count is recorded for auditing.
        response = {"text": raw[-policy.max_output_tokens * 4:], "returncode": proc.returncode}
        response["output_tokens_estimate"] = len(response["text"]) // 4
        status = "completed" if proc.returncode == 0 else "failed"
    except Exception as exc:  # noqa: BLE001
        response = {"error": f"{type(exc).__name__}: {exc}"}
        status = "failed"
    request_tokens = len(json.dumps(packet, ensure_ascii=False)) // 4
    if request_tokens > policy.max_input_tokens:
        status = "failed_input_budget"
        response = {"error": "input token budget exceeded", "input_tokens_estimate": request_tokens}
    store.finish_advice(window_id=wid, provider=policy.provider, model=policy.model, response=response, status=status,
                        input_tokens=request_tokens, output_tokens=(len(response.get("text", "")) // 4 if response else None))
    return {"status": status, "window_id": wid, "elapsed_seconds": time.monotonic() - started, "response": response}
