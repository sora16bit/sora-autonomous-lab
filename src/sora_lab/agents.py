from __future__ import annotations

import json
import hashlib
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentReply:
    role: str
    model: str
    raw: str
    parsed: dict[str, Any] | None


ROLE_PROMPTS = {
    "improver": "JSONだけで回答。形式は {\"reason\":文字列,\"changes\":{許可済み生成パラメータ:数値},\"next_experiment\":文字列,\"uncertainty\":文字列}。仮説は1つ。評価器・採用規則を変えない。",
    "attacker": "JSONだけで回答。提示された稼働中攻撃と同じfamily・IDを親にする。familyはdistance,blocked_distance,marginal,shadow,ensemble,aia_knn,aia_treeから選ぶ。paramsのキーは正確にk,columns,max_iter,max_leaf_nodes,l2_regularization,childrenのどれかだけ。distance/blocked_distance/aia_knnはk(1-25)、distanceは任意でcolumns、marginalはcolumns、shadow/aia_treeはmax_iter(20-500),max_leaf_nodes(3-64),l2_regularization(0-100)、ensembleは既存IDのchildrenを使う。別表記・未知キーを作らず、親に合うfamilyのパラメータだけ返す。columnsは age,BMI,SBP,TG,HDL,ALT,FPG,time,onset,death,sex,prefecture,smoking のみ。秘密ラベルを要求しない。",
    "reviewer": "JSONだけで回答。形式は {\"decision\":\"accept\"または\"reject\",\"reason\":文字列}。提案の実験リーク、秘密境界、再現性、評価の抜けを検査する。",
}

ALLOWED_CHANGE_KEYS = {"tau_general", "tau_rare", "rare_multiplier", "jitter_general", "jitter_rare", "penalizer", "outcome_draws", "outcome_step"}


def _response_schema(role: str) -> dict[str, Any]:
    string = {"type": "string"}
    if role == "improver":
        return {"type": "object", "properties": {
            "reason": string,
            "changes": {"type": "object", "properties": {key: {"type": "number"} for key in sorted(ALLOWED_CHANGE_KEYS)}, "additionalProperties": False},
            "next_experiment": string, "uncertainty": string,
        }, "required": ["reason", "changes", "next_experiment", "uncertainty"], "additionalProperties": False}
    if role == "attacker":
        columns = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "time", "onset", "death", "sex", "prefecture", "smoking"]
        params = {
            "k": {"type": "integer", "minimum": 1, "maximum": 25},
            "distance_k": {"type": "integer", "minimum": 1, "maximum": 25},
            "columns": {"type": "array", "items": {"type": "string", "enum": columns}, "minItems": 1},
            "max_iter": {"type": "integer", "minimum": 20, "maximum": 500},
            "max_leaf_nodes": {"type": "integer", "minimum": 3, "maximum": 64},
            "l2_regularization": {"type": "number", "minimum": 0, "maximum": 100},
            "children": {"type": "array", "items": string, "minItems": 1},
        }
        return {"type": "object", "properties": {
            "reason": string, "parent_attack_id": string,
            "family": {"type": "string", "enum": ["distance", "blocked_distance", "marginal", "shadow", "ensemble", "aia_knn", "aia_tree"]},
            "params": {"type": "object", "properties": params, "additionalProperties": False},
        }, "required": ["reason", "parent_attack_id", "family", "params"], "additionalProperties": False}
    if role == "reviewer":
        return {"type": "object", "properties": {
            "decision": {"type": "string", "enum": ["accept", "reject"]}, "reason": string,
        }, "required": ["decision", "reason"], "additionalProperties": False}
    raise ValueError(f"unknown role: {role}")


def validate_proposal(reply: AgentReply, *, allowed: set[str] | None = None) -> tuple[bool, list[str]]:
    """Validate an LLM reply as a bounded experiment proposal.

    The reply is data only: shell snippets, imports, arbitrary paths and code
    edits are never accepted by this function.
    """
    reasons: list[str] = []
    if reply.parsed is None:
        return False, ["invalid_json"]
    value = reply.parsed
    if set(value) - {"decision", "reason", "next_experiment", "uncertainty", "changes", "risk"}:
        reasons.append("unknown_top_level_key")
    if not isinstance(value.get("reason"), str) or not value.get("reason", "").strip():
        reasons.append("reason_missing")
    if "changes" not in value:
        reasons.append("changes_missing")
    changes = value.get("changes", {})
    if changes is not None and not isinstance(changes, dict):
        reasons.append("changes_not_object")
    else:
        keys = set(changes or {})
        permitted = allowed or ALLOWED_CHANGE_KEYS
        if keys - permitted:
            reasons.append("change_key_not_allowed")
        ranges = {
            "tau_general": (0.0, 10000.0), "tau_rare": (0.0, 10000.0),
            "rare_multiplier": (0.1, 10.0), "jitter_general": (0.0, 1.0),
            "jitter_rare": (0.0, 1.0), "penalizer": (0.0, 10.0),
            "outcome_draws": (0.0, 1000.0), "outcome_step": (0.001, 1.0),
        }
        for key, raw in (changes or {}).items():
            if key not in ranges:
                continue
            try:
                x = float(raw)
                lo, hi = ranges[key]
                if not (lo <= x <= hi):
                    reasons.append(f"{key}_out_of_range")
            except (TypeError, ValueError):
                reasons.append(f"{key}_not_numeric")
    if not isinstance(value.get("reason", ""), str) or len(str(value.get("reason", ""))) > 1000:
        reasons.append("reason_invalid")
    if "next_experiment" not in value:
        reasons.append("next_experiment_missing")
    if not isinstance(value.get("next_experiment", ""), str) or len(str(value.get("next_experiment", ""))) > 1000:
        reasons.append("next_experiment_invalid")
    if "uncertainty" not in value:
        reasons.append("uncertainty_missing")
    if not isinstance(value.get("uncertainty", ""), str) or len(str(value.get("uncertainty", ""))) > 1000:
        reasons.append("uncertainty_invalid")
    return not reasons, reasons


def validate_review(reply: AgentReply) -> tuple[bool, list[str]]:
    if reply.parsed is None:
        return False, ["invalid_json"]
    value = reply.parsed
    reasons = []
    if set(value) - {"decision", "reason"}:
        reasons.append("unknown_top_level_key")
    if str(value.get("decision", "")).lower() not in {"accept", "accepted", "approve", "approved", "reject", "rejected"}:
        reasons.append("decision_invalid")
    if not isinstance(value.get("reason"), str) or not value.get("reason", "").strip() or len(value.get("reason", "")) > 1000:
        reasons.append("reason_invalid")
    return not reasons, reasons


def parse_attack_proposal(reply: AgentReply, *, allowed_parent_ids: set[str]) -> tuple[dict[str, Any] | None, list[str]]:
    """Convert an LLM suggestion into allow-listed, evaluator-backed attack data."""
    if reply.parsed is None:
        return None, ["invalid_json"]
    value = reply.parsed
    reasons: list[str] = []
    if set(value) != {"reason", "parent_attack_id", "family", "params"}:
        reasons.append("attack_schema_mismatch")
    reason = value.get("reason")
    parent = value.get("parent_attack_id")
    family = value.get("family")
    params = value.get("params")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
        reasons.append("reason_invalid")
    if not isinstance(parent, str) or parent not in allowed_parent_ids:
        reasons.append("parent_attack_not_active")
    if not isinstance(params, dict):
        reasons.append("params_not_object")
        params = {}
    normalized_params: dict[str, Any] = {}
    for key, raw in params.items():
        canonical = {"distance_k": "k"}.get(str(key), str(key))
        if canonical in normalized_params:
            reasons.append("duplicate_parameter_alias")
        normalized_params[canonical] = raw
    params = normalized_params
    allowed_by_family = {
        "distance": {"k", "columns"},
        "blocked_distance": {"k"},
        "marginal": {"columns"},
        "shadow": {"max_iter", "max_leaf_nodes", "l2_regularization"},
        "ensemble": {"children"},
        "aia_knn": {"k"},
        "aia_tree": {"max_iter", "max_leaf_nodes", "l2_regularization"},
    }
    if family not in allowed_by_family:
        reasons.append("family_not_allowed")
    elif set(params) - allowed_by_family[family]:
        reasons.append("family_param_not_allowed")
    if not reasons:
        from .attack_registry import AttackSpec
        payload = {"family": family, "params": params}
        attack_id = "llm_" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
        try:
            spec = AttackSpec(attack_id, family, params, parents=(parent,), generation=0)
            spec.validate()
            return {**spec.to_dict(), "hypothesis": reason}, []
        except (TypeError, ValueError) as exc:
            reasons.append(f"attack_invalid:{type(exc).__name__}")
    return None, reasons


def _extract_json(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
        return _normalize_keys(value) if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start : end + 1])
            return _normalize_keys(value) if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def _normalize_keys(value: dict[str, Any]) -> dict[str, Any]:
    """Map only documented Japanese response labels to the strict schema."""
    aliases = {
        "変更対象": "changes", "反証条件": "uncertainty", "期待する指標": "next_experiment",
        "仮説": "reason", "判断": "decision", "決定": "decision",
    }
    normalized = dict(value)
    for source, target in aliases.items():
        if source in normalized and target not in normalized:
            normalized[target] = normalized[source]
        normalized.pop(source, None)
    return normalized


def ask_ollama(*, role: str, report: dict[str, Any], model: str = "qwen3.6:35b", host: str = "http://127.0.0.1:11434", timeout: int = 180) -> AgentReply:
    if role not in ROLE_PROMPTS:
        raise ValueError(f"unknown role: {role}")
    payload = {
        "model": model,
        "stream": False,
        "format": _response_schema(role),
        "options": {"temperature": 0.2, "num_ctx": 16384},
        "messages": [
            {"role": "system", "content": "あなたはPWS Cupのローカル研究役です。秘密データの値や行を要求せず、短い構造化回答だけ返してください。" + ROLE_PROMPTS[role]},
            {"role": "user", "content": json.dumps(report, ensure_ascii=False, sort_keys=True)},
        ],
    }
    request = urllib.request.Request(host.rstrip("/") + "/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"ollama request failed: {type(exc).__name__}: {exc}") from exc
    raw = str(body.get("message", {}).get("content", ""))
    return AgentReply(role=role, model=model, raw=raw, parsed=_extract_json(raw))
