"""Check that OpenRouter refuses a call the balance cannot cover.

Standard library only. The check sends one small request that asks for permission to
write far more text than the remaining credits could pay for. OpenRouter prices a
request at its worst case before serving it, so a working cutoff answers HTTP 402 and
nothing is spent. If the request is served instead, it costs a fraction of a cent (the
prompt asks for one word) and the result says the cutoff was not shown this way.

The result is appended to records/probes.jsonl. It never raises an alarm.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent import guard

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
PROBE_MODEL = "openai/gpt-5.5"
OUTPUT_USD_PER_TOKEN = 30.0 / 1_000_000  # list price on OpenRouter, checked 2026-11-16
MODEL_MAX_TOKENS = 128_000
OVERSHOOT = 1.05     # ask for a little more than the balance could pay for...
MARGIN_USD = 0.10    # ...plus ten cents, so rounding cannot make the request affordable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "bot_config.json"
DEFAULT_RECORDS = REPO_ROOT / "records" / "probes.jsonl"


def probe_max_tokens(remaining: float) -> int | None:
    """Output tokens whose worst-case price is clearly more than `remaining`; None if the model cannot ask for that many."""
    needed = math.ceil((max(remaining, 0.0) * OVERSHOOT + MARGIN_USD) / OUTPUT_USD_PER_TOKEN)
    return None if needed > MODEL_MAX_TOKENS else max(needed, 1000)


def highest_probeable_balance() -> float:
    return (MODEL_MAX_TOKENS * OUTPUT_USD_PER_TOKEN - MARGIN_USD) / OVERSHOOT


def _refusal_details(body: Any) -> dict:
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return {"message": str(body)[:300]}
    metadata = error.get("metadata") if isinstance(error.get("metadata"), dict) else {}
    return {
        "message": str(error.get("message", ""))[:300],
        "reason": metadata.get("reason"),
        "limit_source": metadata.get("limit_source"),
    }


def run_probe(
    api_key: str | None,
    funded_usd: float | None,
    now: datetime | None = None,
    opener: Callable[..., Any] | None = None,
    timeout: float = 60.0,
) -> dict:
    """Run the check once and return the record to save."""
    now = now or datetime.now(timezone.utc)
    record: dict = {"kind": "probe", "ts": now.isoformat(timespec="seconds"), "model": PROBE_MODEL}
    if not api_key:
        return {**record, "outcome": "not_run", "why": "no OPENROUTER_API_KEY in the environment"}

    before = guard.fetch_openrouter_balance(api_key, opener, funded_usd=funded_usd)
    record.update(balance_before=before.remaining, balance_source=before.source, usage_before=before.usage)
    if before.remaining is None:
        return {**record, "outcome": "not_run", "why": f"could not read the balance: {before.detail or before.source}"}

    max_tokens = probe_max_tokens(before.remaining)
    if max_tokens is None:
        ceiling = highest_probeable_balance()
        return {
            **record,
            "outcome": "not_run",
            "why": f"${before.remaining:.2f} is left; this free check only works below ${ceiling:.2f}",
        }
    record.update(max_tokens=max_tokens, worst_case_usd=round(max_tokens * OUTPUT_USD_PER_TOKEN, 4))

    payload = {
        "model": PROBE_MODEL,
        "messages": [{"role": "user", "content": "Reply with the single word OK."}],
        "max_tokens": max_tokens,
        "reasoning": {"effort": "minimal"},
    }
    request = urllib.request.Request(
        CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with (opener or urllib.request.urlopen)(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            body = json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            body = json.loads(exc.read().decode("utf-8") or "{}")
        except (ValueError, OSError):
            body = {}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return {**record, "outcome": "error", "why": f"could not reach OpenRouter: {type(exc).__name__}"}

    record["http_status"] = status
    # Some gateways answer 200 with an error object inside; treat its code as the status.
    inner = body.get("error") if isinstance(body, dict) else None
    if isinstance(inner, dict) and isinstance(inner.get("code"), int):
        status = inner["code"]
        record["http_status"] = status

    if status == 402:
        record.update(outcome="refused", **_refusal_details(body))
        record["why"] = "OpenRouter refused a call that could have cost more than the credits left"
    elif status == 200:
        after = guard.fetch_openrouter_balance(api_key, opener, funded_usd=funded_usd)
        spent = None
        if before.usage is not None and after.usage is not None:
            spent = round(max(after.usage - before.usage, 0.0), 6)
        record.update(outcome="served", usage_after=after.usage, cost_usd=spent)
        record["why"] = "OpenRouter served the call; the cutoff was not shown by this check"
    else:
        record.update(outcome="error", **_refusal_details(body))
        record["why"] = f"unexpected HTTP {status}"
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check that OpenRouter refuses a call the balance cannot cover")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--records", default=str(DEFAULT_RECORDS))
    args = parser.parse_args(argv)
    cfg = guard.load_config(args.config)
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip() or None
    record = run_probe(api_key, cfg.funded_usd)
    guard.append_record(args.records, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
