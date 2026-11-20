"""Count how many of each tournament's questions the bot has forecast.

Read-only: lists the questions in the main tournament and the current MiniBench (open,
closed and resolved) with the bot's token, and checks which ones carry a forecast from
the bot. Questions the bot never forecast score nothing, so this is the bot's coverage.

Writes records/reviews/coverage-latest.json and .md. Run by the review workflow.
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "records" / "reviews"
BOT_START = datetime(2026, 11, 17, 9, 28, tzinfo=timezone.utc)  # first live tournament run
STATUSES = ["open", "closed", "resolved"]


def _status(question: Any) -> str:
    state = getattr(question, "state", None)
    value = getattr(state, "value", state)
    return str(value or "unknown").lower()


def _time(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def summarize(questions: Iterable[Any], bot_start: datetime = BOT_START) -> dict:
    """Coverage figures for one tournament's questions."""
    unique: dict = {}
    for q in questions:
        unique.setdefault(getattr(q, "id_of_question", None) or id(q), q)
    summary = {
        "questions": len(unique),
        "forecast": 0,
        "by_status": {},
        "missed_before_bot_started": 0,
        "missed_after_bot_started": 0,
        "open_not_yet_forecast": 0,
        "missed_examples": [],
    }
    for q in unique.values():
        status = _status(q)
        row = summary["by_status"].setdefault(status, {"questions": 0, "forecast": 0})
        row["questions"] += 1
        if getattr(q, "already_forecasted", False):
            summary["forecast"] += 1
            row["forecast"] += 1
            continue
        if status == "open":
            summary["open_not_yet_forecast"] += 1
            continue
        opened = _time(getattr(q, "open_time", None))
        closes = _time(getattr(q, "close_time", None))
        if opened is not None and closes is not None and closes <= bot_start:
            summary["missed_before_bot_started"] += 1
        elif opened is not None and opened < bot_start and closes is None:
            summary["missed_before_bot_started"] += 1
        else:
            summary["missed_after_bot_started"] += 1
            if len(summary["missed_examples"]) < 10:
                summary["missed_examples"].append(
                    {
                        "url": getattr(q, "page_url", None),
                        "opened": opened.isoformat() if opened else None,
                        "closes": closes.isoformat() if closes else None,
                        "status": status,
                    }
                )
    summary["coverage_pct"] = round(100 * summary["forecast"] / summary["questions"], 1) if summary["questions"] else None
    return summary


def to_markdown(results: dict, generated: str) -> str:
    lines = [f"# Coverage: questions the bot forecast", "", f"generated {generated}", ""]
    for name, s in results.items():
        if "error" in s:
            lines += [f"## {name}", "", f"- could not list questions: {s['error']}", ""]
            continue
        lines += [
            f"## {name}",
            "",
            f"- Questions listed: {s['questions']}",
            f"- Forecast by the bot: {s['forecast']} ({s['coverage_pct']}%)",
            f"- Missed, closed before the bot started: {s['missed_before_bot_started']}",
            f"- Missed after the bot started: {s['missed_after_bot_started']}",
            f"- Open, not forecast yet: {s['open_not_yet_forecast']}",
            "",
        ]
        for status, row in sorted(s["by_status"].items()):
            lines.append(f"  - {status}: {row['forecast']} of {row['questions']} forecast")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    from forecasting_tools import MetaculusClient
    from forecasting_tools.helpers.metaculus_client import ApiFilter

    client = MetaculusClient()
    tournaments = {
        "main": MetaculusClient.CURRENT_AI_COMPETITION_ID,
        "minibench": MetaculusClient.CURRENT_MINIBENCH_ID,
    }
    results: dict = {}
    for name, tournament in tournaments.items():
        try:
            api_filter = ApiFilter(
                allowed_tournaments=[tournament],
                allowed_statuses=STATUSES,
                group_question_mode="unpack_subquestions",
            )
            questions = asyncio.run(
                client.get_questions_matching_filter(
                    api_filter, num_questions=2000, error_if_question_target_missed=False
                )
            )
            results[name] = {"tournament": str(tournament), **summarize(questions)}
        except Exception as exc:  # a listing failure is reported, never raised
            results[name] = {"tournament": str(tournament), "error": f"{type(exc).__name__}: {exc}"[:300]}
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "coverage-latest.json").write_text(json.dumps({"generated": generated, **results}, indent=2, default=str))
    markdown = to_markdown(results, generated)
    (OUT_DIR / "coverage-latest.md").write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
