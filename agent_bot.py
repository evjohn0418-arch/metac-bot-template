"""Our entry in the Metaculus FutureEval bot tournament.

A thin layer over Metaculus's official template bot (main.py, left unchanged):

1. A panel of forecaster models. Each question gets several forecasts, rotated across
   the models in config/bot_config.json; the framework takes the median.
2. A spend plan. Before every run the bot reads how much the OpenRouter key may still
   spend and plans inside it (agent/guard.py). The key's credit limit and the prepaid
   balance are the hard stop; this code only makes the bot slow down before it is cut off.
3. A live switch, a per-run cap and a run record appended to records/runs.jsonl.
4. Questions are forecast a few at a time, sized so the calls in flight fit inside the
   balance. If OpenRouter refuses a call for lack of credits, the run stops there, says so
   in its record, and later runs wait until the balance has grown.

No person reviews or edits a forecast before it is submitted.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from agent import guard
from bot_helpers import print_run_summary_banner, print_startup_banner, silence_noisy_dependencies

silence_noisy_dependencies()

from forecasting_tools import GeneralLlm, MetaculusClient, MonetaryCostManager, ReasonedPrediction  # noqa: E402

from main import FallTemplateBot2026  # noqa: E402

logger = logging.getLogger("agent_bot")

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "bot_config.json"
DEFAULT_RECORDS = REPO_ROOT / "records" / "runs.jsonl"
TEST_TOURNAMENT = "bot-testing-area"
_PLACEHOLDERS = {"", "REPLACE_ME", "1234567890", "your-token-here", "your-api-key-here"}

# Which panel model the current forecast task should use. A ContextVar keeps the choice
# private to each concurrent task, so forecasts running side by side cannot swap models.
_PANEL_MEMBER: ContextVar = ContextVar("panel_member", default=None)


class PanelBot(FallTemplateBot2026):
    """The template bot, with its forecasts spread across a panel of models."""

    def __init__(self, *, panel: list, **kwargs) -> None:
        if not panel:
            raise ValueError("the panel needs at least one model")
        self._panel = list(panel)
        self._panel_counters: dict = {}
        # forecasts that came back, and forecasts that failed, per panel model
        self.panel_stats: dict = {member.model: {"ok": 0, "failed": 0} for member in self._panel}
        super().__init__(**kwargs)

    def get_llm(self, purpose: str = "default", guarantee_type=None):
        member = _PANEL_MEMBER.get()
        if purpose == "default" and member is not None:
            return member.model if guarantee_type == "string_name" else member
        return super().get_llm(purpose, guarantee_type)

    def make_llm_dict(self) -> dict:
        llm_dict = super().make_llm_dict()
        llm_dict["forecaster_panel"] = ", ".join(member.model for member in self._panel)
        return llm_dict

    async def _make_prediction(self, question, research: str):
        key = getattr(question, "id_of_question", None) or id(question)
        slot = self._panel_counters.get(key, 0)
        self._panel_counters[key] = slot + 1
        member = self._panel[slot % len(self._panel)]
        token = _PANEL_MEMBER.set(member)
        try:
            prediction = await super()._make_prediction(question, research)
        except BaseException:
            self.panel_stats[member.model]["failed"] += 1
            raise
        finally:
            _PANEL_MEMBER.reset(token)
        self.panel_stats[member.model]["ok"] += 1
        return ReasonedPrediction(
            prediction_value=prediction.prediction_value,
            reasoning=f"*Forecaster model*: {member.model}\n\n{prediction.reasoning}",
        )


def _real_env(name: str) -> str | None:
    value = (os.getenv(name) or "").strip()
    return None if value in _PLACEHOLDERS else value


def build_llm(spec):
    """A model given as a name stays a name; one given with settings becomes a configured model."""
    if isinstance(spec, dict):
        settings = {k: v for k, v in spec.items() if k != "model"}
        return GeneralLlm(model=spec["model"], **settings)
    return spec


def build_researcher(cfg: guard.Config):
    if cfg.prefer_asknews_when_keys_present and _real_env("ASKNEWS_CLIENT_ID") and _real_env("ASKNEWS_SECRET"):
        return "asknews/news-summaries"
    return build_llm(cfg.researcher_fallback) or "no_research"


def build_bot(cfg: guard.Config, predictions: int, publish: bool, skip_forecasted: bool, client=None) -> PanelBot:
    panel = [GeneralLlm(model=member.model, **member.settings) for member in cfg.panel]
    return PanelBot(
        panel=panel,
        research_reports_per_question=1,
        predictions_per_research_report=predictions,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=skip_forecasted,
        extra_metadata_in_explanation=True,
        metaculus_client=client,
        llms={
            "default": panel[0],
            "summarizer": build_llm(cfg.summarizer),
            "researcher": build_researcher(cfg),
            "parser": build_llm(cfg.parser),
        },
    )


def resolve_tournament(track: guard.TrackPlan):
    """The tournament to forecast on. Tracks with no explicit id follow the framework's current season."""
    if track.tournament is not None:
        return track.tournament
    if track.name == "main":
        return MetaculusClient.CURRENT_AI_COMPETITION_ID
    if track.name == "minibench":
        return MetaculusClient.CURRENT_MINIBENCH_ID
    if track.name == guard.TEST_TRACK:
        return TEST_TOURNAMENT
    raise ValueError(f"track '{track.name}' has no tournament")


def all_reports_flat(results: list) -> list:
    return [report for reports in results for report in reports]


def _run_url() -> str | None:
    server, repo, run_id = os.getenv("GITHUB_SERVER_URL"), os.getenv("GITHUB_REPOSITORY"), os.getenv("GITHUB_RUN_ID")
    return f"{server}/{repo}/actions/runs/{run_id}" if server and repo and run_id else None


def _finish_with_alarm(records_path: Path, records: list, record: dict, now: datetime) -> int:
    """Write the record. Exit 1 (which emails the repo owner) at most once per quiet period."""
    alarm = guard.should_alarm(records, now)
    record["alarm"] = alarm
    guard.append_record(records_path, record)
    return 1 if alarm else 0


def report_errors(report) -> list[str]:
    """Every error a question's result carries: the failure itself, or errors inside a result that still succeeded."""
    if isinstance(report, BaseException):
        return [f"{type(report).__name__}: {report}"]
    return [str(e) for e in (getattr(report, "errors", None) or [])]


async def _forecast_jobs(work: list, batch_sizes: list[int]) -> list:
    """Run each track's bot in turn inside one event loop, a few questions at a time.

    Stops starting new questions as soon as the provider refuses a call for lack of credits.
    The questions not started are left for a later run; they are not counted as failures.
    """
    results = []
    cut_off = False
    for (job, bot), size in zip(work, batch_sizes):
        reports: list = []
        questions = list(job.questions)
        for start in range(0, len(questions), max(size, 1)):
            if cut_off:
                break
            batch = questions[start : start + max(size, 1)]
            batch_reports = await bot.forecast_questions(batch, return_exceptions=True)
            reports.extend(batch_reports)
            cut_off = any(guard.is_provider_refusal(e) for r in batch_reports for e in report_errors(r))
        results.append(reports)
    return results


def run(argv: list[str] | None = None, now: datetime | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the forecasting bot inside its spend plan")
    parser.add_argument("--mode", choices=list(guard.MODES), default="tournament")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--records", default=str(DEFAULT_RECORDS))
    parser.add_argument("--dry-run", action="store_true", help="forecast but do not submit to Metaculus")
    args = parser.parse_args(argv)

    now = now or datetime.now(timezone.utc)
    mode: str = args.mode
    records_path = Path(args.records)
    cfg = guard.load_config(args.config)
    records = guard.read_records(records_path)
    publish = not args.dry_run

    if mode == "tournament" and not cfg.live:
        print("Bot is not live (config has live=false). Nothing was forecast.")
        return 0

    metaculus_token = _real_env("METACULUS_TOKEN")
    openrouter_key = _real_env("OPENROUTER_API_KEY")
    if not metaculus_token or not openrouter_key:
        missing = [n for n, v in (("METACULUS_TOKEN", metaculus_token), ("OPENROUTER_API_KEY", openrouter_key)) if not v]
        reason = f"missing secrets: {', '.join(missing)}"
        print(f"Cannot run: {reason}")
        record = guard.build_skip_record(now=now, mode=mode, reason=reason, balance=guard.Balance(None, "none"))
        return _finish_with_alarm(records_path, records, record, now)

    balance_before = guard.fetch_openrouter_balance(openrouter_key, funded_usd=cfg.funded_usd)
    plan = guard.plan_run(cfg, balance_before.remaining, now, mode, records)
    summary = ", ".join(f"{t.name}: {t.predictions} forecasts/question" for t in plan.tracks) or "no tracks"
    print(
        f"Plan: go={plan.go} ({plan.reason}); balance={balance_before.remaining} [{balance_before.source}]; "
        f"{summary}; up to {plan.max_questions} questions; run cap ${plan.max_cost_usd:.2f}"
    )
    if not plan.go:
        if guard.should_log_skip(records, plan.reason, now):
            guard.append_record(
                records_path, guard.build_skip_record(now=now, mode=mode, reason=plan.reason, balance=balance_before)
            )
        return 0

    print_startup_banner(mode, will_publish=publish)
    client = MetaculusClient()
    questions_by_track: dict = {}
    fetch_errors: list[str] = []
    for track in plan.tracks:
        try:
            tournament = resolve_tournament(track)
            questions_by_track[track.name] = client.get_all_open_questions_from_tournament(tournament)
        except Exception as exc:  # one tournament failing must not stop the other
            fetch_errors.append(f"could not list questions for {track.name}: {type(exc).__name__}: {exc}")
            logger.error(fetch_errors[-1])

    jobs = guard.allocate(plan, questions_by_track, now, cfg, mode)
    for job in jobs:
        print(f"Questions on {job.track.name}: {job.stats}")

    skip_forecasted = mode == "tournament"
    work = [
        (job, build_bot(cfg, job.track.predictions, publish, skip_forecasted, client)) for job in jobs if job.questions
    ]

    batch_sizes = [
        guard.parallel_questions(cfg, balance_before.remaining, job.track.predictions, len(job.questions))
        for job, _bot in work
    ]
    for (job, _bot), size in zip(work, batch_sizes):
        print(f"Forecasting {len(job.questions)} question(s) on {job.track.name}, {size} at a time")

    results: list = []
    tracked_cost: float | None = None
    if work:
        with MonetaryCostManager(plan.max_cost_usd) as cost_manager:
            results = asyncio.run(_forecast_jobs(work, batch_sizes))
            tracked_cost = cost_manager.current_usage

    reports_by_track = {job.track.name: reports for (job, _bot), reports in zip(work, results)}
    bots_by_track = {job.track.name: bot for job, bot in work}
    partial_errors = [
        f"partial: {e}" for r in all_reports_flat(results) if not isinstance(r, BaseException) for e in report_errors(r)
    ]
    all_reports = [report for reports in results for report in reports]
    failures = [r for r in all_reports if isinstance(r, BaseException)]
    questions_ok = len(all_reports) - len(failures)

    track_results = []
    for job in jobs:
        reports = reports_by_track.get(job.track.name, [])
        failed = sum(1 for r in reports if isinstance(r, BaseException))
        track_results.append(
            {
                "name": job.track.name,
                "predictions": job.track.predictions,
                "raw_est_cost_per_question": round(job.track.raw_est_cost_per_question, 4),
                "est_cost_per_question": round(job.track.est_cost_per_question, 4),
                **job.stats,
                "questions_ok": len(reports) - failed,
                "questions_failed": failed,
                "questions_not_started": len(job.questions) - len(reports),
                "forecasts_by_model": bots_by_track[job.track.name].panel_stats if job.track.name in bots_by_track else {},
            }
        )

    balance_after = (
        guard.fetch_openrouter_balance(openrouter_key, funded_usd=cfg.funded_usd) if work else balance_before
    )
    record = guard.build_run_record(
        now=now,
        mode=mode,
        plan=plan,
        balance_before=balance_before,
        balance_after=balance_after,
        track_results=track_results,
        tracked_cost_usd=tracked_cost,
        models=[member.model for member in cfg.panel],
        errors=fetch_errors + [f"{type(f).__name__}: {f}" for f in failures] + partial_errors,
        published=publish,
        run_url=_run_url(),
    )

    try:
        for (_job, bot), reports in zip(work, results):
            bot.log_report_summary(reports, raise_errors=False)
        print_run_summary_banner(all_reports, will_publish=publish)
    except Exception as exc:  # a summary problem must never lose the record
        logger.error(f"could not print the run summary: {exc}")

    if record["cutoff_hit"]:
        # Expected when credits run out: the provider's limit did its job. Record it; do not raise an alarm.
        print(
            f"OpenRouter refused {record['provider_refusals']} call(s) for lack of credits. "
            "The bot will wait until the balance has grown."
        )
        guard.append_record(records_path, record)
        return 0

    nothing_listed = not questions_by_track and bool(fetch_errors)
    everything_failed = (bool(all_reports) and questions_ok == 0) or nothing_listed
    if everything_failed:
        return _finish_with_alarm(records_path, records, record, now)
    if work or fetch_errors:
        guard.append_record(records_path, record)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    sys.exit(run())
