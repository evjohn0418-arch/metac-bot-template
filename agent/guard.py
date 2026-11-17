"""Spend planning, question selection and run records for the forecasting bot.

Standard library only, so every rule here can be tested without the forecasting
framework. Nothing in this file calls a model. It decides how much the bot may
do on a run, and it writes down what the run did.

The real spending limit lives outside this code: the OpenRouter key has a credit
limit and the account holds prepaid credits with auto top-up off. This module
reads that limit and plans inside it, so the bot slows down before it is cut off.

Credits can be bought in stages. `budget.funded_usd` is the total bought so far; the
bot treats (funded - used) as its balance when that is less than the key's own limit.
A wrong value here can only make the plan wrong. It cannot make OpenRouter serve a
call the account cannot pay for.
"""
from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"
MODES = ("tournament", "test_questions")
TEST_TRACK = "test"


class ConfigError(ValueError):
    """The config file is missing a field or holds a value that cannot be used."""


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class PanelMember:
    model: str
    est_cost_per_forecast: float
    settings: dict
    worst_case_cost_per_call: float | None = None  # dollars if a call used every token it is allowed


@dataclass(frozen=True)
class Track:
    """One tournament the bot forecasts on. Tracks are listed in priority order."""

    name: str
    tournament: str | int | None  # None means: use the framework's current id for this track name
    enabled: bool
    predictions_per_question: int
    last_open_date: date
    expected_questions_per_day: float


@dataclass(frozen=True)
class Config:
    live: bool
    panel: tuple[PanelMember, ...]
    tracks: tuple[Track, ...]
    min_predictions_per_question: int
    researcher_fallback: dict | str | None
    prefer_asknews_when_keys_present: bool
    parser: dict | str
    summarizer: dict | str
    est_overhead_per_question: float
    credit_floor_usd: float
    max_questions_per_run: int
    max_cost_per_run_usd: float
    test_question_limit: int
    skip_after_spot_scoring_time: bool
    funded_usd: float | None = None  # credits bought so far; None means "go by the key's limit alone"
    worst_case_overhead_per_question: float | None = None
    inflight_share: float = 0.5  # share of the balance that calls in flight may add up to, worst case


def _need(mapping: dict, key: str, kind: type | tuple, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{where}: missing '{key}'")
    value = mapping[key]
    # bool is an int in Python; never accept true/false where a number is wanted
    if isinstance(value, bool) and kind is not bool:
        raise ConfigError(f"{where}: '{key}' must not be true/false")
    if not isinstance(value, kind):
        raise ConfigError(f"{where}: '{key}' has the wrong type ({type(value).__name__})")
    return value


def _amount(value: float, name: str, allow_zero: bool = False) -> float:
    if math.isnan(value) or math.isinf(value) or value < 0 or (value == 0 and not allow_zero):
        raise ConfigError(f"'{name}' must be {'zero or more' if allow_zero else 'above zero'}")
    return float(value)


def _iso_date(value: str, name: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'{name}' must be a date like 2026-12-15") from exc


def parse_config(raw: dict) -> Config:
    if not isinstance(raw, dict):
        raise ConfigError("config must be a JSON object")
    number = (int, float)

    panel_raw = _need(raw, "panel", list, "config")
    if not panel_raw:
        raise ConfigError("config: 'panel' needs at least one model")
    panel = []
    for position, entry in enumerate(panel_raw):
        where = f"panel[{position}]"
        if not isinstance(entry, dict):
            raise ConfigError(f"{where}: must be an object")
        model = _need(entry, "model", str, where).strip()
        if not model:
            raise ConfigError(f"{where}: 'model' is empty")
        cost = _amount(_need(entry, "est_cost_per_forecast", number, where), f"{where}.est_cost_per_forecast")
        settings = entry.get("settings", {})
        if not isinstance(settings, dict):
            raise ConfigError(f"{where}: 'settings' must be an object")
        worst = entry.get("worst_case_cost_per_call")
        if worst is not None:
            if isinstance(worst, bool) or not isinstance(worst, number):
                raise ConfigError(f"{where}: 'worst_case_cost_per_call' must be a number")
            worst = _amount(worst, f"{where}.worst_case_cost_per_call")
            if worst < cost:
                raise ConfigError(f"{where}: worst_case_cost_per_call is below est_cost_per_forecast")
        panel.append(
            PanelMember(model=model, est_cost_per_forecast=cost, settings=dict(settings), worst_case_cost_per_call=worst)
        )

    minimum = _need(raw, "min_predictions_per_question", int, "config")
    if minimum < 1:
        raise ConfigError("min_predictions_per_question must be 1 or more")

    tracks_raw = _need(raw, "tracks", list, "config")
    if not tracks_raw:
        raise ConfigError("config: 'tracks' needs at least one tournament")
    tracks, names = [], set()
    for position, entry in enumerate(tracks_raw):
        where = f"tracks[{position}]"
        if not isinstance(entry, dict):
            raise ConfigError(f"{where}: must be an object")
        name = _need(entry, "name", str, where).strip()
        if not name or name == TEST_TRACK or name in names:
            raise ConfigError(f"{where}: 'name' must be unique, non-empty and not '{TEST_TRACK}'")
        names.add(name)
        tournament = entry.get("tournament")
        if tournament is not None and (isinstance(tournament, bool) or not isinstance(tournament, (str, int))):
            raise ConfigError(f"{where}: 'tournament' must be a slug, an id or null")
        if tournament is None and name not in ("main", "minibench"):
            raise ConfigError(f"{where}: only 'main' and 'minibench' may leave 'tournament' null")
        predictions = _need(entry, "predictions_per_question", int, where)
        if predictions < minimum:
            raise ConfigError(f"{where}: predictions_per_question is below min_predictions_per_question")
        tracks.append(
            Track(
                name=name,
                tournament=tournament,
                enabled=bool(entry.get("enabled", True)),
                predictions_per_question=predictions,
                last_open_date=_iso_date(_need(entry, "last_open_date", str, where), f"{where}.last_open_date"),
                expected_questions_per_day=_amount(
                    _need(entry, "expected_questions_per_day", number, where),
                    f"{where}.expected_questions_per_day",
                    allow_zero=True,
                ),
            )
        )

    researcher = _need(raw, "researcher", dict, "config")
    fallback = researcher.get("fallback")
    if fallback is not None and not isinstance(fallback, (dict, str)):
        raise ConfigError("researcher.fallback must be an object, a name or null")
    if isinstance(fallback, dict) and not isinstance(fallback.get("model"), str):
        raise ConfigError("researcher.fallback needs a 'model' name")

    budget = _need(raw, "budget", dict, "config")
    max_questions = _need(budget, "max_questions_per_run", int, "budget")
    test_limit = _need(budget, "test_question_limit", int, "budget")
    if max_questions < 1 or test_limit < 1:
        raise ConfigError("budget: question limits must be 1 or more")

    def optional_amount(mapping: dict, key: str, where: str) -> float | None:
        value = mapping.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, number):
            raise ConfigError(f"{where}: '{key}' must be a number or null")
        return _amount(value, f"{where}.{key}", allow_zero=True)

    def model_spec(key: str) -> dict | str:
        value = _need(raw, key, (str, dict), "config")
        if isinstance(value, dict):
            if not isinstance(value.get("model"), str) or not value["model"].strip():
                raise ConfigError(f"config: '{key}' needs a 'model' name")
            return dict(value)
        if not value.strip():
            raise ConfigError(f"config: '{key}' is empty")
        return value

    share = budget.get("inflight_share", 0.5)
    if isinstance(share, bool) or not isinstance(share, number) or not 0 < share <= 1:
        raise ConfigError("budget: 'inflight_share' must be above 0 and at most 1")

    return Config(
        live=_need(raw, "live", bool, "config"),
        panel=tuple(panel),
        tracks=tuple(tracks),
        min_predictions_per_question=minimum,
        researcher_fallback=fallback,
        prefer_asknews_when_keys_present=bool(researcher.get("prefer_asknews_when_keys_present", True)),
        parser=model_spec("parser"),
        summarizer=model_spec("summarizer"),
        est_overhead_per_question=_amount(
            _need(raw, "est_overhead_per_question", number, "config"), "est_overhead_per_question", allow_zero=True
        ),
        credit_floor_usd=_amount(_need(budget, "credit_floor_usd", number, "budget"), "budget.credit_floor_usd", allow_zero=True),
        max_questions_per_run=max_questions,
        max_cost_per_run_usd=_amount(_need(budget, "max_cost_per_run_usd", number, "budget"), "budget.max_cost_per_run_usd"),
        test_question_limit=test_limit,
        skip_after_spot_scoring_time=bool(raw.get("skip_after_spot_scoring_time", True)),
        funded_usd=optional_amount(budget, "funded_usd", "budget"),
        worst_case_overhead_per_question=optional_amount(raw, "worst_case_overhead_per_question", "config"),
        inflight_share=float(share),
    )


def load_config(path: str | Path) -> Config:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config file is not valid JSON: {exc}") from exc
    return parse_config(raw)


# --------------------------------------------------------------------------- costs


def estimated_cost_per_question(cfg: Config, predictions: int) -> float:
    """Estimated dollars for one question: the first `predictions` panel slots plus research and parsing."""
    if predictions < 1:
        raise ValueError("predictions must be 1 or more")
    forecasts = sum(cfg.panel[slot % len(cfg.panel)].est_cost_per_forecast for slot in range(predictions))
    return forecasts + cfg.est_overhead_per_question


def worst_case_cost_per_question(cfg: Config, predictions: int) -> float:
    """Dollars one question could cost if every call used all the tokens it is allowed.

    OpenRouter prices a request at its worst case before serving it and refuses it when the
    worst cases of the calls in flight do not fit in the balance. Where the config gives no
    worst case, four times the estimate stands in.
    """
    if predictions < 1:
        raise ValueError("predictions must be 1 or more")
    total = 0.0
    for slot in range(predictions):
        member = cfg.panel[slot % len(cfg.panel)]
        total += member.worst_case_cost_per_call or member.est_cost_per_forecast * 4
    overhead = cfg.worst_case_overhead_per_question
    return total + (overhead if overhead is not None else cfg.est_overhead_per_question * 4)


def parallel_questions(cfg: Config, balance: float | None, predictions: int, cap: int) -> int:
    """How many questions to forecast side by side, so calls in flight fit inside the balance."""
    cap = max(cap, 1)
    if balance is None:
        return cap
    worst = worst_case_cost_per_question(cfg, predictions)
    if worst <= 0:
        return cap
    return max(1, min(cap, int((balance * cfg.inflight_share) // worst)))


def observed_cost_ratio(records: Sequence[dict], min_questions: int = 5, window: int = 60) -> float | None:
    """Actual cost divided by estimated cost over recent runs, or None until there is enough data.

    Uses the provider's own figures where they exist (the rise in the key's usage, else the fall
    in the balance), and never less than what the framework tracked, so a late-posting charge
    cannot make the bot look cheaper than it is.
    """
    runs = [r for r in records if r.get("kind") == "run" and (r.get("questions_attempted") or 0) > 0][-window:]
    if sum(r["questions_attempted"] for r in runs) < min_questions:
        return None
    estimated = sum(r.get("raw_est_total_usd") or 0.0 for r in runs)
    if estimated <= 0:
        return None
    actual = 0.0
    for r in runs:
        tracked = r.get("tracked_cost_usd") or 0.0
        provider = r.get("provider_cost_usd")
        before, after = r.get("balance_before"), r.get("balance_after")
        if isinstance(provider, (int, float)) and not isinstance(provider, bool) and provider >= 0:
            actual += max(provider, tracked)
        elif before is not None and after is not None and before >= after:
            actual += max(before - after, tracked)
        else:
            actual += tracked
    if actual <= 0:
        return None
    return min(max(actual / estimated, 0.25), 4.0)


def expected_remaining_questions(track: Track, today: date) -> float:
    return max((track.last_open_date - today).days, 0) * track.expected_questions_per_day


def planned_predictions(cfg: Config, balance: float, today: date, ratio: float = 1.0) -> dict[str, int]:
    """Forecasts per question for each enabled track that the credits can sustain to the end of the plan.

    Starts from the configured counts. If the expected remaining cost is more than the credits left,
    it trims one forecast at a time, lowest-priority track first, never below the minimum.
    """
    tracks = [t for t in cfg.tracks if t.enabled]
    counts = {t.name: t.predictions_per_question for t in tracks}
    spendable = balance - cfg.credit_floor_usd

    def remaining_cost() -> float:
        return sum(
            expected_remaining_questions(t, today) * estimated_cost_per_question(cfg, counts[t.name]) * ratio
            for t in tracks
        )

    while remaining_cost() > spendable:
        trimmable = [t for t in reversed(tracks) if counts[t.name] > cfg.min_predictions_per_question]
        if not trimmable:
            break
        counts[trimmable[0].name] -= 1
    return counts


# --------------------------------------------------------------------------- run plan


@dataclass(frozen=True)
class TrackPlan:
    name: str
    tournament: str | int | None
    predictions: int
    est_cost_per_question: float      # calibrated by observed costs
    raw_est_cost_per_question: float  # straight from the config estimates


@dataclass(frozen=True)
class RunPlan:
    go: bool
    reason: str
    tracks: tuple[TrackPlan, ...] = ()
    max_questions: int = 0
    max_cost_usd: float = 0.0
    spendable_usd: float | None = None


REFUSAL_MARKERS = ("requires more credits", "insufficient credits", "openrouter_credits", "openrouter_key_limit",
                   "weight_exceeds_budget", "in_flight_budget_exhausted", "key limit exceeded", "payment required")


def is_provider_refusal(error: Any) -> bool:
    """True when an error says the provider would not serve a call for lack of credits (HTTP 402)."""
    text = str(error).lower()
    if any(marker in text for marker in REFUSAL_MARKERS):
        return True
    return "402" in text and ("credit" in text or "limit" in text or "budget" in text)


def waiting_for_credits(records: Sequence[dict], balance: float | None, slack: float = 0.05) -> bool:
    """True when the provider cut the last run off and the balance has not grown since.

    Stops the bot from knocking on a closed door every 20 minutes. New credits (or a higher
    funded_usd) raise the balance and the bot starts again by itself.
    """
    if balance is None:
        return False
    for entry in reversed(records):
        if entry.get("kind") != "run":
            continue
        if not entry.get("cutoff_hit"):
            return False
        after = entry.get("balance_after")
        return isinstance(after, (int, float)) and not isinstance(after, bool) and balance <= after + slack
    return False


def plan_run(
    cfg: Config,
    balance: float | None,
    now: datetime,
    mode: str,
    records: Sequence[dict] = (),
) -> RunPlan:
    """Decide whether this run may forecast, on which tracks, and with how many forecasts per question."""
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    testing = mode == "test_questions"
    if not testing and not cfg.live:
        return RunPlan(False, "not live: config has live=false")

    enabled = [t for t in cfg.tracks if t.enabled]
    if not testing and not enabled:
        return RunPlan(False, "no tracks enabled")

    ratio = observed_cost_ratio(records) or 1.0
    limit = cfg.test_question_limit if testing else cfg.max_questions_per_run

    def track_plan(name: str, tournament: str | int | None, predictions: int) -> TrackPlan:
        raw = estimated_cost_per_question(cfg, predictions)
        return TrackPlan(name, tournament, predictions, raw * ratio, raw)

    full_test = max(t.predictions_per_question for t in cfg.tracks)

    if balance is None:
        # The key has no limit we can read. The prepaid balance still stops spending at zero,
        # so run, but stay inside the per-run caps.
        if testing:
            plans = (track_plan(TEST_TRACK, None, full_test),)
        else:
            plans = tuple(track_plan(t.name, t.tournament, t.predictions_per_question) for t in enabled)
        return RunPlan(True, "balance unknown: per-run caps only", plans, limit, cfg.max_cost_per_run_usd, None)

    if waiting_for_credits(records, balance):
        return RunPlan(False, f"provider cut the last run off for lack of credits; ${balance:.2f} left, waiting for more")

    spendable = balance - cfg.credit_floor_usd
    cheapest = estimated_cost_per_question(cfg, cfg.min_predictions_per_question) * ratio
    at_floor = RunPlan(False, f"credits at floor: ${balance:.2f} left, floor ${cfg.credit_floor_usd:.2f}")
    if spendable < cheapest:
        return at_floor

    if testing:
        predictions = full_test
        while predictions > cfg.min_predictions_per_question and estimated_cost_per_question(cfg, predictions) * ratio > spendable:
            predictions -= 1
        plans = (track_plan(TEST_TRACK, None, predictions),)
        reason = "ok"
    else:
        counts = planned_predictions(cfg, balance, now.date(), ratio)
        plans = tuple(track_plan(t.name, t.tournament, counts[t.name]) for t in enabled)
        trimmed = [f"{t.name} {counts[t.name]}" for t in enabled if counts[t.name] < t.predictions_per_question]
        reason = "ok" if not trimmed else "stretching credits: " + ", ".join(trimmed) + " forecasts per question"

    return RunPlan(True, reason, plans, limit, min(cfg.max_cost_per_run_usd, spendable), spendable)


# --------------------------------------------------------------------------- question selection


def parse_time(value: Any) -> datetime | None:
    """Read an API timestamp or a datetime; treat a time without a zone as UTC."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def spot_scoring_time(question: Any) -> datetime | None:
    """When the question is scored. A forecast made after this moment earns nothing."""
    api_json = getattr(question, "api_json", None)
    raw = None
    if isinstance(api_json, dict) and isinstance(api_json.get("question"), dict):
        raw = api_json["question"].get("spot_scoring_time")
    return parse_time(raw) or parse_time(getattr(question, "cp_reveal_time", None))


def _deadline(question: Any) -> datetime | None:
    moments = [m for m in (spot_scoring_time(question), parse_time(getattr(question, "close_time", None))) if m]
    return min(moments) if moments else None


def select_questions(
    questions: Iterable[Any],
    now: datetime,
    cfg: Config,
    limit: int,
    mode: str = "tournament",
) -> tuple[list[Any], dict]:
    """Pick the questions for this run and say what was left out and why.

    Tournament runs skip questions already forecast, questions already closed and (by default)
    questions whose scoring moment has passed, then take the nearest deadlines first.
    Test runs take a mix of question types instead.
    """
    now = parse_time(now) or datetime.now(timezone.utc)
    limit = max(limit, 0)
    seen: set = set()
    unique = []
    for question in questions:
        key = getattr(question, "id_of_question", None) or id(question)
        if key in seen:
            continue
        seen.add(key)
        unique.append(question)

    stats = {"open_seen": len(unique), "already_forecast": 0, "closed": 0, "past_spot": 0, "selected": 0, "deferred": 0}

    if mode == "test_questions":
        by_type: dict[str, list[Any]] = {}
        for question in unique:
            by_type.setdefault(type(question).__name__, []).append(question)
        picked: list[Any] = []
        while len(picked) < limit and any(by_type.values()):
            for name in sorted(by_type):
                if by_type[name] and len(picked) < limit:
                    picked.append(by_type[name].pop(0))
        stats["selected"] = len(picked)
        stats["deferred"] = len(unique) - len(picked)
        return picked, stats

    candidates = []
    for question in unique:
        if getattr(question, "already_forecasted", False):
            stats["already_forecast"] += 1
            continue
        close = parse_time(getattr(question, "close_time", None))
        if close is not None and close <= now:
            stats["closed"] += 1
            continue
        spot = spot_scoring_time(question)
        if cfg.skip_after_spot_scoring_time and spot is not None and spot <= now:
            stats["past_spot"] += 1
            continue
        candidates.append(question)

    far_future = datetime.max.replace(tzinfo=timezone.utc)
    candidates.sort(key=lambda q: _deadline(q) or far_future)
    picked = candidates[:limit]
    stats["selected"] = len(picked)
    stats["deferred"] = len(candidates) - len(picked)
    return picked, stats


@dataclass(frozen=True)
class Job:
    track: TrackPlan
    questions: tuple
    stats: dict


def allocate(
    plan: RunPlan,
    questions_by_track: dict[str, Sequence[Any]],
    now: datetime,
    cfg: Config,
    mode: str,
) -> list[Job]:
    """Share this run's question slots and dollars across the tracks, in priority order."""
    slots = plan.max_questions
    dollars = plan.max_cost_usd
    jobs = []
    for track in plan.tracks:
        affordable = int(dollars // track.est_cost_per_question) if track.est_cost_per_question > 0 else slots
        limit = max(min(slots, affordable), 0)
        picked, stats = select_questions(questions_by_track.get(track.name, ()), now, cfg, limit, mode)
        slots -= len(picked)
        dollars -= len(picked) * track.est_cost_per_question
        jobs.append(Job(track, tuple(picked), stats))
    return jobs


# --------------------------------------------------------------------------- provider balance


@dataclass(frozen=True)
class Balance:
    remaining: float | None
    source: str
    detail: str = ""
    usage: float | None = None  # what this key has spent in total, by OpenRouter's own count


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return float(value)


def fetch_openrouter_balance(
    api_key: str | None,
    opener: Callable[..., Any] | None = None,
    timeout: float = 15.0,
    funded_usd: float | None = None,
) -> Balance:
    """Ask OpenRouter how much this key may still spend.

    The answer is the smallest of: what is left of the key's own limit, what is left of the
    credits bought so far (funded_usd minus the key's usage), and any budget OpenRouter reports
    as the one that blocks first. Returns remaining=None when none of these can be read. The
    caller then falls back to the per-run caps, and the prepaid balance remains the hard stop.
    """
    if not api_key:
        return Balance(None, "none", "no OPENROUTER_API_KEY in the environment")
    request = urllib.request.Request(
        OPENROUTER_KEY_URL, headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    )
    try:
        with (opener or urllib.request.urlopen)(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return Balance(None, "error", f"could not read the key: {type(exc).__name__}")
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return Balance(None, "error", "unexpected reply from OpenRouter")
    usage = _number(data.get("usage"))
    candidates: list[tuple[float, str]] = []
    key_left = _number(data.get("limit_remaining"))
    if key_left is not None:
        candidates.append((key_left, "key_limit"))
    if funded_usd is not None:
        if usage is None:
            return Balance(None, "error", "OpenRouter did not report this key's usage")
        candidates.append((funded_usd - usage, "funded"))
    effective = data.get("effective_budget")
    if isinstance(effective, dict):
        budget_left = _number(effective.get("remaining_usd"))
        if budget_left is not None:
            candidates.append((budget_left, "budget"))
    if not candidates:
        return Balance(None, "no_key_limit", "the key has no credit limit set", usage)
    remaining, source = min(candidates, key=lambda item: item[0])
    return Balance(max(remaining, 0.0), source, "", usage)


# --------------------------------------------------------------------------- records


def read_records(path: str | Path) -> list[dict]:
    """Read the run ledger. Lines that are not valid JSON are skipped, not fatal."""
    file = Path(path)
    if not file.exists():
        return []
    records = []
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            records.append(entry)
    return records


def append_record(path: str | Path, record: dict) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def should_log_skip(records: Sequence[dict], reason: str, now: datetime, repeat_after_hours: float = 24.0) -> bool:
    """A skipped run is logged when the reason is new, or once a day, so the ledger stays readable."""
    for entry in reversed(records):
        if entry.get("kind") == "run":
            return True
        if entry.get("kind") == "skip":
            if entry.get("reason") != reason:
                return True
            last = parse_time(entry.get("ts"))
            return last is None or (parse_time(now) - last) >= timedelta(hours=repeat_after_hours)
    return True


def should_alarm(records: Sequence[dict], now: datetime, quiet_hours: float = 12.0) -> bool:
    """Fail the workflow (which emails the repo owner) at most once per quiet period."""
    for entry in reversed(records):
        if entry.get("alarm"):
            last = parse_time(entry.get("ts"))
            return last is None or (parse_time(now) - last) >= timedelta(hours=quiet_hours)
    return True


def build_run_record(
    *,
    now: datetime,
    mode: str,
    plan: RunPlan,
    balance_before: Balance,
    balance_after: Balance,
    track_results: Sequence[dict],
    tracked_cost_usd: float | None,
    models: Sequence[str],
    errors: Sequence[str] = (),
    published: bool = True,
    run_url: str | None = None,
    alarm: bool = False,
) -> dict:
    """One ledger line for a run. `track_results` holds, per track: name, predictions, raw and
    calibrated cost estimates, selection stats, and how many questions succeeded and failed."""
    questions_ok = sum(t["questions_ok"] for t in track_results)
    questions_failed = sum(t["questions_failed"] for t in track_results)
    raw_total = sum(t["raw_est_cost_per_question"] * (t["questions_ok"] + t["questions_failed"]) for t in track_results)
    refusals = sum(1 for e in errors if is_provider_refusal(e))
    provider_cost = None
    if balance_before.usage is not None and balance_after.usage is not None:
        provider_cost = round(max(balance_after.usage - balance_before.usage, 0.0), 4)
    return {
        "kind": "run",
        "ts": parse_time(now).isoformat(timespec="seconds"),
        "mode": mode,
        "published": published,
        "reason": plan.reason,
        "balance_before": balance_before.remaining,
        "balance_after": balance_after.remaining,
        "balance_source": balance_before.source,
        "questions_attempted": questions_ok + questions_failed,
        "questions_ok": questions_ok,
        "questions_failed": questions_failed,
        "raw_est_total_usd": round(raw_total, 4),
        "tracked_cost_usd": None if tracked_cost_usd is None else round(tracked_cost_usd, 4),
        "usage_before": balance_before.usage,
        "usage_after": balance_after.usage,
        "provider_cost_usd": provider_cost,
        "provider_refusals": refusals,
        "cutoff_hit": refusals > 0,
        "tracks": [dict(t) for t in track_results],
        "models": list(models),
        "errors": [str(e)[:300] for e in errors][:10],
        "run_url": run_url,
        "alarm": alarm,
    }


def build_skip_record(*, now: datetime, mode: str, reason: str, balance: Balance) -> dict:
    return {
        "kind": "skip",
        "ts": parse_time(now).isoformat(timespec="seconds"),
        "mode": mode,
        "reason": reason,
        "balance_before": balance.remaining,
        "balance_source": balance.source,
    }
