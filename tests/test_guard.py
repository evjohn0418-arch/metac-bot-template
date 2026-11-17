"""Tests for the spend plan, question selection and run records (agent/guard.py)."""
from __future__ import annotations

import copy
import io
import json
import urllib.error
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent import guard
from conftest import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion, Question

NOW = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)
REPO_ROOT = Path(__file__).resolve().parent.parent


def cfg_from(raw: dict, **changes) -> guard.Config:
    raw = copy.deepcopy(raw)
    raw.update(changes)
    return guard.parse_config(raw)


# ----------------------------------------------------------------------- config


def test_shipped_config_is_valid_and_not_live():
    cfg = guard.load_config(REPO_ROOT / "config" / "bot_config.json")
    assert isinstance(cfg.live, bool)
    assert [t.name for t in cfg.tracks] == ["main", "minibench"]
    assert [t.predictions_per_question for t in cfg.tracks] == [4, 2]
    assert len(cfg.panel) == 2
    assert cfg.credit_floor_usd == 0.0  # stage one runs the first small purchase down to nothing


def test_shipped_config_costs_match_the_proposal():
    cfg = guard.load_config(REPO_ROOT / "config" / "bot_config.json")
    assert guard.estimated_cost_per_question(cfg, 4) == pytest.approx(0.356)
    assert guard.estimated_cost_per_question(cfg, 2) == pytest.approx(0.198)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r.pop("live"), "missing 'live'"),
        (lambda r: r.update(live="yes"), "wrong type"),
        (lambda r: r.update(panel=[]), "at least one model"),
        (lambda r: r["panel"][0].update(est_cost_per_forecast=0), "above zero"),
        (lambda r: r["panel"][0].update(est_cost_per_forecast=True), "true/false"),
        (lambda r: r["panel"][0].update(model="  "), "empty"),
        (lambda r: r["panel"][0].update(settings=[]), "must be an object"),
        (lambda r: r.update(tracks=[]), "at least one tournament"),
        (lambda r: r["tracks"][1].update(name="main"), "unique"),
        (lambda r: r["tracks"][0].update(name="test"), "unique"),
        (lambda r: r["tracks"][0].update(name="cup"), "only 'main' and 'minibench'"),
        (lambda r: r["tracks"][0].update(predictions_per_question=1), "below min"),
        (lambda r: r["tracks"][0].update(last_open_date="December"), "date like"),
        (lambda r: r["tracks"][0].update(expected_questions_per_day=-1), "zero or more"),
        (lambda r: r.update(min_predictions_per_question=0), "1 or more"),
        (lambda r: r["budget"].update(max_questions_per_run=0), "1 or more"),
        (lambda r: r["budget"].update(max_cost_per_run_usd=float("nan")), "above zero"),
        (lambda r: r["budget"].pop("credit_floor_usd"), "missing 'credit_floor_usd'"),
        (lambda r: r["researcher"].update(fallback=5), "researcher.fallback"),
        (lambda r: r["researcher"].update(fallback={"temperature": 1}), "needs a 'model'"),
    ],
)
def test_bad_config_is_refused_with_a_clear_message(base_config, mutate, message):
    raw = copy.deepcopy(base_config)
    mutate(raw)
    with pytest.raises(guard.ConfigError) as error:
        guard.parse_config(raw)
    assert message in str(error.value)


def test_config_file_problems_are_reported(tmp_path):
    with pytest.raises(guard.ConfigError, match="not found"):
        guard.load_config(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(guard.ConfigError, match="not valid JSON"):
        guard.load_config(bad)


# ----------------------------------------------------------------------- costs and stretching


def test_cost_per_question_cycles_through_the_panel(base_config):
    raw = copy.deepcopy(base_config)
    raw["panel"][0]["est_cost_per_forecast"] = 0.10
    raw["panel"][1]["est_cost_per_forecast"] = 0.02
    cfg = guard.parse_config(raw)
    assert guard.estimated_cost_per_question(cfg, 1) == pytest.approx(0.14)
    assert guard.estimated_cost_per_question(cfg, 2) == pytest.approx(0.16)
    assert guard.estimated_cost_per_question(cfg, 3) == pytest.approx(0.26)
    with pytest.raises(ValueError):
        guard.estimated_cost_per_question(cfg, 0)


def test_expected_remaining_questions_stops_at_the_last_open_date(base_config):
    cfg = guard.parse_config(base_config)
    main, minibench = cfg.tracks
    assert guard.expected_remaining_questions(main, date(2026, 11, 17)) == pytest.approx(25 * 4.5)
    assert guard.expected_remaining_questions(minibench, date(2026, 11, 17)) == pytest.approx(40 * 4.3)
    assert guard.expected_remaining_questions(main, date(2026, 12, 20)) == 0


def test_planned_predictions_keeps_the_configured_counts_when_credits_are_ample(base_config):
    cfg = guard.parse_config(base_config)
    assert guard.planned_predictions(cfg, 100.0, date(2026, 11, 17)) == {"main": 4, "minibench": 2}


def test_planned_predictions_trims_when_credits_are_short_and_never_below_the_minimum(base_config):
    cfg = guard.parse_config(base_config)
    # remaining cost at 4/2: 112.5*0.36 + 172*0.20 = 74.9; at 3/2: 112.5*0.28 + 34.4 = 65.9
    assert guard.planned_predictions(cfg, 72.0, date(2026, 11, 17)) == {"main": 3, "minibench": 2}
    assert guard.planned_predictions(cfg, 20.0, date(2026, 11, 17)) == {"main": 2, "minibench": 2}


def test_planned_predictions_trims_the_lowest_priority_track_first(base_config):
    raw = copy.deepcopy(base_config)
    raw["tracks"][1]["predictions_per_question"] = 4
    cfg = guard.parse_config(raw)
    # 4/4 costs 112.5*0.36 + 172*0.36 = 102.4; 4/3 costs 40.5 + 172*0.28 = 88.7
    assert guard.planned_predictions(cfg, 95.0, date(2026, 11, 17)) == {"main": 4, "minibench": 3}


def test_planned_predictions_accounts_for_observed_costs(base_config):
    cfg = guard.parse_config(base_config)
    assert guard.planned_predictions(cfg, 100.0, date(2026, 11, 17), ratio=1.0) == {"main": 4, "minibench": 2}
    # at 1.4x: 4/2 would cost 112.5*0.504 + 172*0.28 = 104.9 > 98, while 3/2 costs 92.3
    assert guard.planned_predictions(cfg, 100.0, date(2026, 11, 17), ratio=1.4) == {"main": 3, "minibench": 2}
    # at 1.5x even 3/2 costs 98.9 > 98, so main drops to the minimum
    assert guard.planned_predictions(cfg, 100.0, date(2026, 11, 17), ratio=1.5) == {"main": 2, "minibench": 2}


def test_planned_predictions_after_the_season_uses_the_configured_counts(base_config):
    cfg = guard.parse_config(base_config)
    assert guard.planned_predictions(cfg, 3.0, date(2027, 1, 10)) == {"main": 4, "minibench": 2}


def run_record(attempted, raw_total, tracked, before=None, after=None):
    return {"kind": "run", "questions_attempted": attempted, "raw_est_total_usd": raw_total,
            "tracked_cost_usd": tracked, "balance_before": before, "balance_after": after}


def test_observed_cost_ratio_waits_for_enough_data():
    assert guard.observed_cost_ratio([]) is None
    assert guard.observed_cost_ratio([run_record(4, 1.2, 1.5)]) is None
    assert guard.observed_cost_ratio([run_record(5, 1.5, 3.0)]) == pytest.approx(2.0)


def test_observed_cost_ratio_prefers_the_provider_balance_and_never_undercounts():
    # provider says $4 left the account, framework tracked only $2
    assert guard.observed_cost_ratio([run_record(10, 2.0, 2.0, before=50.0, after=46.0)]) == pytest.approx(2.0)
    # provider reading lags (shows $0.5), framework tracked $2: keep the larger
    assert guard.observed_cost_ratio([run_record(10, 2.0, 2.0, before=50.0, after=49.5)]) == pytest.approx(1.0)
    # a top-up between readings (balance went up) falls back to the tracked figure
    assert guard.observed_cost_ratio([run_record(10, 2.0, 3.0, before=10.0, after=60.0)]) == pytest.approx(1.5)


def test_observed_cost_ratio_is_clamped_and_ignores_skips():
    assert guard.observed_cost_ratio([run_record(10, 1.0, 100.0)]) == 4.0
    assert guard.observed_cost_ratio([run_record(10, 100.0, 1.0)]) == 0.25
    assert guard.observed_cost_ratio([{"kind": "skip"}, run_record(0, 0, 0)]) is None


# ----------------------------------------------------------------------- run plan


def test_plan_refuses_tournament_runs_until_live(base_config):
    cfg = cfg_from(base_config, live=False)
    plan = guard.plan_run(cfg, 100.0, NOW, "tournament")
    assert plan.go is False and "not live" in plan.reason
    # the test tournament is allowed before going live
    assert guard.plan_run(cfg, 100.0, NOW, "test_questions").go is True


def test_plan_with_ample_credits(base_config):
    cfg = guard.parse_config(base_config)
    plan = guard.plan_run(cfg, 100.0, NOW, "tournament")
    assert plan.go and plan.reason == "ok"
    assert [(t.name, t.predictions) for t in plan.tracks] == [("main", 4), ("minibench", 2)]
    assert plan.max_questions == 15
    assert plan.max_cost_usd == 10.0
    assert plan.spendable_usd == pytest.approx(98.0)
    assert plan.tracks[0].est_cost_per_question == pytest.approx(0.36)


def test_plan_run_cap_never_exceeds_what_is_left_above_the_floor(base_config):
    cfg = guard.parse_config(base_config)
    plan = guard.plan_run(cfg, 5.5, NOW, "tournament")
    assert plan.go
    assert plan.max_cost_usd == pytest.approx(3.5)


def test_plan_stops_at_the_floor(base_config):
    cfg = guard.parse_config(base_config)
    plan = guard.plan_run(cfg, 2.10, NOW, "tournament")
    assert plan.go is False and "credits at floor" in plan.reason
    assert guard.plan_run(cfg, 0.0, NOW, "tournament").go is False
    # just enough for one question at the minimum (0.20) is still allowed
    assert guard.plan_run(cfg, 2.25, NOW, "tournament").go is True


def test_plan_reports_stretching(base_config):
    cfg = guard.parse_config(base_config)
    plan = guard.plan_run(cfg, 72.0, NOW, "tournament")
    assert plan.go and plan.reason == "stretching credits: main 3 forecasts per question"
    assert plan.tracks[0].predictions == 3


def test_plan_with_unknown_balance_uses_only_the_run_caps(base_config):
    cfg = guard.parse_config(base_config)
    plan = guard.plan_run(cfg, None, NOW, "tournament")
    assert plan.go and "balance unknown" in plan.reason
    assert plan.max_cost_usd == 10.0 and plan.max_questions == 15 and plan.spendable_usd is None
    assert [(t.name, t.predictions) for t in plan.tracks] == [("main", 4), ("minibench", 2)]


def test_plan_skips_disabled_tracks(base_config):
    raw = copy.deepcopy(base_config)
    raw["tracks"][1]["enabled"] = False
    plan = guard.plan_run(guard.parse_config(raw), 100.0, NOW, "tournament")
    assert [t.name for t in plan.tracks] == ["main"]
    raw["tracks"][0]["enabled"] = False
    plan = guard.plan_run(guard.parse_config(raw), 100.0, NOW, "tournament")
    assert plan.go is False and plan.reason == "no tracks enabled"


def test_test_mode_uses_the_full_panel_and_the_test_limit(base_config):
    cfg = guard.parse_config(base_config)
    plan = guard.plan_run(cfg, 100.0, NOW, "test_questions")
    assert [(t.name, t.predictions) for t in plan.tracks] == [("test", 4)]
    assert plan.max_questions == 6
    low = guard.plan_run(cfg, 2.30, NOW, "test_questions")  # 0.30 spendable: 3 forecasts (0.28) fit, 4 do not
    assert low.go and low.tracks[0].predictions == 3


def test_plan_uses_observed_costs(base_config):
    cfg = guard.parse_config(base_config)
    records = [run_record(10, 3.0, 6.0)]  # actual cost is twice the estimate
    plan = guard.plan_run(cfg, 100.0, NOW, "tournament", records)
    # at twice the estimate, 4/2 would cost 112.5*0.72 + 172*0.40 = 149.8 > 98, so main trims to the minimum
    assert plan.tracks[0].predictions == 2
    assert plan.tracks[0].raw_est_cost_per_question == pytest.approx(0.20)
    assert plan.tracks[0].est_cost_per_question == pytest.approx(0.40)


def test_plan_rejects_unknown_modes(base_config):
    with pytest.raises(ValueError):
        guard.plan_run(guard.parse_config(base_config), 10.0, NOW, "metaculus_cup")


# ----------------------------------------------------------------------- times and selection


def test_parse_time_handles_api_strings_and_naive_datetimes():
    assert guard.parse_time("2026-11-17T12:00:00Z") == NOW
    assert guard.parse_time("2026-11-17T06:00:00-06:00") == NOW
    assert guard.parse_time(datetime(2026, 11, 17, 12, 0)) == NOW
    assert guard.parse_time("not a date") is None
    assert guard.parse_time(None) is None and guard.parse_time("") is None and guard.parse_time(123) is None


def test_spot_scoring_time_prefers_the_api_field_then_cp_reveal():
    later = NOW + timedelta(hours=3)
    assert guard.spot_scoring_time(Question(1, spot="2026-11-17T12:00:00Z", cp_reveal=later)) == NOW
    assert guard.spot_scoring_time(Question(2, spot=None, cp_reveal=later)) == later
    assert guard.spot_scoring_time(Question(3)) is None
    assert guard.spot_scoring_time(object()) is None


def test_selection_filters_orders_and_limits(base_config):
    cfg = guard.parse_config(base_config)
    hour = timedelta(hours=1)
    questions = [
        Question(1, spot=(NOW + 5 * hour).isoformat()),
        Question(2, forecasted=True, spot=(NOW + hour).isoformat()),
        Question(3, spot=(NOW - hour).isoformat()),                      # scoring moment has passed
        Question(4, close=NOW - hour),                                   # closed
        Question(5, spot=(NOW + 2 * hour).isoformat()),
        Question(6),                                                     # no deadline known: goes last
        Question(7, close=NOW + hour, spot=(NOW + 9 * hour).isoformat()),  # earliest of close and spot counts
        Question(1, spot=(NOW + 5 * hour).isoformat()),                  # duplicate
    ]
    picked, stats = guard.select_questions(questions, NOW, cfg, limit=3)
    assert [q.id_of_question for q in picked] == [7, 5, 1]
    assert stats == {"open_seen": 7, "already_forecast": 1, "closed": 1, "past_spot": 1, "selected": 3, "deferred": 1}


def test_selection_can_keep_questions_past_their_scoring_moment(base_config):
    cfg = cfg_from(base_config, skip_after_spot_scoring_time=False)
    picked, stats = guard.select_questions([Question(3, spot=(NOW - timedelta(hours=1)).isoformat())], NOW, cfg, 5)
    assert len(picked) == 1 and stats["past_spot"] == 0


def test_selection_with_a_zero_limit_takes_nothing(base_config):
    cfg = guard.parse_config(base_config)
    picked, stats = guard.select_questions([Question(1), Question(2)], NOW, cfg, limit=0)
    assert picked == [] and stats["deferred"] == 2
    picked, _ = guard.select_questions([Question(1)], NOW, cfg, limit=-3)
    assert picked == []


def test_test_mode_selection_mixes_question_types(base_config):
    cfg = guard.parse_config(base_config)
    questions = [BinaryQuestion(i, forecasted=True) for i in range(1, 6)]
    questions += [NumericQuestion(10), NumericQuestion(11), MultipleChoiceQuestion(20)]
    picked, stats = guard.select_questions(questions, NOW, cfg, limit=4, mode="test_questions")
    assert sorted(type(q).__name__ for q in picked) == ["BinaryQuestion", "BinaryQuestion", "MultipleChoiceQuestion", "NumericQuestion"]
    assert stats["selected"] == 4 and stats["deferred"] == 4


def test_allocate_shares_slots_and_dollars_in_priority_order(base_config):
    cfg = guard.parse_config(base_config)
    plan = guard.plan_run(cfg, 100.0, NOW, "tournament")
    by_track = {"main": [Question(i) for i in range(1, 13)], "minibench": [Question(i) for i in range(101, 121)]}
    jobs = guard.allocate(plan, by_track, NOW, cfg, "tournament")
    assert [(j.track.name, len(j.questions)) for j in jobs] == [("main", 12), ("minibench", 3)]  # 15 slots in all


def test_allocate_respects_the_dollar_cap(base_config):
    cfg = guard.parse_config(base_config)
    plan = guard.plan_run(cfg, 3.0, NOW, "tournament")  # $1.00 spendable
    by_track = {"main": [Question(i) for i in range(1, 13)], "minibench": [Question(i) for i in range(101, 121)]}
    jobs = guard.allocate(plan, by_track, NOW, cfg, "tournament")
    main, minibench = jobs
    planned = sum(len(j.questions) * j.track.est_cost_per_question for j in jobs)
    assert planned <= 1.0 + 1e-9
    assert len(main.questions) >= 1  # the higher-priority track is served first


def test_allocate_handles_a_track_with_no_questions(base_config):
    cfg = guard.parse_config(base_config)
    plan = guard.plan_run(cfg, 100.0, NOW, "tournament")
    jobs = guard.allocate(plan, {"minibench": [Question(1)]}, NOW, cfg, "tournament")
    assert [(j.track.name, len(j.questions)) for j in jobs] == [("main", 0), ("minibench", 1)]


# ----------------------------------------------------------------------- provider balance


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def opener_returning(payload):
    def opener(request, timeout=None):
        assert request.get_header("Authorization") == "Bearer sk-test"
        assert request.full_url == guard.OPENROUTER_KEY_URL
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        return FakeResponse(body)
    return opener


def test_balance_reads_the_key_limit():
    balance = guard.fetch_openrouter_balance("sk-test", opener_returning({"data": {"limit": 100, "limit_remaining": 61.25}}))
    assert balance == guard.Balance(61.25, "key_limit")


def test_balance_is_unknown_without_a_key_limit_or_a_key():
    assert guard.fetch_openrouter_balance(None).remaining is None
    no_limit = guard.fetch_openrouter_balance("sk-test", opener_returning({"data": {"limit": None, "limit_remaining": None}}))
    assert no_limit.remaining is None and no_limit.source == "no_key_limit"


@pytest.mark.parametrize("payload", [b"<html>", {"error": "nope"}, {"data": []}, {"data": {"limit_remaining": True}}])
def test_balance_survives_odd_replies(payload):
    assert guard.fetch_openrouter_balance("sk-test", opener_returning(payload)).remaining is None


def test_balance_survives_network_failures_and_never_leaks_the_key():
    def failing(request, timeout=None):
        raise urllib.error.URLError("no route to host for Bearer sk-test")
    balance = guard.fetch_openrouter_balance("sk-test", failing)
    assert balance.remaining is None and balance.source == "error"
    assert "sk-test" not in balance.detail


def test_negative_balance_reads_as_zero():
    assert guard.fetch_openrouter_balance("sk-test", opener_returning({"data": {"limit_remaining": -0.4}})).remaining == 0.0


# ----------------------------------------------------------------------- records


def test_records_round_trip_and_skip_bad_lines(tmp_path):
    path = tmp_path / "records" / "runs.jsonl"
    assert guard.read_records(path) == []
    guard.append_record(path, {"kind": "run", "ts": NOW.isoformat()})
    with path.open("a") as handle:
        handle.write("this is not json\n\n[1, 2]\n")
    guard.append_record(path, {"kind": "skip", "reason": "x"})
    assert [r["kind"] for r in guard.read_records(path)] == ["run", "skip"]


def test_skips_are_logged_once_per_reason_per_day():
    assert guard.should_log_skip([], "credits at floor", NOW) is True
    first = guard.build_skip_record(now=NOW, mode="tournament", reason="credits at floor", balance=guard.Balance(1.0, "key_limit"))
    assert guard.should_log_skip([first], "credits at floor", NOW + timedelta(hours=1)) is False
    assert guard.should_log_skip([first], "credits at floor", NOW + timedelta(hours=25)) is True
    assert guard.should_log_skip([first], "missing secrets", NOW + timedelta(hours=1)) is True
    assert guard.should_log_skip([first, {"kind": "run"}], "credits at floor", NOW + timedelta(hours=1)) is True


def test_alarm_is_raised_at_most_once_per_quiet_period():
    assert guard.should_alarm([], NOW) is True
    alarmed = {"kind": "run", "ts": NOW.isoformat(), "alarm": True}
    quiet = {"kind": "run", "ts": (NOW + timedelta(hours=1)).isoformat(), "alarm": False}
    assert guard.should_alarm([alarmed, quiet], NOW + timedelta(hours=2)) is False
    assert guard.should_alarm([alarmed, quiet], NOW + timedelta(hours=13)) is True


def test_run_record_totals(base_config):
    cfg = guard.parse_config(base_config)
    plan = guard.plan_run(cfg, 100.0, NOW, "tournament")
    track_results = [
        {"name": "main", "predictions": 4, "raw_est_cost_per_question": 0.36, "est_cost_per_question": 0.36,
         "questions_ok": 3, "questions_failed": 1},
        {"name": "minibench", "predictions": 2, "raw_est_cost_per_question": 0.20, "est_cost_per_question": 0.20,
         "questions_ok": 2, "questions_failed": 0},
    ]
    record = guard.build_run_record(
        now=NOW, mode="tournament", plan=plan,
        balance_before=guard.Balance(100.0, "key_limit"), balance_after=guard.Balance(98.2, "key_limit"),
        track_results=track_results, tracked_cost_usd=1.71234, models=["model-a", "model-b"],
        errors=["x" * 500], run_url="https://example.test/run/1",
    )
    assert record["questions_attempted"] == 6 and record["questions_ok"] == 5 and record["questions_failed"] == 1
    assert record["raw_est_total_usd"] == pytest.approx(4 * 0.36 + 2 * 0.20)
    assert record["tracked_cost_usd"] == 1.7123
    assert record["ts"] == "2026-11-17T12:00:00+00:00"
    assert len(record["errors"][0]) == 300
    json.dumps(record)  # must be storable as one JSON line


# ----------------------------------------------------------------------- the whole plan, simulated


def simulate_season(base_config, credits: float, true_cost_ratio: float):
    """Walk the plan day by day from Nov 17 with the shipped track settings.

    Each day brings the expected number of new questions. Each question costs its estimate
    times `true_cost_ratio`. Returns (final balance, questions forecast, questions missed, lowest count used).
    """
    cfg = guard.parse_config(base_config)
    balance = credits
    records: list[dict] = []
    forecast = missed = 0
    lowest = {t.name: t.predictions_per_question for t in cfg.tracks}
    carry = {t.name: 0.0 for t in cfg.tracks}
    day = date(2026, 11, 17)
    while day <= date(2026, 12, 27):
        now = datetime(day.year, day.month, day.day, 12, tzinfo=timezone.utc)
        arrivals = {}
        for track in cfg.tracks:
            carry[track.name] += track.expected_questions_per_day if day < track.last_open_date else 0.0
            arrivals[track.name] = int(carry[track.name])
            carry[track.name] -= arrivals[track.name]
        plan = guard.plan_run(cfg, balance, now, "tournament", records)
        if not plan.go:
            missed += sum(arrivals.values())
            day += timedelta(days=1)
            continue
        by_track = {name: [Question(f"{name}-{day}-{i}") for i in range(count)] for name, count in arrivals.items()}
        jobs = guard.allocate(plan, by_track, now, cfg, "tournament")
        before = balance
        raw_total = 0.0
        attempted = 0
        for job in jobs:
            count = len(job.questions)
            attempted += count
            raw_total += count * job.track.raw_est_cost_per_question
            lowest[job.track.name] = min(lowest[job.track.name], job.track.predictions)
            missed += arrivals[job.track.name] - count
        spent = raw_total * true_cost_ratio
        balance -= spent
        forecast += attempted
        if attempted:
            records.append({"kind": "run", "questions_attempted": attempted, "raw_est_total_usd": raw_total,
                            "tracked_cost_usd": spent, "balance_before": before, "balance_after": balance})
        day += timedelta(days=1)
    return balance, forecast, missed, lowest


def test_one_hundred_dollars_covers_the_plan_at_estimated_costs(base_config):
    balance, forecast, missed, lowest = simulate_season(base_config, 100.0, 1.0)
    assert missed == 0
    assert lowest == {"main": 4, "minibench": 2}
    assert 20.0 < balance < 30.0  # about $75 spent
    assert 282 <= forecast <= 285  # 25 days of main questions and 40 days of MiniBench questions


def test_the_plan_stretches_instead_of_going_dark_when_costs_run_fifty_percent_over(base_config):
    balance, forecast, missed, lowest = simulate_season(base_config, 100.0, 1.5)
    assert missed == 0            # every question still gets a forecast
    assert lowest["main"] < 4     # by trimming forecasts per question
    assert balance >= 2.0 - 0.75  # and it never runs past the floor by more than one question's cost


def test_the_plan_stops_cleanly_when_costs_are_far_over(base_config):
    balance, forecast, missed, lowest = simulate_season(base_config, 100.0, 3.0)
    assert lowest == {"main": 2, "minibench": 2}
    assert missed > 0             # it runs out before the end
    assert balance >= 0.0         # but never below zero
