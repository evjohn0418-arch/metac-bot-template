"""Tests for the bot wrapper (agent_bot.py) against stand-ins for the framework (see conftest.py)."""
from __future__ import annotations

import asyncio
import copy
import json
from collections import Counter
from datetime import datetime, timedelta, timezone

import pytest

import agent_bot
from agent import guard
from conftest import (
    BinaryQuestion,
    GeneralLlm,
    HardLimitExceededError,
    MetaculusClient,
    MultipleChoiceQuestion,
    NumericQuestion,
    Question,
)

NOW = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)
MAIN = MetaculusClient.CURRENT_AI_COMPETITION_ID
MINI = MetaculusClient.CURRENT_MINIBENCH_ID


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("METACULUS_TOKEN", "metaculus-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    for name in ("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def workspace(tmp_path, base_config):
    """Writes a config file and returns helpers to run the bot and read its records."""
    config_path = tmp_path / "bot_config.json"
    records_path = tmp_path / "records" / "runs.jsonl"

    def write(**changes):
        raw = copy.deepcopy(base_config)
        raw.update(changes)
        config_path.write_text(json.dumps(raw))
        return raw

    write()

    def run(*extra, now=NOW):
        return agent_bot.run(["--config", str(config_path), "--records", str(records_path), *extra], now=now)

    return type("Workspace", (), {"write": staticmethod(write), "run": staticmethod(run),
                                  "records": staticmethod(lambda: guard.read_records(records_path))})


@pytest.fixture
def balances(monkeypatch):
    """Feed the bot a scripted sequence of provider balances (the last one repeats)."""
    script: list = [guard.Balance(100.0, "key_limit")]
    seen: list = []

    def fake(api_key, *args, **kwargs):
        seen.append(api_key)
        return script[min(len(seen) - 1, len(script) - 1)]

    monkeypatch.setattr(guard, "fetch_openrouter_balance", fake)
    return type("Balances", (), {"script": script, "seen": seen})


def panel_bot(predictions: int, models=("model-a", "model-b")) -> agent_bot.PanelBot:
    panel = [GeneralLlm(model=m) for m in models]
    return agent_bot.PanelBot(
        panel=panel,
        predictions_per_research_report=predictions,
        llms={"default": panel[0], "summarizer": "summary-model", "researcher": "no_research", "parser": "parser-model"},
    )


# ----------------------------------------------------------------------- the panel


def test_each_question_gets_an_even_split_across_the_panel():
    GeneralLlm.behaviour = {"model-a": (0.2, 0.01), "model-b": (0.6, 0.01)}
    bot = panel_bot(predictions=4)
    questions = [Question(i) for i in range(1, 8)]
    reports = asyncio.run(bot.forecast_questions(questions))
    assert len(reports) == 7
    for report in reports:
        used = Counter(f.reasoning.splitlines()[0] for f in report.forecasts)
        assert used == {"*Forecaster model*: model-a": 2, "*Forecaster model*: model-b": 2}
        assert report.prediction == pytest.approx(0.4)  # median of 0.2, 0.2, 0.6, 0.6


def test_the_label_always_matches_the_model_that_answered():
    # Different answers per model make any mix-up between concurrent forecasts visible.
    GeneralLlm.behaviour = {"model-a": (0.11, 0.01), "model-b": (0.77, 0.01), "model-c": (0.5, 0.01)}
    bot = panel_bot(predictions=6, models=("model-a", "model-b", "model-c"))
    reports = asyncio.run(bot.forecast_questions([Question(i) for i in range(1, 12)]))
    answers = {"model-a": 0.11, "model-b": 0.77, "model-c": 0.5}
    checked = 0
    for report in reports:
        for forecast in report.forecasts:
            label = forecast.reasoning.splitlines()[0].removeprefix("*Forecaster model*: ")
            assert forecast.prediction_value == answers[label]
            assert forecast.reasoning.endswith(f"reasoned by {label}")
            checked += 1
    assert checked == 11 * 6


def test_an_odd_count_starts_with_the_first_model():
    bot = panel_bot(predictions=3)
    (report,) = asyncio.run(bot.forecast_questions([Question(1)]))
    used = Counter(f.reasoning.splitlines()[0] for f in report.forecasts)
    assert used == {"*Forecaster model*: model-a": 2, "*Forecaster model*: model-b": 1}


def test_other_model_roles_are_untouched_and_the_panel_is_disclosed():
    bot = panel_bot(predictions=2)
    assert bot.get_llm("parser") == "parser-model"
    assert bot.get_llm("default", "llm").model == "model-a"  # outside a forecast: the configured default
    assert bot.make_llm_dict()["forecaster_panel"] == "model-a, model-b"
    with pytest.raises(ValueError):
        agent_bot.PanelBot(panel=[], llms={})


def test_one_failing_model_does_not_sink_the_question():
    GeneralLlm.behaviour = {"model-a": (0.3, 0.01), "model-b": RuntimeError("model-b is down")}
    bot = panel_bot(predictions=4)
    (report,) = asyncio.run(bot.forecast_questions([Question(1)]))
    assert report.prediction == pytest.approx(0.3)
    assert len(report.forecasts) == 2 and len(report.errors) == 2


# ----------------------------------------------------------------------- building the bot


def test_researcher_prefers_asknews_only_when_both_keys_are_real(env, base_config, monkeypatch):
    cfg = guard.parse_config(base_config)
    assert agent_bot.build_researcher(cfg).model == "search-model"
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
    assert agent_bot.build_researcher(cfg).model == "search-model"  # secret still missing
    monkeypatch.setenv("ASKNEWS_SECRET", "REPLACE_ME")
    assert agent_bot.build_researcher(cfg).model == "search-model"  # placeholder is not a key
    monkeypatch.setenv("ASKNEWS_SECRET", "secret")
    assert agent_bot.build_researcher(cfg) == "asknews/news-summaries"


def test_researcher_can_be_switched_off(env, base_config):
    raw = copy.deepcopy(base_config)
    raw["researcher"] = {"prefer_asknews_when_keys_present": False, "fallback": None}
    assert agent_bot.build_researcher(guard.parse_config(raw)) == "no_research"


def test_model_settings_from_the_config_reach_the_models(env, base_config):
    bot = agent_bot.build_bot(guard.parse_config(base_config), predictions=4, publish=False, skip_forecasted=True)
    assert [m.model for m in bot._panel] == ["model-a", "model-b"]
    assert bot._panel[0].kwargs == {"reasoning_effort": "high"}
    assert bot.get_llm("researcher").kwargs == {"temperature": 0.1}
    assert bot.predictions_per_research_report == 4 and bot.extra_metadata_in_explanation is True


def test_tracks_resolve_to_the_right_tournaments():
    plan = lambda name, tournament=None: guard.TrackPlan(name, tournament, 2, 0.2, 0.2)
    assert agent_bot.resolve_tournament(plan("main")) == MAIN
    assert agent_bot.resolve_tournament(plan("minibench")) == MINI
    assert agent_bot.resolve_tournament(plan("test")) == "bot-testing-area"
    assert agent_bot.resolve_tournament(plan("main", "spring-2027")) == "spring-2027"
    with pytest.raises(ValueError):
        agent_bot.resolve_tournament(plan("other"))


# ----------------------------------------------------------------------- whole runs


def test_nothing_happens_until_the_live_switch_is_on(env, workspace, balances):
    workspace.write(live=False)
    MetaculusClient.questions = {MAIN: [Question(1)]}
    assert workspace.run() == 0
    assert workspace.records() == [] and MetaculusClient.list_calls == [] and balances.seen == []
    assert GeneralLlm.calls == []


def test_a_normal_run_forecasts_the_right_questions_and_records_it(env, workspace, balances):
    hour = timedelta(hours=1)
    balances.script[:] = [guard.Balance(100.0, "key_limit"), guard.Balance(99.1, "key_limit")]
    GeneralLlm.behaviour = {"model-a": (0.2, 0.05), "model-b": (0.6, 0.05), "search-model": (0.0, 0.02)}
    MetaculusClient.questions = {
        MAIN: [
            Question(1, spot=(NOW + 3 * hour).isoformat()),
            Question(2, forecasted=True),
            Question(3, spot=(NOW - hour).isoformat()),  # too late to score: must not cost anything
        ],
        MINI: [Question(11), Question(12)],
    }
    assert workspace.run() == 0

    forecast_calls = Counter(prompt for _model, prompt in GeneralLlm.calls if prompt.startswith("forecast"))
    assert forecast_calls == {"forecast 1": 4, "forecast 11": 2, "forecast 12": 2}
    assert sorted(MetaculusClient.published) == [(1, pytest.approx(0.4)), (11, pytest.approx(0.4)), (12, pytest.approx(0.4))]

    (record,) = workspace.records()
    assert record["kind"] == "run" and record["mode"] == "tournament" and record["published"] is True
    assert record["questions_ok"] == 3 and record["questions_failed"] == 0 and record["alarm"] is False
    assert record["balance_before"] == 100.0 and record["balance_after"] == 99.1
    # 8 forecasts at $0.05 plus 3 research calls at $0.02
    assert record["tracked_cost_usd"] == pytest.approx(0.46)
    assert record["raw_est_total_usd"] == pytest.approx(0.36 + 2 * 0.20)
    main, minibench = record["tracks"]
    assert (main["name"], main["predictions"], main["selected"], main["already_forecast"], main["past_spot"]) == ("main", 4, 1, 1, 1)
    assert (minibench["name"], minibench["predictions"], minibench["questions_ok"]) == ("minibench", 2, 2)
    assert record["models"] == ["model-a", "model-b"]


def test_a_run_with_nothing_to_do_leaves_no_record(env, workspace, balances):
    MetaculusClient.questions = {MAIN: [Question(2, forecasted=True)], MINI: []}
    assert workspace.run() == 0
    assert workspace.records() == [] and GeneralLlm.calls == []


def test_a_dry_run_submits_nothing(env, workspace, balances):
    MetaculusClient.questions = {MAIN: [Question(1)]}
    assert workspace.run("--dry-run") == 0
    assert MetaculusClient.published == []
    assert workspace.records()[0]["published"] is False


def test_the_bot_stops_at_the_credit_floor_and_logs_it_once(env, workspace, balances):
    balances.script[:] = [guard.Balance(2.05, "key_limit")]
    MetaculusClient.questions = {MAIN: [Question(1)]}
    assert workspace.run() == 0
    assert workspace.run(now=NOW + timedelta(minutes=20)) == 0
    records = workspace.records()
    assert len(records) == 1 and records[0]["kind"] == "skip" and "credits at floor" in records[0]["reason"]
    assert GeneralLlm.calls == [] and MetaculusClient.list_calls == []


def test_low_credits_trim_the_forecasts_per_question(env, workspace, balances):
    balances.script[:] = [guard.Balance(30.0, "key_limit")]
    MetaculusClient.questions = {MAIN: [Question(1)], MINI: [Question(11)]}
    assert workspace.run() == 0
    forecast_calls = Counter(prompt for _model, prompt in GeneralLlm.calls if prompt.startswith("forecast"))
    assert forecast_calls == {"forecast 1": 2, "forecast 11": 2}
    assert "stretching credits" in workspace.records()[0]["reason"]


def test_missing_secrets_raise_one_alarm_then_stay_quiet(workspace, balances, monkeypatch):
    monkeypatch.delenv("METACULUS_TOKEN", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "REPLACE_ME")
    assert workspace.run() == 1
    assert workspace.run(now=NOW + timedelta(minutes=20)) == 0
    assert workspace.run(now=NOW + timedelta(hours=13)) == 1
    records = workspace.records()
    assert [r["alarm"] for r in records] == [True, False, True]
    assert "METACULUS_TOKEN" in records[0]["reason"] and "OPENROUTER_API_KEY" in records[0]["reason"]
    assert balances.seen == [] and MetaculusClient.list_calls == []


def test_total_failure_raises_one_alarm_and_keeps_the_errors(env, workspace, balances):
    GeneralLlm.behaviour = {"model-a": RuntimeError("401 bad key"), "model-b": RuntimeError("401 bad key")}
    MetaculusClient.questions = {MAIN: [Question(1), Question(2)]}
    assert workspace.run() == 1
    assert workspace.run(now=NOW + timedelta(minutes=20)) == 0
    first, second = workspace.records()
    assert first["alarm"] is True and second["alarm"] is False
    assert first["questions_ok"] == 0 and first["questions_failed"] == 2
    assert any("All predictions failed" in e for e in first["errors"])
    assert MetaculusClient.published == []


def test_partial_failure_is_recorded_without_an_alarm(env, workspace, balances):
    GeneralLlm.behaviour = {"model-a": (0.3, 0.01), "model-b": RuntimeError("model-b is down")}
    MetaculusClient.questions = {MAIN: [Question(1)]}
    assert workspace.run() == 0
    (record,) = workspace.records()
    assert record["questions_ok"] == 1 and record["alarm"] is False


def test_one_tournament_failing_to_list_does_not_stop_the_other(env, workspace, balances):
    MetaculusClient.questions = {MAIN: ConnectionError("metaculus timeout"), MINI: [Question(11)]}
    assert workspace.run() == 0
    (record,) = workspace.records()
    assert record["questions_ok"] == 1
    assert any("could not list questions for main" in e for e in record["errors"])


def test_every_tournament_failing_to_list_is_an_alarm(env, workspace, balances):
    MetaculusClient.questions = {MAIN: ConnectionError("down"), MINI: ConnectionError("down")}
    assert workspace.run() == 1
    assert workspace.records()[0]["alarm"] is True


def test_the_run_cap_is_enforced_while_the_run_is_under_way(env, workspace, balances):
    raw = workspace.write()
    raw["budget"]["max_cost_per_run_usd"] = 0.25
    raw["budget"]["max_questions_per_run"] = 15
    workspace.write(budget=raw["budget"], tracks=[raw["tracks"][0]])
    # the plan expects $0.36 a question, so it allows none... make the estimate cheap and the reality dear
    cheap = copy.deepcopy(raw["panel"])
    for member in cheap:
        member["est_cost_per_forecast"] = 0.001
    workspace.write(budget=raw["budget"], tracks=[raw["tracks"][0]], panel=cheap, est_overhead_per_question=0.0)
    GeneralLlm.behaviour = {"model-a": (0.2, 0.05), "model-b": (0.6, 0.05), "search-model": (0.0, 0.0)}
    MetaculusClient.questions = {MAIN: [Question(i) for i in range(1, 11)]}
    workspace.run()
    (record,) = workspace.records()
    assert record["tracked_cost_usd"] <= 0.25 + 1e-9          # the hard cap held
    assert record["questions_attempted"] == 10
    assert record["questions_failed"] > 0                     # the rest were refused, not paid for
    spent = sum(0.05 for _m, prompt in GeneralLlm.calls if prompt.startswith("forecast"))
    assert spent <= 0.25 + 1e-9


def test_an_unreadable_balance_still_runs_inside_the_run_caps(env, workspace, balances):
    balances.script[:] = [guard.Balance(None, "error", "could not read the key")]
    MetaculusClient.questions = {MAIN: [Question(i) for i in range(1, 30)]}
    assert workspace.run() == 0
    (record,) = workspace.records()
    assert record["balance_before"] is None and "balance unknown" in record["reason"]
    assert record["questions_attempted"] == 15  # the per-run question cap


def test_the_test_mode_works_before_going_live_and_mixes_question_types(env, workspace, balances):
    workspace.write(live=False)
    MetaculusClient.questions = {
        "bot-testing-area": [BinaryQuestion(i, forecasted=True) for i in range(1, 9)]
        + [NumericQuestion(20), NumericQuestion(21), MultipleChoiceQuestion(30)]
    }
    assert workspace.run("--mode", "test_questions") == 0
    assert MetaculusClient.list_calls == ["bot-testing-area"]
    (record,) = workspace.records()
    assert record["mode"] == "test_questions" and record["questions_ok"] == 6
    forecast_calls = Counter(prompt for _model, prompt in GeneralLlm.calls if prompt.startswith("forecast"))
    assert set(forecast_calls.values()) == {4}                # the full panel on every test question
    assert {"forecast 20", "forecast 21", "forecast 30"} <= set(forecast_calls)


def test_observed_costs_feed_back_into_the_next_plan(env, workspace, balances):
    # First run: 5 questions that really cost twice their estimate.
    balances.script[:] = [guard.Balance(100.0, "key_limit"), guard.Balance(96.4, "key_limit"), guard.Balance(96.4, "key_limit")]
    MetaculusClient.questions = {MAIN: [Question(i) for i in range(1, 6)]}
    assert workspace.run() == 0
    first = workspace.records()[0]
    assert first["raw_est_total_usd"] == pytest.approx(1.8) and first["balance_before"] - first["balance_after"] == pytest.approx(3.6)
    # Second run plans with the observed cost, so it trims the main track.
    MetaculusClient.questions = {MAIN: [Question(i) for i in range(6, 8)]}
    assert workspace.run(now=NOW + timedelta(minutes=20)) == 0
    second = workspace.records()[1]
    assert second["tracks"][0]["predictions"] == 2
    assert second["tracks"][0]["est_cost_per_question"] == pytest.approx(0.40)


def test_run_url_is_recorded_on_github(env, workspace, balances, monkeypatch):
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "someone/forecast-bot")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    MetaculusClient.questions = {MAIN: [Question(1)]}
    workspace.run()
    assert workspace.records()[0]["run_url"] == "https://github.com/someone/forecast-bot/actions/runs/42"


def test_the_hard_limit_error_type_is_what_stops_spending():
    # Guards the assumption in test_the_run_cap_is_enforced...: the stand-in refuses before charging.
    from conftest import MonetaryCostManager
    with MonetaryCostManager(0.03):
        with pytest.raises(HardLimitExceededError):
            asyncio.run(GeneralLlm(model="x").invoke("p")) if GeneralLlm.behaviour.update({"x": (0.5, 0.05)}) is None else None


# ----------------------------------------------------------------------- staged funding and the provider cutoff

REFUSAL = (
    'OpenrouterException - {"error":{"message":"This request requires more credits, or fewer max_tokens.",'
    '"code":402,"metadata":{"reason":"weight_exceeds_budget"}}}'
)


class RefuseAfter(dict):
    """Model behaviour that answers normally for a number of calls, then refuses like an empty account."""

    def __init__(self, calls: int) -> None:
        super().__init__()
        self.left = calls

    def get(self, model, default=None):
        if self.left <= 0:
            return RuntimeError(REFUSAL)
        self.left -= 1
        return (0.4, 0.05)


def drain_budget(raw: dict) -> dict:
    budget = dict(raw["budget"])
    budget.update(credit_floor_usd=0.0, funded_usd=5.0)
    return budget


def test_the_recorded_funding_is_what_the_balance_is_read_against(env, workspace, monkeypatch, base_config):
    workspace.write(budget=drain_budget(base_config))
    asked: list = []

    def fake(api_key, *args, **kwargs):
        asked.append(kwargs.get("funded_usd"))
        return guard.Balance(5.0, "funded", "", 0.0)

    monkeypatch.setattr(guard, "fetch_openrouter_balance", fake)
    MetaculusClient.questions = {MAIN: [Question(1)], MINI: []}
    assert workspace.run() == 0
    assert asked == [5.0, 5.0]  # before and after the run


def test_a_cutoff_stops_the_run_records_it_and_raises_no_alarm(env, workspace, balances, base_config):
    workspace.write(budget=drain_budget(base_config), tracks=[base_config["tracks"][0]])
    balances.script[:] = [guard.Balance(3.0, "funded", "", 2.0), guard.Balance(2.1, "funded", "", 2.9)]
    # 3 calls a question (1 research + 2 forecasts): questions 1 and 2 finish, question 3 is refused
    GeneralLlm.behaviour = RefuseAfter(7)
    MetaculusClient.questions = {MAIN: [Question(i) for i in range(1, 6)]}
    assert workspace.run() == 0

    (record,) = workspace.records()
    assert record["cutoff_hit"] is True and record["provider_refusals"] >= 1 and record["alarm"] is False
    assert record["questions_ok"] == 2 and record["questions_failed"] == 1
    assert record["tracks"][0]["questions_not_started"] == 2
    assert record["provider_cost_usd"] == pytest.approx(0.9)
    started = {prompt for _model, prompt in GeneralLlm.calls if prompt.startswith("forecast")}
    assert started == {"forecast 1", "forecast 2"}            # nothing after the refusal was paid for
    assert sorted(q for q, _p in MetaculusClient.published) == [1, 2]


def test_after_a_cutoff_the_bot_waits_quietly_then_resumes_when_funded(env, workspace, balances, base_config):
    workspace.write(budget=drain_budget(base_config), tracks=[base_config["tracks"][0]])
    balances.script[:] = [guard.Balance(0.5, "funded", "", 4.5), guard.Balance(0.45, "funded", "", 4.55)]
    GeneralLlm.behaviour = RefuseAfter(1)  # research goes through, every forecast is refused
    MetaculusClient.questions = {MAIN: [Question(1), Question(2)]}
    assert workspace.run() == 0
    assert workspace.records()[0]["cutoff_hit"] is True and workspace.records()[0]["questions_ok"] == 0

    # Twenty minutes later nothing has changed: no model call, no question listing, one quiet line.
    GeneralLlm.calls.clear()
    MetaculusClient.list_calls.clear()
    balances.script[:] = [guard.Balance(0.45, "funded", "", 4.55)]
    balances.seen.clear()
    assert workspace.run(now=NOW + timedelta(minutes=20)) == 0
    assert workspace.run(now=NOW + timedelta(minutes=40)) == 0
    records = workspace.records()
    assert [r["kind"] for r in records] == ["run", "skip"] and "waiting" in records[1]["reason"]
    assert GeneralLlm.calls == [] and MetaculusClient.list_calls == []

    # Credits arrive (funded_usd raised after a purchase): the next run forecasts again.
    balances.script[:] = [guard.Balance(94.0, "funded", "", 4.55), guard.Balance(93.3, "funded", "", 5.25)]
    balances.seen.clear()
    GeneralLlm.behaviour = {}
    assert workspace.run(now=NOW + timedelta(hours=2)) == 0
    assert workspace.records()[-1]["kind"] == "run" and workspace.records()[-1]["questions_ok"] == 2
    assert workspace.records()[-1]["cutoff_hit"] is False


def test_a_refusal_hidden_inside_a_question_that_still_succeeded_is_caught(env, workspace, balances, base_config):
    workspace.write(budget=drain_budget(base_config), tracks=[base_config["tracks"][0]])
    balances.script[:] = [guard.Balance(100.0, "funded", "", 0.0), guard.Balance(99.5, "funded", "", 0.5)]
    GeneralLlm.behaviour = {"model-a": (0.3, 0.05), "model-b": RuntimeError(REFUSAL)}
    MetaculusClient.questions = {MAIN: [Question(1)]}
    assert workspace.run() == 0
    (record,) = workspace.records()
    assert record["questions_ok"] == 1 and record["questions_failed"] == 0
    assert record["cutoff_hit"] is True and record["provider_refusals"] == 2
    assert any(e.startswith("partial: ") for e in record["errors"])
    assert record["tracks"][0]["forecasts_by_model"] == {"model-a": {"ok": 2, "failed": 0}, "model-b": {"ok": 0, "failed": 2}}


def test_a_clean_run_counts_forecasts_per_model(env, workspace, balances):
    MetaculusClient.questions = {MAIN: [Question(1), Question(2)], MINI: [Question(11)]}
    assert workspace.run() == 0
    main, minibench = workspace.records()[0]["tracks"]
    assert main["forecasts_by_model"] == {"model-a": {"ok": 4, "failed": 0}, "model-b": {"ok": 4, "failed": 0}}
    assert minibench["forecasts_by_model"] == {"model-a": {"ok": 1, "failed": 0}, "model-b": {"ok": 1, "failed": 0}}
    assert main["questions_not_started"] == 0 and workspace.records()[0]["cutoff_hit"] is False


@pytest.mark.parametrize("balance", [3.0, 20.0, 100.0])
def test_questions_are_forecast_in_batches_that_fit_the_balance(env, workspace, balances, base_config, monkeypatch, balance):
    workspace.write(budget=drain_budget(base_config), tracks=[base_config["tracks"][0]])
    balances.script[:] = [guard.Balance(balance, "funded", "", 0.0)]
    MetaculusClient.questions = {MAIN: [Question(i) for i in range(1, 14)]}
    sizes: list = []
    original = agent_bot.PanelBot.forecast_questions

    async def recording(self, questions, return_exceptions=False):
        sizes.append(len(questions))
        return await original(self, questions, return_exceptions)

    monkeypatch.setattr(agent_bot.PanelBot, "forecast_questions", recording)
    assert workspace.run() == 0
    (record,) = workspace.records()
    track = record["tracks"][0]
    cfg = guard.parse_config({**base_config, "budget": drain_budget(base_config), "tracks": [base_config["tracks"][0]]})
    batch = guard.parallel_questions(cfg, balance, track["predictions"], track["selected"])
    assert max(sizes) == batch and sum(sizes) == track["selected"] == record["questions_ok"]
    assert all(size == batch for size in sizes[:-1])
    if balance == 3.0:
        assert batch == 1                                     # a small balance means one question at a time


def test_models_given_with_settings_are_built_with_them_and_names_stay_names(base_config):
    assert agent_bot.build_llm("parser-model") == "parser-model" and agent_bot.build_llm(None) is None
    built = agent_bot.build_llm({"model": "parser-model", "max_tokens": 4000})
    assert isinstance(built, GeneralLlm) and built.model == "parser-model" and built.kwargs == {"max_tokens": 4000}
    raw = copy.deepcopy(base_config)
    raw["parser"] = {"model": "parser-model", "max_tokens": 4000}
    raw["panel"][0]["settings"] = {"max_tokens": 24000}
    bot = agent_bot.build_bot(guard.parse_config(raw), predictions=2, publish=False, skip_forecasted=False)
    assert bot.get_llm("parser", "llm").kwargs == {"max_tokens": 4000}
    assert bot.get_llm("summarizer") == "summary-model"
    assert bot._panel[0].kwargs == {"max_tokens": 24000}
