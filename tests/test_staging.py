"""Tests for staged funding, worst-case planning, the provider cutoff and the cutoff check."""
from __future__ import annotations

import copy
import io
import json
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent import guard, probe

NOW = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)
REPO_ROOT = Path(__file__).resolve().parent.parent


def cfg_from(raw: dict, **changes) -> guard.Config:
    raw = copy.deepcopy(raw)
    raw.update(changes)
    return guard.parse_config(raw)


def with_budget(raw: dict, **changes) -> dict:
    raw = copy.deepcopy(raw)
    raw["budget"].update(changes)
    return raw


class FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def key_reply(**data):
    return {"data": data}


def opener_for(key_payloads, chat=None):
    """Answers the key endpoint from a list (the last one repeats) and the chat endpoint from `chat`."""
    seen = {"key": 0, "chat": []}

    def opener(request, timeout=None):
        assert request.get_header("Authorization") == "Bearer sk-test"
        if request.full_url == guard.OPENROUTER_KEY_URL:
            payload = key_payloads[min(seen["key"], len(key_payloads) - 1)]
            seen["key"] += 1
            return FakeResponse(json.dumps(payload).encode())
        assert request.full_url == probe.CHAT_URL
        seen["chat"].append(json.loads(request.data.decode()))
        if isinstance(chat, Exception):
            raise chat
        return FakeResponse(json.dumps(chat).encode())

    opener.seen = seen
    return opener


# ----------------------------------------------------------------------- config


def test_shipped_config_starts_unfunded_and_caps_every_model():
    raw = json.loads((REPO_ROOT / "config" / "bot_config.json").read_text())
    cfg = guard.parse_config(raw)
    assert cfg.live is False
    assert cfg.funded_usd == 0.0  # nothing can run until a purchase is recorded here
    for member in cfg.panel:
        assert member.settings["max_tokens"] > 0
        assert member.worst_case_cost_per_call >= member.est_cost_per_forecast
    for spec in (cfg.parser, cfg.summarizer, cfg.researcher_fallback):
        assert isinstance(spec, dict) and spec["max_tokens"] > 0
    # an unfunded bot never forecasts, whatever the key's own limit says
    assert guard.plan_run(cfg_from(raw, live=True), 0.0, NOW, "tournament").go is False
    assert guard.plan_run(cfg, 0.0, NOW, "test_questions").go is False


def test_shipped_worst_cases_match_list_prices():
    cfg = guard.load_config(REPO_ROOT / "config" / "bot_config.json")
    gemini, gpt = cfg.panel
    # output cap times list price per token, plus up to 10k input tokens
    assert gemini.worst_case_cost_per_call == pytest.approx(24000 * 12e-6 + 10000 * 2e-6, abs=0.011)
    assert gpt.worst_case_cost_per_call == pytest.approx(8000 * 30e-6 + 10000 * 5e-6, abs=0.011)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r["budget"].update(funded_usd=-1), "funded_usd"),
        (lambda r: r["budget"].update(funded_usd=True), "funded_usd"),
        (lambda r: r["budget"].update(funded_usd="5"), "funded_usd"),
        (lambda r: r["budget"].update(inflight_share=0), "inflight_share"),
        (lambda r: r["budget"].update(inflight_share=1.5), "inflight_share"),
        (lambda r: r["panel"][0].update(worst_case_cost_per_call=0.01), "below est_cost_per_forecast"),
        (lambda r: r["panel"][0].update(worst_case_cost_per_call="lots"), "worst_case_cost_per_call"),
        (lambda r: r.update(parser={"max_tokens": 10}), "'parser' needs a 'model' name"),
        (lambda r: r.update(summarizer=""), "'summarizer' is empty"),
        (lambda r: r.update(worst_case_overhead_per_question=-0.1), "worst_case_overhead_per_question"),
    ],
)
def test_bad_staging_config_is_refused(base_config, mutate, message):
    raw = copy.deepcopy(base_config)
    mutate(raw)
    with pytest.raises(guard.ConfigError, match=message):
        guard.parse_config(raw)


def test_old_style_config_still_loads(base_config):
    cfg = guard.parse_config(base_config)
    assert cfg.funded_usd is None and cfg.parser == "parser-model" and cfg.inflight_share == 0.5
    assert cfg.panel[0].worst_case_cost_per_call is None


# ----------------------------------------------------------------------- worst case and batching


def test_worst_case_uses_config_figures_and_falls_back_to_four_times_the_estimate(base_config):
    plain = guard.parse_config(base_config)
    assert guard.worst_case_cost_per_question(plain, 4) == pytest.approx(4 * 0.08 * 4 + 0.04 * 4)
    raw = copy.deepcopy(base_config)
    raw["panel"][0]["worst_case_cost_per_call"] = 0.31
    raw["panel"][1]["worst_case_cost_per_call"] = 0.29
    raw["worst_case_overhead_per_question"] = 0.15
    cfg = guard.parse_config(raw)
    assert guard.worst_case_cost_per_question(cfg, 4) == pytest.approx(2 * 0.31 + 2 * 0.29 + 0.15)
    assert guard.worst_case_cost_per_question(cfg, 3) == pytest.approx(2 * 0.31 + 0.29 + 0.15)
    with pytest.raises(ValueError):
        guard.worst_case_cost_per_question(cfg, 0)


def test_questions_run_one_at_a_time_on_a_small_balance_and_together_on_a_large_one():
    cfg = guard.load_config(REPO_ROOT / "config" / "bot_config.json")
    assert guard.parallel_questions(cfg, 5.0, 4, cap=6) == 1      # $2.50 of headroom, $1.35 a question
    assert guard.parallel_questions(cfg, 0.4, 2, cap=6) == 1      # never less than one
    assert guard.parallel_questions(cfg, 100.0, 4, cap=15) == 15  # $50 of headroom covers the whole run
    assert guard.parallel_questions(cfg, 20.0, 4, cap=15) == 7
    assert guard.parallel_questions(cfg, None, 4, cap=15) == 15   # balance unknown: the per-run caps apply
    assert guard.parallel_questions(cfg, 50.0, 4, cap=0) == 1


# ----------------------------------------------------------------------- balance with staged funding


def read(payload, funded):
    return guard.fetch_openrouter_balance("sk-test", opener_for([payload]), funded_usd=funded)


def test_balance_is_the_smaller_of_the_key_limit_and_the_credits_bought():
    # $5 bought, key allows $100: the $5 is what binds
    first = read(key_reply(limit=100, limit_remaining=98.2, usage=1.8), funded=5.0)
    assert first == guard.Balance(pytest.approx(3.2), "funded", "", 1.8)
    # $150 bought (hypothetically), key allows $100: the key binds
    second = read(key_reply(limit=100, limit_remaining=40.0, usage=60.0), funded=150.0)
    assert (second.remaining, second.source) == (40.0, "key_limit")


def test_balance_never_goes_below_zero_and_reports_usage():
    spent = read(key_reply(limit=100, limit_remaining=94.9, usage=5.1), funded=5.0)
    assert spent.remaining == 0.0 and spent.source == "funded" and spent.usage == 5.1


def test_unfunded_means_zero_even_when_the_key_would_allow_spending():
    assert read(key_reply(limit=100, limit_remaining=100, usage=0), funded=0.0).remaining == 0.0


def test_funding_without_a_usage_figure_is_an_error_not_a_guess():
    balance = read(key_reply(limit=100, limit_remaining=100), funded=5.0)
    assert balance.remaining is None and balance.source == "error"


def test_a_budget_that_blocks_first_is_respected():
    payload = key_reply(limit=100, limit_remaining=90, usage=10, effective_budget={"remaining_usd": 1.25})
    assert read(payload, funded=50.0) == guard.Balance(1.25, "budget", "", 10.0)
    odd = key_reply(limit=100, limit_remaining=90, usage=10, effective_budget={"remaining_usd": True})
    assert read(odd, funded=None).source == "key_limit"


def test_without_funding_the_key_limit_still_works_as_before():
    balance = read(key_reply(limit=100, limit_remaining=61.25, usage=38.75), funded=None)
    assert (balance.remaining, balance.source, balance.usage) == (61.25, "key_limit", 38.75)


# ----------------------------------------------------------------------- the provider cutoff


REAL_LOOKING_REFUSALS = [
    'litellm.APIError: APIError: OpenrouterException - {"error":{"message":"This request requires more credits, or fewer max_tokens. '
    'You requested up to 24000 tokens, but can only afford 1532.","code":402,"metadata":{"reason":"weight_exceeds_budget"}}}',
    'OpenrouterException - {"error":{"message":"Insufficient credits. Add more using https://openrouter.ai/settings/credits","code":402}}',
    '{"error":{"code":402,"message":"Key limit exceeded","metadata":{"limit_source":"openrouter_key_limit"}}}',
    "RuntimeError: All predictions failed: ['Error code: 402 - not enough credits']",
]
NOT_REFUSALS = [
    "TimeoutError: model took longer than 300 seconds",
    "ValueError: could not parse a probability from the answer",
    "RateLimitError: 429 too many requests",
    "question 402 has no resolution criteria",
    "",
]


@pytest.mark.parametrize("text", REAL_LOOKING_REFUSALS)
def test_refusals_for_lack_of_credits_are_recognised(text):
    assert guard.is_provider_refusal(text) is True
    assert guard.is_provider_refusal(RuntimeError(text)) is True


@pytest.mark.parametrize("text", NOT_REFUSALS)
def test_other_errors_are_not_mistaken_for_a_cutoff(text):
    assert guard.is_provider_refusal(text) is False


def run_record(cfg, before, after, errors=(), usage=(None, None)):
    plan = guard.plan_run(cfg, max(before, 1.0), NOW, "tournament")
    return guard.build_run_record(
        now=NOW, mode="tournament", plan=plan,
        balance_before=guard.Balance(before, "funded", "", usage[0]),
        balance_after=guard.Balance(after, "funded", "", usage[1]),
        track_results=[{"name": "main", "predictions": 2, "raw_est_cost_per_question": 0.2,
                        "est_cost_per_question": 0.2, "questions_ok": 1, "questions_failed": 1}],
        tracked_cost_usd=0.3, models=["model-a", "model-b"], errors=errors,
    )


def test_run_record_carries_the_providers_own_cost_and_the_cutoff(base_config):
    cfg = guard.parse_config(with_budget(base_config, credit_floor_usd=0.0))
    clean = run_record(cfg, 5.0, 4.6, usage=(0.0, 0.4123456))
    assert clean["provider_cost_usd"] == 0.4123 and clean["usage_after"] == 0.4123456
    assert clean["provider_refusals"] == 0 and clean["cutoff_hit"] is False
    cut = run_record(cfg, 0.5, 0.2, errors=[REAL_LOOKING_REFUSALS[0], "TimeoutError: slow", REAL_LOOKING_REFUSALS[1]])
    assert cut["provider_refusals"] == 2 and cut["cutoff_hit"] is True and cut["provider_cost_usd"] is None
    json.dumps(cut)


def test_observed_costs_prefer_the_providers_figure(base_config):
    cfg = guard.parse_config(with_budget(base_config, credit_floor_usd=0.0))
    record = run_record(cfg, 5.0, 4.9, usage=(1.0, 1.9))  # balance fell $0.10 but the key's usage rose $0.90
    record["questions_attempted"] = 6
    record["raw_est_total_usd"] = 0.6
    assert guard.observed_cost_ratio([record]) == pytest.approx(1.5)


def test_after_a_cutoff_the_bot_waits_and_resumes_when_credits_arrive(base_config):
    cfg = guard.parse_config(with_budget(base_config, credit_floor_usd=0.0))
    cut = run_record(cfg, 0.6, 0.45, errors=[REAL_LOOKING_REFUSALS[1]])
    assert guard.waiting_for_credits([cut], 0.45) is True
    waiting = guard.plan_run(cfg, 0.45, NOW, "tournament", [cut])
    assert waiting.go is False and "waiting" in waiting.reason
    assert guard.plan_run(cfg, 0.45, NOW, "test_questions", [cut]).go is False
    # new credits lift the balance: the bot starts again without anyone telling it to
    assert guard.waiting_for_credits([cut], 95.0) is False
    assert guard.plan_run(cfg, 95.0, NOW, "tournament", [cut]).go is True
    # a later clean run clears the wait; skip lines in between do not
    clean = run_record(cfg, 95.0, 94.0)
    skip = guard.build_skip_record(now=NOW, mode="tournament", reason="x", balance=guard.Balance(0.45, "funded"))
    assert guard.waiting_for_credits([cut, clean], 0.45) is False
    assert guard.waiting_for_credits([cut, skip, skip], 0.45) is True
    assert guard.waiting_for_credits([cut], None) is False and guard.waiting_for_credits([], 0.1) is False


# ----------------------------------------------------------------------- stage one, start to finish


def test_stage_one_five_dollars_covers_the_test_then_drains_on_real_questions():
    """$5 bought, floor $0: the test run fits, the leftover goes to tournament questions at the minimum
    forecast count, and planning stops by itself when less than one question's cost is left."""
    raw = json.loads((REPO_ROOT / "config" / "bot_config.json").read_text())
    raw["budget"]["funded_usd"] = 5.0
    cfg = guard.parse_config(raw)

    test_plan = guard.plan_run(cfg, 5.0, NOW, "test_questions")
    assert test_plan.go and test_plan.tracks[0].predictions == 4 and test_plan.max_questions == 6
    test_cost = 6 * test_plan.tracks[0].est_cost_per_question
    assert test_cost == pytest.approx(2.136) and test_plan.max_cost_usd == 5.0

    live = guard.parse_config({**raw, "live": True})
    balance = 5.0 - test_cost
    plan = guard.plan_run(live, balance, NOW, "tournament")
    assert plan.go and "stretching credits" in plan.reason
    assert [t.predictions for t in plan.tracks] == [2, 2]
    per_question = plan.tracks[0].est_cost_per_question
    assert per_question == pytest.approx(0.198)
    questions = 0
    while guard.plan_run(live, balance, NOW, "tournament").go:
        balance -= per_question
        questions += 1
    assert questions == 14 and 0 <= balance < per_question
    assert "credits at floor" in guard.plan_run(live, balance, NOW, "tournament").reason


# ----------------------------------------------------------------------- the cutoff check


def test_probe_asks_for_more_than_the_balance_could_pay_for():
    for remaining in (0.0, 0.37, 1.0, 2.9):
        tokens = probe.probe_max_tokens(remaining)
        assert tokens * probe.OUTPUT_USD_PER_TOKEN >= remaining * 1.05 + 0.10 - 1e-9
        assert tokens <= probe.MODEL_MAX_TOKENS
    assert probe.probe_max_tokens(3.6) is None  # too much left for a free check
    assert probe.highest_probeable_balance() == pytest.approx(3.56, abs=0.01)


def test_probe_records_a_refusal_and_spends_nothing():
    refusal = urllib.error.HTTPError(
        probe.CHAT_URL, 402, "Payment Required", None,
        io.BytesIO(json.dumps({"error": {"message": "This request requires more credits, or fewer max_tokens.",
                                         "code": 402, "metadata": {"reason": "weight_exceeds_budget"}}}).encode()),
    )
    opener = opener_for([key_reply(limit=100, limit_remaining=97.2, usage=2.8)], chat=refusal)
    record = probe.run_probe("sk-test", 5.0, now=NOW, opener=opener)
    assert record["outcome"] == "refused" and record["http_status"] == 402
    assert record["reason"] == "weight_exceeds_budget" and record["balance_before"] == pytest.approx(2.2)
    (sent,) = opener.seen["chat"]
    assert sent["model"] == probe.PROBE_MODEL and sent["max_tokens"] == record["max_tokens"]
    assert record["worst_case_usd"] > record["balance_before"]
    assert "sk-test" not in json.dumps(record)


def test_probe_reports_honestly_when_the_call_is_served():
    served = {"choices": [{"message": {"content": "OK"}}], "usage": {"completion_tokens": 3}}
    opener = opener_for(
        [key_reply(limit=100, limit_remaining=97.5, usage=2.5), key_reply(limit=100, limit_remaining=97.4995, usage=2.5005)],
        chat=served,
    )
    record = probe.run_probe("sk-test", 5.0, now=NOW, opener=opener)
    assert record["outcome"] == "served" and record["cost_usd"] == pytest.approx(0.0005)
    assert "not shown" in record["why"]


def test_probe_handles_a_402_wrapped_in_a_200_and_other_statuses():
    wrapped = {"error": {"message": "Insufficient credits", "code": 402, "metadata": {"limit_source": "openrouter_credits"}}}
    record = probe.run_probe("sk-test", 5.0, now=NOW, opener=opener_for([key_reply(limit_remaining=99, usage=4.5)], chat=wrapped))
    assert record["outcome"] == "refused" and record["limit_source"] == "openrouter_credits"
    busy = urllib.error.HTTPError(probe.CHAT_URL, 503, "busy", None, io.BytesIO(b"not json"))
    record = probe.run_probe("sk-test", 5.0, now=NOW, opener=opener_for([key_reply(limit_remaining=99, usage=4.5)], chat=busy))
    assert record["outcome"] == "error" and record["http_status"] == 503


def test_probe_does_not_run_without_a_key_a_balance_or_a_low_enough_balance():
    assert probe.run_probe(None, 5.0, now=NOW)["outcome"] == "not_run"
    unreadable = probe.run_probe("sk-test", 5.0, now=NOW, opener=opener_for([{"data": {"limit_remaining": 9}}]))
    assert unreadable["outcome"] == "not_run" and "balance" in unreadable["why"]
    opener = opener_for([key_reply(limit=100, limit_remaining=100, usage=0.0)])
    rich = probe.run_probe("sk-test", 5.0, now=NOW, opener=opener)
    assert rich["outcome"] == "not_run" and opener.seen["chat"] == []


def test_probe_command_writes_one_record(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    records = tmp_path / "records" / "probes.jsonl"
    assert probe.main(["--config", str(REPO_ROOT / "config" / "bot_config.json"), "--records", str(records)]) == 0
    (record,) = guard.read_records(records)
    assert record["kind"] == "probe" and record["outcome"] == "not_run"
    assert "not_run" in capsys.readouterr().out
