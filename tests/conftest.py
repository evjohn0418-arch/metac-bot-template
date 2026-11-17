"""Stand-ins for the forecasting framework, so the bot's own logic can be tested offline.

The real framework (forecasting-tools) and Metaculus's template (main.py) are not installed
where these tests run. The classes below copy the parts of their behaviour that agent_bot.py
relies on, as read from their source on 2026-11-14:

- ForecastBot.forecast_questions gathers one task per question.
- Each question runs research once, then `predictions_per_research_report` calls to
  `_make_prediction`, gathered concurrently, each of which asks `get_llm("default", "llm")`.
- Cost managers nest through a ContextVar list; a limit is checked before each model call.

They are a model of the framework, not the framework. The first real proof is the Test Bot
workflow on GitHub.
"""
from __future__ import annotations

import asyncio
import statistics
import sys
import types
from contextvars import ContextVar
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ----------------------------------------------------------------------- forecasting_tools


class HardLimitExceededError(Exception):
    pass


class MonetaryCostManager:
    _active: ContextVar[list] = ContextVar("_active_cost_managers", default=[])

    def __init__(self, hard_limit: float = 0) -> None:
        self.hard_limit = hard_limit
        self.current_usage = 0.0

    def __enter__(self):
        managers = self._active.get().copy()
        managers.append(self)
        self._active.set(managers)
        return self

    def __exit__(self, *exc) -> None:
        managers = self._active.get().copy()
        managers.remove(self)
        self._active.set(managers)

    @classmethod
    def raise_error_if_limit_would_be_reached(cls, amount: float = 0) -> None:
        for manager in cls._active.get():
            if manager.hard_limit != 0 and manager.hard_limit - manager.current_usage < amount:
                raise HardLimitExceededError(f"limit {manager.hard_limit} would be exceeded")

    @classmethod
    def increase_current_usage_in_parent_managers(cls, amount: float) -> None:
        for manager in cls._active.get():
            manager.current_usage += amount


class GeneralLlm:
    """Fake model. `behaviour[model]` gives (answer, cost) or an Exception to raise."""

    behaviour: dict = {}
    calls: list = []

    def __init__(self, model: str, **kwargs) -> None:
        self.model = model
        self.kwargs = kwargs

    def to_dict(self) -> dict:
        return {"model": self.model, **self.kwargs}

    async def invoke(self, prompt: str) -> float:
        await asyncio.sleep(0)  # let other tasks interleave, as real network calls do
        outcome = GeneralLlm.behaviour.get(self.model, (0.5, 0.01))
        if isinstance(outcome, Exception):
            raise outcome
        answer, cost = outcome
        MonetaryCostManager.raise_error_if_limit_would_be_reached(cost)
        await asyncio.sleep(0)
        MonetaryCostManager.increase_current_usage_in_parent_managers(cost)
        GeneralLlm.calls.append((self.model, prompt))
        return answer


class ReasonedPrediction:
    def __init__(self, prediction_value, reasoning: str) -> None:
        self.prediction_value = prediction_value
        self.reasoning = reasoning


class ForecastReport:
    def __init__(self, question, prediction, explanation, price_estimate, errors, forecasts) -> None:
        self.question = question
        self.prediction = prediction
        self.explanation = explanation
        self.price_estimate = price_estimate
        self.errors = errors
        self.forecasts = forecasts  # the individual ReasonedPrediction objects


class MetaculusClient:
    CURRENT_AI_COMPETITION_ID = "fall-main-id"
    CURRENT_MINIBENCH_ID = "minibench"
    questions: dict = {}   # tournament -> list of questions, or an Exception to raise
    published: list = []
    list_calls: list = []

    def get_all_open_questions_from_tournament(self, tournament):
        MetaculusClient.list_calls.append(tournament)
        result = MetaculusClient.questions.get(tournament, [])
        if isinstance(result, Exception):
            raise result
        return list(result)


forecasting_tools = types.ModuleType("forecasting_tools")
forecasting_tools.GeneralLlm = GeneralLlm
forecasting_tools.MetaculusClient = MetaculusClient
forecasting_tools.MonetaryCostManager = MonetaryCostManager
forecasting_tools.ReasonedPrediction = ReasonedPrediction
forecasting_tools.ForecastReport = ForecastReport
forecasting_tools.HardLimitExceededError = HardLimitExceededError
sys.modules["forecasting_tools"] = forecasting_tools


# ----------------------------------------------------------------------- main (the template bot)


class FallTemplateBot2026:
    def __init__(
        self,
        *,
        research_reports_per_question: int = 1,
        predictions_per_research_report: int = 1,
        use_research_summary_to_forecast: bool = False,
        publish_reports_to_metaculus: bool = False,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions: bool = False,
        llms=None,
        extra_metadata_in_explanation: bool = False,
        metaculus_client=None,
    ) -> None:
        self.research_reports_per_question = research_reports_per_question
        self.predictions_per_research_report = predictions_per_research_report
        self.publish_reports_to_metaculus = publish_reports_to_metaculus
        self.skip_previously_forecasted_questions = skip_previously_forecasted_questions
        self.extra_metadata_in_explanation = extra_metadata_in_explanation
        self._llms = llms or {}
        self.metaculus_client = metaculus_client or MetaculusClient()
        self._research_lock = asyncio.Semaphore(1)

    def get_llm(self, purpose: str = "default", guarantee_type=None):
        if purpose not in self._llms:
            raise ValueError(f"Unknown llm requested from llm dict for purpose: '{purpose}'")
        llm = self._llms[purpose]
        if llm is None:
            raise ValueError(f"LLM is undefined for purpose: {purpose}")
        if guarantee_type is None:
            return llm
        if guarantee_type == "llm":
            return llm if isinstance(llm, GeneralLlm) else GeneralLlm(model=llm)
        if guarantee_type == "string_name":
            return llm if isinstance(llm, str) else llm.model
        raise ValueError(f"Unknown guarantee_type: {guarantee_type}")

    def make_llm_dict(self) -> dict:
        return {k: (v.to_dict() if isinstance(v, GeneralLlm) else v) for k, v in self._llms.items()}

    async def run_research(self, question) -> str:
        async with self._research_lock:
            researcher = self.get_llm("researcher")
            if isinstance(researcher, GeneralLlm):
                await researcher.invoke(f"research {question.id_of_question}")
            return f"research for {question.id_of_question}"

    async def forecast_questions(self, questions, return_exceptions: bool = False):
        if self.skip_previously_forecasted_questions:
            questions = [q for q in questions if not q.already_forecasted]
        return await asyncio.gather(
            *[self._run_individual_question(q) for q in questions], return_exceptions=return_exceptions
        )

    async def _run_individual_question(self, question):
        with MonetaryCostManager() as cost_manager:
            research = await self.run_research(question)
            results = await asyncio.gather(
                *[self._make_prediction(question, research) for _ in range(self.predictions_per_research_report)],
                return_exceptions=True,
            )
            valid = [r for r in results if not isinstance(r, BaseException)]
            errors = [str(r) for r in results if isinstance(r, BaseException)]
            if not valid:
                raise RuntimeError(f"All predictions failed: {errors}")
            aggregated = statistics.median(p.prediction_value for p in valid)
            cost = cost_manager.current_usage
        explanation = "\n".join(p.reasoning for p in valid)
        report = ForecastReport(question, aggregated, explanation, cost, errors, valid)
        if self.publish_reports_to_metaculus:
            MetaculusClient.published.append((question.id_of_question, aggregated))
        return report

    async def _make_prediction(self, question, research: str):
        llm = self.get_llm("default", "llm")
        named = self.get_llm("default", "string_name")
        answer = await llm.invoke(f"forecast {question.id_of_question}")
        # the model object and its name must still agree after the awaits above
        assert named == llm.model
        return ReasonedPrediction(prediction_value=answer, reasoning=f"reasoned by {llm.model}")

    @classmethod
    def log_report_summary(cls, reports, raise_errors: bool = True) -> None:
        failures = [r for r in reports if isinstance(r, BaseException)]
        if failures and raise_errors:
            raise RuntimeError(f"{len(failures)} errors occurred while forecasting")


main_module = types.ModuleType("main")
main_module.FallTemplateBot2026 = FallTemplateBot2026
sys.modules["main"] = main_module


# ----------------------------------------------------------------------- bot_helpers

bot_helpers = types.ModuleType("bot_helpers")
bot_helpers.silence_noisy_dependencies = lambda: None
bot_helpers.print_startup_banner = lambda run_mode, will_publish: None
bot_helpers.print_run_summary_banner = lambda reports, will_publish, tournament_url=None: None
sys.modules["bot_helpers"] = bot_helpers


# ----------------------------------------------------------------------- shared fixtures


class Question:
    def __init__(self, qid, forecasted=False, close=None, spot=None, cp_reveal=None):
        self.id_of_question = qid
        self.already_forecasted = forecasted
        self.close_time = close
        self.cp_reveal_time = cp_reveal
        self.api_json = {"question": {"spot_scoring_time": spot}}


class BinaryQuestion(Question):
    pass


class NumericQuestion(Question):
    pass


class MultipleChoiceQuestion(Question):
    pass


@pytest.fixture(autouse=True)
def _reset_fakes():
    GeneralLlm.behaviour = {}
    GeneralLlm.calls = []
    MetaculusClient.questions = {}
    MetaculusClient.published = []
    MetaculusClient.list_calls = []
    yield


@pytest.fixture
def base_config() -> dict:
    """A copy of the shipped config's shape with easy numbers."""
    return {
        "live": True,
        "panel": [
            {"model": "model-a", "est_cost_per_forecast": 0.08, "settings": {"reasoning_effort": "high"}},
            {"model": "model-b", "est_cost_per_forecast": 0.08, "settings": {}},
        ],
        "tracks": [
            {"name": "main", "tournament": None, "enabled": True, "predictions_per_question": 4,
             "last_open_date": "2026-12-12", "expected_questions_per_day": 4.5},
            {"name": "minibench", "tournament": None, "enabled": True, "predictions_per_question": 2,
             "last_open_date": "2026-12-27", "expected_questions_per_day": 4.3},
        ],
        "min_predictions_per_question": 2,
        "researcher": {"prefer_asknews_when_keys_present": True, "fallback": {"model": "search-model", "temperature": 0.1}},
        "parser": "parser-model",
        "summarizer": "summary-model",
        "est_overhead_per_question": 0.04,
        "budget": {"credit_floor_usd": 2.0, "max_questions_per_run": 15, "max_cost_per_run_usd": 10.0, "test_question_limit": 6},
        "skip_after_spot_scoring_time": True,
    }
