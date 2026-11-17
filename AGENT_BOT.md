# How this bot works

This repository is a fork of Metaculus's official bot template. `main.py` is the
template's bot, unchanged. `agent_bot.py` runs it with three additions.

## What is added

1. **A panel of models.** Each question gets several forecasts, rotated across the
   models listed in `config/bot_config.json`. The framework takes the median. Every
   forecast's reasoning starts with the model that wrote it.
2. **A spend plan.** Before each run the bot asks OpenRouter how much its key has
   spent and may still spend. It then plans inside that amount: how many questions to
   take, how many forecasts each gets, and how many questions to work on at once. If
   credits run short it trims forecasts per question so the bot keeps covering
   questions to the end of the season. See `agent/guard.py`.
3. **Records.** Every run that does something appends one line to
   `records/runs.jsonl`: questions seen, forecast and skipped, forecasts per model,
   OpenRouter's own usage figure before and after, and any errors. A review job saves
   the bot's scores and leaderboard standing under `records/reviews/` twice a week.

## What limits the spending

The limit is not in this code. It is the credit limit on the OpenRouter key and the
prepaid balance behind it, with auto top-up off. When either runs out, OpenRouter
refuses the bot's calls. The spend plan only makes the bot slow down before that.

Credits are bought in stages. `budget.funded_usd` in the config is the total bought so
far, and the bot treats what is left of it as its balance. A wrong figure there can make
the plan wrong; it cannot make OpenRouter serve a call the account cannot pay for.

Every model call carries a `max_tokens` cap. OpenRouter prices a request at its worst
case before serving it, so the caps keep small balances usable.

When OpenRouter refuses a call for lack of credits, the run stops, records
`cutoff_hit`, and later runs wait until the balance has grown. The **Check the spending
cutoff** job tests the same thing on purpose: it sends one request that asks for more
than the credits left could pay for and records whether it was refused
(`records/probes.jsonl`).

## Research and forecasting

Research uses AskNews when its keys are set, otherwise a search-enabled model through
OpenRouter. Prompts, parsing and aggregation are the template's.

## No human in the loop

Forecasts are produced and submitted by the scheduled job. Nobody previews a forecast
on an open question or edits one. Reviews read resolved questions only. Changes to the
bot are commits to this repository, so the history shows what changed and when.

## Switches

- `config/bot_config.json` → `"live"`: the scheduled job does nothing until this is `true`.
- `tracks[].enabled`: turn a tournament on or off.
- `panel`, `predictions_per_question`, `budget`: models, forecasts per question, caps.
- `budget.funded_usd`: credits bought so far. At `0` nothing can run.

## Workflows

- **Test Bot**: forecasts on a few questions in Metaculus's bot-testing-area.
- **Forecast on new AI tournament questions** (every 20 minutes): the live job.
- **Review recent forecasts** (Monday and Thursday): read-only scoring and standing.
- **Check the spending cutoff**: one request, no forecasts.

Test Bot, the review and the cutoff check can each be started by hand, or by a commit
that changes `config/run_test.txt`, `config/run_review.txt` or `config/run_probe.txt`.

## Tests

`tests/` checks the spend plan, staged funding, the cutoff handling, question selection,
records and the panel logic against stand-ins for the framework. Run with `python -m pytest tests` (needs only
`pytest`). They do not call Metaculus or any model; the Test Bot workflow does that.
