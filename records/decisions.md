# Decisions

Newest first. Each entry: date, what changed, the evidence, who decided.

## 2026-11-24: GitHub disabled the fork's workflows (agent)

- No runs after Nov 24 00:04 UTC. The Actions page said GitHub had disabled workflows on this fork
  "to reduce the amount of compute that is wasted", and a manual start was refused (HTTP 422). No
  questions were missed in that window (none opened, per the run records up to the stop).
- Response: the schedule goes back to every 20 minutes, the setting that ran without trouble from
  Nov 17 to Nov 20, and the workflows were re-enabled from the Actions page under the owner's
  standing instruction to run the GitHub setup and scheduled runs.
- 20:45 UTC: the manual run had sat in "Queued" for 1h45 and no scheduled run had started. GitHub
  billing shows no problem (public repository, $0 billable). The schedule change was pushed a minute
  before the re-enable, so it may not have registered: the stuck run was cancelled and the workflow
  file touched again to register the schedule afresh.
- 20:45 UTC: a fresh manual run started and finished normally, forecasting 1 open main question.
  The stuck run #168 cannot be cancelled ("has not been queued yet") and is left alone. Outage:
  about 20.7 hours (Nov 24 00:04 to 20:45 UTC). The coverage count shows 3 main questions opened and
  closed in that window without a forecast.

## 2026-11-21: parser set to temperature 0 (agent)

- Twice (Nov 18 and Nov 21) one forecast on a multiple-choice question was lost because the
  parser read the same answer two different ways ("Sampled outputs from structure_output are not
  the same"). The questions still went in on the other forecasts. The parser now runs at
  temperature 0 so its two readings agree.
- The 10-minute schedule gives about 28 runs a day (GitHub starts them every 40 to 60 minutes),
  up from about 23. All 100 most recent runs succeeded.

## 2026-11-20: coverage counted, schedule tightened (agent)

- New coverage count (agent/coverage.py, run by the review job). Main tournament: 168 questions so
  far, 12 forecast, all 156 missed ones closed before the bot started on Nov 17; none missed since.
  Current MiniBench (Nov 16 to Dec 4): all 60 questions already closed, 14 forecast, 46 closed before
  the bot started. So this MiniBench round is a partial round for the bot and does not count as one
  of the two full rounds in the stop rule.
- Questions are scored about 2 to 3 hours after they open. Two MiniBench forecasts went in with
  only 0.3 and 0.5 hours to spare, because GitHub started the 20-minute schedule only about hourly.
  The schedule now asks for a run every 10 minutes; public repositories pay nothing for this.

## 2026-11-19: cost measurement fixed (agent)

- With $47 of credits the plan still cut main-tournament questions to 2 forecasts. Cause: the
  cost ratio took the framework's own tally as a floor, and that tally runs 2 to 3 times the real
  charge. The ratio now uses the rise in the key's usage between consecutive runs, which counts
  late charges, and leaves out the newest run until its costs are final.
- On the real records the ratio is about 0.46, so the plan goes back to 4 main and 2 MiniBench
  forecasts. Expected remaining spend at that rate: about $32 of the $45 above the floor.
- First reviews (Nov 19): 7 main and 14 MiniBench questions forecast, none resolved yet, no
  leaderboard entry yet.

## 2026-11-18: stage two funded (owner purchase, agent config)

- The owner bought the stage-two credits on Nov 18 (asked for $45; about $47.48 with the 5.5% fee).
  `funded_usd` raised to 50.0 (5 + 45) and `credit_floor_usd` back to 2.0.
- Stage one was not run down to zero: the cutoff check had already shown OpenRouter refusing a
  request the credits could not cover, and draining would have cost missed questions.
- Proof the new credits exist comes when the key's usage passes $5 with no refusals.
- Total E3 cash is now about $53.28, under the approved $105.50.

## 2026-11-17: first live day, cost measurement, cutoff check (agent)

- Six live tournament runs by 18:02 UTC: 11 questions forecast (2 main, 9 MiniBench), none failed.
- OpenRouter posts part of each run's charges after the run ends, so a run's own
  `provider_cost_usd` understates it. Real costs are taken from the key's cumulative usage
  (`usage_before` of the next run): the test cost $0.906 for 6 questions at 4 forecasts
  (~$0.15 each); live questions so far ~$0.094 each at 2 forecasts. The spend plan's cost ratio
  uses the larger of tracked and provider figures, so it errs toward caution.
- Balance is about $3.06, under the $3.56 ceiling for the free cutoff check, so the check was
  started by changing `config/run_probe.txt`.

## 2026-11-17: live test passed, bot switched live (agent)

- Test Bot run 1: 6 of 6 test questions forecast and submitted, 24 forecasts (12 per model),
  no errors. OpenRouter's own usage rose $0.7648, about $0.13 a question with 4 forecasts,
  against the $0.36 estimate. The framework's own tally said $1.81; OpenRouter's figure is the
  one that counts.
- Switched `live` to true. The remaining ~$4.24 now goes to real tournament questions until
  OpenRouter refuses a call, which is the cutoff test. Floor stays $0 for stage one.

## 2026-11-17: stage one funded, live test started (agent)

- Both secrets are in the fork and Actions are on. `funded_usd` set to 5.0 for the $5 stage-one
  purchase; the bot reads its balance as $5 minus the key's usage. If the purchase has not gone
  through, OpenRouter refuses the calls and nothing is spent.
- Started Test Bot on Metaculus's test questions. `live` stays false until the test passes.

## 2026-11-16: credits bought in stages (owner instruction, agent design)

- The owner asked for the smallest purchase that covers the live test first. OpenRouter's
  minimum is $5, so stage one is $5. The key keeps a $100 limit as the overall cap.
- `budget.funded_usd` records what has been bought. It ships at 0, so nothing can run
  until a purchase is recorded.
- Stage one runs with no floor: the test first, then real tournament questions at the
  minimum forecast count until the credits are gone. That shows the cutoff with real
  records and wastes nothing. The floor returns for stage two.
- Every model call now has a `max_tokens` cap, and questions are forecast a few at a
  time. Reason: OpenRouter refuses requests whose worst-case price does not fit in the
  balance, which matters on a $5 balance.
- Stage two is sized from the costs stage one measures, and keeps total cash at or under
  the approved $105.50.

## 2026-11-16: experiment approved (owner)

- The owner approved the experiment as proposed: $100 of OpenRouter credits plus the
  purchase fee, no top-up without a new approval, decision about 2027-01-04.
- The agent may push code and settings, switch the bot live after the Test Bot run
  passes, and switch it off at any time.
- Checked the same day: the panel models and the parser model are listed on OpenRouter
  under the names in `config/bot_config.json`. A live call is still unproven until the
  Test Bot run.

## 2026-11-15: initial configuration (agent, pending owner approval)

- Panel: Gemini 3.1 Pro (high reasoning) and GPT-5.5 (minimal reasoning). Both sat in the
  top seven of Metaculus's model leaderboard as mirrored on 2026-09-04, and Metaculus's own
  cost notes put each near $0.08 a forecast.
- Main tournament: 4 forecasts a question. MiniBench: 2. The main tournament holds about
  nine times the prize money per question, so it gets the larger share of the credits.
- Research: AskNews if granted, otherwise Sonar Pro through OpenRouter.
- Not live. The switch turns on only after the Test Bot run passes.
