# Decisions

Newest first. Each entry: date, what changed, the evidence, who decided.

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
