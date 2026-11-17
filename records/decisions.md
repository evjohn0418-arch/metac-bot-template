# Decisions

Newest first. Each entry: date, what changed, the evidence, who decided.

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
