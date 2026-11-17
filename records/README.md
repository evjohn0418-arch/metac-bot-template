# Records

Written by the bot's jobs, not by hand.

- `runs.jsonl`: one line per run that forecast, skipped for a new reason, or failed.
- `reviews/`: scores on resolved questions and leaderboard standing, saved twice a week.
- `probes.jsonl`: results of the spending-cutoff check.
- `logs/`: the last lines of a job's output, with secret values removed. Test runs always save one; live runs only when they fail.
- `decisions.md`: changes made to the bot and why, newest first.

These are working records. Money counts as earned only when a payment record exists;
see the budget tracker.
