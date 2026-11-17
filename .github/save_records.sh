#!/usr/bin/env bash
# Commit anything new under records/ back to the repository, so every run leaves a record
# that can be read later. Retries if someone else pushed in the meantime.
set -u
MESSAGE="${1:-Record}"
git config user.name "forecast-bot"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add records/ 2>/dev/null || exit 0
if git diff --cached --quiet; then
  echo "No new record to save."
  exit 0
fi
git commit -m "$MESSAGE" >/dev/null
for attempt in 1 2 3; do
  if git push; then
    echo "Record saved."
    exit 0
  fi
  git pull --rebase || true
done
echo "Could not save the record after 3 tries."
exit 1
