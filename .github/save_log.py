"""Save the tail of a job's output under records/logs/, with secret values removed.

The repository is public and the agent that maintains the bot cannot read GitHub's own
job logs, so a short, cleaned copy is kept with the other records.
Usage: save_log.py <name> "<step outcomes>"
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SECRET_ENV_NAMES = ("METACULUS_TOKEN", "OPENROUTER_API_KEY", "ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET")
MAX_LINES = 150
MAX_WIDTH = 400


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "job"
    outcomes = sys.argv[2] if len(sys.argv) > 2 else ""
    secrets = [v for v in (os.getenv(n) for n in SECRET_ENV_NAMES) if v and len(v) >= 6]
    source = Path("run.log")
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines() if source.exists() else []
    tail = []
    for line in lines[-MAX_LINES:]:
        for secret in secrets:
            line = line.replace(secret, "[hidden]")
        tail.append(line[:MAX_WIDTH])
    server, repo, run_id = os.getenv("GITHUB_SERVER_URL"), os.getenv("GITHUB_REPOSITORY"), os.getenv("GITHUB_RUN_ID")
    header = [
        f"job: {name}",
        f"saved: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"steps: {outcomes}",
        f"run: {server}/{repo}/actions/runs/{run_id}" if server and repo and run_id else "run: (not on GitHub)",
        f"lines kept: {len(tail)} of {len(lines)}",
        "",
    ]
    target = Path("records") / "logs" / f"{name}-latest.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(header + tail) + "\n", encoding="utf-8")
    print(f"Saved {len(tail)} log lines to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
