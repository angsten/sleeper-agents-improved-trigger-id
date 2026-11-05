"""The Modal bill, by hour, read-only.

    uv run python -m harness.spend --since YYYY-MM-DDTHH:MM            # the project's apps, UTC
    uv run python -m harness.spend --since YYYY-MM-DDTHH:MM --all      # every object the workspace was billed for

This prints the billed cost from Modal's own billing report (the Workspace billing API of the Modal client in uv.lock,
1.6.1: modal.Workspace.from_context().billing.report, the API behind `modal billing report`), at hourly resolution
from the given start (UTC) to now: each object's cost per hour (an object is an app, by its name), each hour's total
and the running total. It makes no other Modal call and writes nothing.

By default it keeps the project's own apps, sleeper-s003 (training, readouts, downloads) and sleeper-s004-sft (answer
generation): the workspace also carries other projects' apps, whose bills are not this project's. --all shows everything.

What it cannot show: Modal reports whole hours only, so the hour in progress is missing, and a start inside an hour is
rounded down to that hour's beginning (Modal's report rule); the figures can lag behind the dashboard.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal


def parse_utc(s: str) -> datetime:
    """An ISO time, read as UTC if it carries no zone; a time with another zone is converted to UTC."""
    t = datetime.fromisoformat(s)
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def fetch(start: datetime, end: datetime | None = None) -> list[dict]:
    """Modal's hourly billing report, read-only: [{"hour", "object_id", "description", "cost"}]."""
    import modal

    rows = modal.Workspace.from_context().billing.report(start=start, end=end, resolution="h")
    return [{"hour": r.interval_start.astimezone(timezone.utc), "object_id": r.object_id, "description": r.description,
             "cost": Decimal(r.cost)} for r in rows]


def by_hour(rows: list[dict], objects: list[str] | None = None) -> dict:
    """Each hour's cost per object, the hour's total and the running total, over the objects named (by description or
    id; all if none). Costs stay Decimals."""
    keep = [r for r in rows if not objects or r["description"] in objects or r["object_id"] in objects]
    hours: dict[datetime, dict] = {}
    for r in sorted(keep, key=lambda r: (r["hour"], r["description"], r["object_id"])):
        h = hours.setdefault(r["hour"], {"objects": {}, "total": Decimal(0)})
        name = r["description"] or r["object_id"]
        h["objects"][name] = h["objects"].get(name, Decimal(0)) + r["cost"]
        h["total"] += r["cost"]
    running = Decimal(0)
    for h in hours.values():
        running += h["total"]
        h["running_total"] = running
    return {"hours": hours, "total": running}


def show(res: dict, start: datetime) -> str:
    lines = [f"Modal billed cost by hour (UTC) since {start:%Y-%m-%d %H:%M}, whole hours only (the hour in progress is not reported)"]
    for hour, h in res["hours"].items():
        parts = ", ".join(f"{k} ${v:.2f}" for k, v in sorted(h["objects"].items()))
        lines.append(f"{hour:%Y-%m-%d %H:00}  ${h['total']:8.2f}  running ${h['running_total']:8.2f}  ({parts})")
    lines.append(f"total ${res['total']:.2f}")
    return "\n".join(lines)


PROJECT_APPS = ("sleeper-s003", "sleeper-s004-sft")  # harness/modal_app.py APP_NAME and harness/sft_app.py APP_NAME


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="harness.spend", description="The Modal bill by hour from a start time; read-only")
    ap.add_argument("--since", required=True, help="the start, ISO; UTC unless it carries a zone, e.g. 2026-10-23T14:00")
    ap.add_argument("--until", default="", help="the end (exclusive), ISO; default now")
    ap.add_argument("--object", action="append", default=[], help="an app's name or object id to keep; repeatable; "
                                                                   f"default the project's apps, {', '.join(PROJECT_APPS)}")
    ap.add_argument("--all", action="store_true", help="every object the workspace was billed for, other projects' included")
    a = ap.parse_args(argv)
    start = parse_utc(a.since)
    end = parse_utc(a.until) if a.until else None
    print(show(by_hour(fetch(start, end), None if a.all else (a.object or list(PROJECT_APPS))), start))


if __name__ == "__main__":
    main()
