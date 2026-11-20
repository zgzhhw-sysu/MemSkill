"""Inspect exchange collection without importing models or acquiring a GPU lease."""

import json
import sys
from pathlib import Path


def contents(path):
    return json.loads(path.read_bytes())


def stage_jobs(jobs):
    return {
        "pilot": [j for j in jobs if j["session"] == 5 and j["slot"] == 0],
        "breadth": [j for j in jobs if j["slot"] == 0],
        "full": jobs,
    }


def resolved(out, job):
    name = job["id"] + ".json"
    q = out / "questions" / name
    return q.is_file() and (
        contents(q)["status"] != "accepted" or (out / "outcomes" / name).is_file()
    )


def state(out, role):
    reg = out / "registration.json"
    if role == "questions":
        if (out / "questions-complete.json").is_file():
            return "done"
        if out.name == "framed" and not (out.parent / "registration.json").is_file():
            return "wait"
        if out.name == "framed" and reg.is_file():
            ready = {f.name for f in (out.parent / "questions").glob("*.json")}
            if (
                not ready - {f.name for f in (out / "questions").glob("*.json")}
                and not (out.parent / "questions-complete.json").exists()
            ):
                return "wait"
        return "work"
    if not reg.is_file():
        return "wait"
    if role == "policy":
        # Re-score a frozen policy only when more measured panel windows arrive.
        for stage, jobs in stage_jobs(contents(reg)["jobs"]).items():
            if all(resolved(out, j) for j in jobs):
                folder = out / "policy" / stage
                if not (folder / "scored-state.json").is_file() or contents(
                    folder / "scored-state.json"
                ) != panel_state(out):
                    return "work"
        return "wait"
    src, dst = (
        ("questions", "prepared") if role == "prepare" else ("prepared", "outcomes")
    )
    finished = out / (
        "questions-complete.json" if role == "prepare" else "prepare.complete"
    )
    upstream_done = finished.is_file()
    for f in (out / src).glob("*.json"):
        if not (out / dst / f.name).is_file() and (
            role != "prepare" or contents(f)["status"] == "accepted"
        ):
            return "work"
    return "done" if upstream_done else "wait"


def panel_state(out):
    root = out.parent if out.name == "framed" else out
    base = root.parents[1]
    return [
        sum(1 for _ in p.glob(pattern))
        for p, pattern in [
            (base / "late-pairs/outcomes", "*/pair.json"),
            (base / "late-pairs/donor-factorial", "*/complete.json"),
            (root.parent / "witness-passages/four-paths/cells", "*/complete.json"),
            (base / "late-pairs/cross-window-prefix", "*/pair.json"),
        ]
    ]


if __name__ == "__main__":
    print(state(Path(sys.argv[1]), sys.argv[2]))
