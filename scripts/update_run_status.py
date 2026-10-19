"""Collect the n2 run status and publish it to the pinned GitHub status issue.

The project homepage shows what is running on n2 without anyone editing the repo:
a scheduled task on the owner's machine runs ``push`` once a day, which
reads the Slurm queue and the registered run ledgers over ssh (read-only) and
rewrites the body of one pinned issue (label ``run-status``). The site workflow
runs ``fetch`` before building, so the page follows the issue, not a commit.

Modes:
    push   collect from n2 and update the issue when the status changed, or when
           the last update is older than ``heartbeat_hours`` (default mode).
    fetch  --out PATH: copy the JSON block from the issue to PATH (used in CI).
    show   collect and print the status JSON; touches nothing on GitHub.

Tracked runs are registered on n2, one JSON file per run in ``registry``, so a
long run can be added by whoever starts it, without a PR::

    {"id": "o2-dft", "title": "开放世界 DFT 认证（O2）", "pr": 119,
     "log": "/home/lhshen/xtbflow-runs/.../o2-dispatch-v4.log",
     "format": "dispatch-jsonl", "total": 28, "started": "2026-10-17T14:40:00+08:00"}

``format`` is ``dispatch-jsonl`` (one JSON record with an ``outcome`` per finished
item, as written by the review dispatchers) or ``none`` (title and PR only).

Slurm accounting is disabled on n2, so there is no job history; jobs that leave
the queue are remembered here as "recently left" for ``recent_hours``.

Standard library only, so the scheduled task can use the system Python.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "run_status.json"
BEGIN, END = "<!-- RUN_STATUS_JSON", "RUN_STATUS_JSON -->"
NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # keep the scheduled task from flashing consoles

# Runs on n2 via `ssh n2 python3 - USER REGISTRY`; reads only.
REMOTE = r'''
import datetime, glob, json, os, socket, subprocess, sys
user, registry = sys.argv[1], sys.argv[2]
now = datetime.datetime.now().astimezone()
fields = ["job", "name", "state", "elapsed", "limit", "reason", "submitted", "started"]
out = subprocess.run(["squeue", "-u", user, "-h", "-o", "%i|%j|%T|%M|%l|%R|%V|%S"],
                     capture_output=True, text=True, timeout=30)
queue = [dict(zip(fields, line.split("|", 7))) for line in out.stdout.splitlines() if line.strip()]
runs = []
for path in sorted(glob.glob(os.path.join(registry, "*.json"))):
    try:
        run = json.load(open(path, encoding="utf-8"))
    except Exception as exc:
        runs.append({"id": os.path.basename(path), "error": "bad registry file: %s" % exc})
        continue
    log = run.get("log")
    run["log_exists"] = bool(log) and os.path.exists(log)
    if run["log_exists"]:
        run["log_mtime"] = datetime.datetime.fromtimestamp(os.path.getmtime(log)).astimezone().isoformat(timespec="seconds")
    if run.get("format") == "dispatch-jsonl" and run["log_exists"]:
        done, counts, last, complete = 0, {}, None, False
        for line in open(log, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            if "All planned items reached terminal states" in line:
                complete = True
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and "outcome" in rec:
                done += 1
                key = rec.get("detail") or rec.get("outcome")
                counts[key] = counts.get(key, 0) + 1
                last = {k: rec.get(k) for k in ("job_id", "kind", "item_id", "outcome", "detail")}
        run.update(done=done, counts=counts, last=last, complete=complete)
    runs.append(run)
print(json.dumps({"collected_at": now.isoformat(timespec="seconds"), "host": socket.gethostname(),
                  "queue": queue, "runs": runs}))  # ASCII escapes: immune to the remote locale
'''


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def run(cmd: list[str], *, input_text: str | None = None, timeout: int = 60) -> str:
    proc = subprocess.run(cmd, input=input_text, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, creationflags=NO_WINDOW)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} exited {proc.returncode}: {proc.stderr.strip()[:400]}")
    return proc.stdout


def now_local() -> dt.datetime:
    return dt.datetime.now().astimezone().replace(microsecond=0)


def parse_time(text: str | None) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(text) if text else None
    except ValueError:
        return None


# ------------------------------------------------------------------ collection

def collect(cfg: dict) -> dict:
    out = run(["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={min(cfg['ssh_timeout_s'], 30)}",
               cfg["ssh_host"], "python3", "-", cfg["slurm_user"], cfg["registry"]],
              input_text=REMOTE, timeout=cfg["ssh_timeout_s"])
    return json.loads(out)


def merge(cfg: dict, fresh: dict, previous: dict | None) -> dict:
    """Fold a fresh n2 snapshot into the previous payload (recent jobs, finished runs)."""
    now = parse_time(fresh["collected_at"]) or now_local()
    in_queue = {j["job"] for j in fresh["queue"]}
    recent = [] if not previous else [r for r in previous.get("recent", []) if r["job"] not in in_queue]
    for job in (previous or {}).get("queue", []):
        if job["job"] not in in_queue and all(r["job"] != job["job"] for r in recent):
            recent.append({"job": job["job"], "name": job["name"], "last_state": job["state"],
                           "last_elapsed": job["elapsed"], "left_queue_by": fresh["collected_at"]})
    horizon = now - dt.timedelta(hours=cfg["recent_hours"])
    recent = [r for r in recent if (parse_time(r["left_queue_by"]) or now) >= horizon]
    keep_until = now - dt.timedelta(days=cfg["finished_run_days"])
    runs = [r for r in fresh["runs"]
            if not r.get("complete") or (parse_time(r.get("log_mtime")) or now) >= keep_until]
    return {"collected_at": fresh["collected_at"], "host": fresh["host"], "schedule": cfg["schedule_note"], "error": None,
            "last_attempt": fresh["collected_at"], "queue": fresh["queue"], "runs": runs,
            "recent": recent}


def substance(payload: dict | None) -> str:
    """The part whose change is worth an issue edit (timestamps excluded)."""
    if not payload:
        return ""
    keep = {k: payload.get(k) for k in ("queue", "runs", "recent", "error")}
    # Elapsed time changes on every check; only state transitions are worth an edit.
    keep["queue"] = [{k: v for k, v in j.items() if k not in ("elapsed", "reason")} for j in keep.get("queue") or []]
    return json.dumps(keep, ensure_ascii=False, sort_keys=True)


# ------------------------------------------------------------------ issue I/O

def gh(*args: str, input_text: str | None = None) -> str:
    return run(["gh", *args], input_text=input_text, timeout=60)


def find_issue(cfg: dict) -> dict | None:
    rows = json.loads(gh("issue", "list", "--repo", cfg["repo"], "--label", cfg["label"], "--state", "open",
                         "--json", "number,body,url", "--limit", "5"))
    return min(rows, key=lambda r: r["number"]) if rows else None


def extract(body: str) -> dict | None:
    m = re.search(re.escape(BEGIN) + r"\s*(\{.*?\})\s*" + re.escape(END), body or "", re.S)
    return json.loads(m.group(1)) if m else None


def ensure_issue(cfg: dict) -> dict:
    issue = find_issue(cfg)
    if issue:
        return issue
    gh("label", "create", cfg["label"], "--repo", cfg["repo"], "--color", "1D76DB", "--force",
       "--description", "由 scripts/update_run_status.py 自动维护的运行状态")
    url = gh("issue", "create", "--repo", cfg["repo"], "--title", cfg["issue_title"], "--label", cfg["label"],
             "--body", "初始化中，稍后由自动任务写入。").strip()
    number = int(url.rstrip("/").rsplit("/", 1)[-1])
    node = json.loads(gh("api", f"repos/{cfg['repo']}/issues/{number}", "--jq", "{id: .node_id}"))["id"]
    gh("api", "graphql", "-f", "query=mutation($id: ID!) { pinIssue(input: {issueId: $id}) { issue { number } } }",
       "-f", f"id={node}")
    return {"number": number, "body": "", "url": url}


STATE_ZH = {"RUNNING": "运行中", "PENDING": "排队", "COMPLETING": "收尾中", "CONFIGURING": "启动中"}


def render(cfg: dict, payload: dict) -> str:
    lines = [f"<!-- 本 issue 由 scripts/update_run_status.py 自动维护（本机定时任务，{cfg['schedule_note']} 检查一次）。"
             "请不要手动编辑；改动会在下次检查时被覆盖。 -->",
             "## n2 运行状态（自动更新）", "",
             f"最近一次采集：**{payload['collected_at'].replace('T', ' ')}**（{payload['host']}）。本机定时任务"
             f"{cfg['schedule_note']} 检查一次 n2；项目主页每天重建时读取本 issue。", ""]
    if payload.get("error"):
        lines += [f"> [!WARNING]", f"> {payload['last_attempt']} 采集失败：{payload['error']}。下面是上次成功采集的结果。", ""]
    lines += ["### Slurm 队列", ""]
    if payload["queue"]:
        lines += ["| 作业 | 名称 | 状态 | 已运行 | 时限 | 节点/原因 |", "|---|---|---|---|---|---|"]
        lines += [f"| {j['job']} | `{j['name']}` | {STATE_ZH.get(j['state'], j['state'])} | {j['elapsed']} | "
                  f"{j['limit']} | {j['reason']} |" for j in payload["queue"]]
    else:
        lines.append(f"`{cfg['slurm_user']}` 当前没有排队或运行中的作业。")
    lines += ["", "### 登记的长任务", ""]
    if payload["runs"]:
        lines += ["| 任务 | PR | 进度 | 状态 | 最近一项 |", "|---|---|---|---|---|"]
        for r in payload["runs"]:
            if r.get("error"):
                lines.append(f"| {r['id']} | | | 登记文件有误：{r['error']} | |")
                continue
            prog = f"{r.get('done', '?')} / {r.get('total', '?')}" if "done" in r else "—"
            state = "已全部到达终态" if r.get("complete") else ("日志缺失" if not r.get("log_exists") else "进行中")
            last = r.get("last") or {}
            last_text = f"{last.get('job_id', '')} {last.get('kind', '')} `{last.get('detail') or last.get('outcome') or ''}`" if last else ""
            pr = f"#{r['pr']}" if r.get("pr") else ""
            lines.append(f"| {r.get('title', r['id'])} | {pr} | {prog} | {state} | {last_text.strip()} |")
    else:
        lines.append(f"没有登记的长任务（登记目录 `{cfg['registry']}`）。")
    if payload["recent"]:
        lines += ["", f"### 最近 {cfg['recent_hours']} 小时离开队列的作业", "",
                  "| 作业 | 名称 | 最后状态 | 最后已运行 | 离开队列不晚于 |", "|---|---|---|---|---|"]
        lines += [f"| {r['job']} | `{r['name']}` | {STATE_ZH.get(r['last_state'], r['last_state'])} | "
                  f"{r['last_elapsed']} | {r['left_queue_by']} |" for r in payload["recent"]]
    data = json.dumps(payload, ensure_ascii=False, indent=1).replace("-->", "--\\u003e")
    lines += ["", BEGIN, data, END, ""]
    return "\n".join(lines)


def update_issue(cfg: dict, issue: dict, body: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md", delete=False) as fh:
        fh.write(body)
        path = fh.name
    try:
        gh("issue", "edit", str(issue["number"]), "--repo", cfg["repo"], "--body-file", path)
    finally:
        os.unlink(path)


# ------------------------------------------------------------------ modes

def log(message: str) -> None:
    folder = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "xtbflow"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "run_status.log"
    lines = path.read_text(encoding="utf-8").splitlines()[-499:] if path.exists() else []
    lines.append(f"{now_local().isoformat()} {message}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def push(cfg: dict, force: bool) -> int:
    issue = ensure_issue(cfg)
    previous = extract(issue.get("body", ""))
    try:
        payload = merge(cfg, collect(cfg), previous)
    except Exception as exc:  # n2 unreachable, ssh timeout, bad output
        if previous is None:
            log(f"collect failed and no previous status: {exc}")
            return 1
        payload = dict(previous, error=str(exc)[:300], last_attempt=now_local().isoformat())
    last = parse_time((previous or {}).get("collected_at" if not payload.get("error") else "last_attempt"))
    stale = last is None or now_local() - last >= dt.timedelta(hours=cfg["heartbeat_hours"])
    if not force and substance(payload) == substance(previous) and not stale:
        log(f"#{issue['number']} unchanged")
        return 0
    update_issue(cfg, issue, render(cfg, payload))
    log(f"#{issue['number']} updated (queue={len(payload['queue'])}, runs={len(payload['runs'])}, "
        f"error={bool(payload.get('error'))})")
    return 0


def fetch(cfg: dict, out: Path) -> int:
    issue = find_issue(cfg)
    payload = extract(issue["body"]) if issue else None
    if not payload:
        print("no run-status issue with a data block; the site falls back to progress.yaml")
        return 0
    payload["issue_url"] = issue["url"]
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out} from {issue['url']} (collected {payload['collected_at']})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", nargs="?", default="push", choices=["push", "fetch", "show"])
    parser.add_argument("--out", type=Path, help="fetch: where to write the status JSON")
    parser.add_argument("--force", action="store_true", help="push: edit the issue even if nothing changed")
    args = parser.parse_args()
    cfg = load_config()
    if args.mode == "show":
        print(json.dumps(merge(cfg, collect(cfg), None), ensure_ascii=False, indent=1))
        return 0
    if args.mode == "fetch":
        if not args.out:
            parser.error("fetch needs --out")
        return fetch(cfg, args.out)
    try:
        return push(cfg, args.force)
    except Exception as exc:
        log(f"push failed: {exc}")
        raise


if __name__ == "__main__":
    sys.exit(main())
