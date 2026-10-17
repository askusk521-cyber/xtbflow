"""Check the project homepage records against git, and refresh the GitHub snapshot.

The homepage (mkdocs.yml, portal/) renders figures and tables from
portal/records/*.yaml. Evidence there is addressed as (PR head commit, path) so
links stay valid when branches move. This script verifies that every such path
exists at that commit, that archive paths exist in the working tree, and that
each recorded head commit still matches the PR's current head on GitHub.

Usage:
    python scripts/check_portal.py                    # offline checks (needs git objects)
    python scripts/check_portal.py --refresh-github   # also rewrite github_snapshot.json via gh
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "portal" / "records"
SNAPSHOT = RECORDS / "github_snapshot.json"


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, encoding="utf-8")


# A plain YAML scalar ends at " #" (comment), so "见 #119" silently loses its tail.
UNQUOTED_HASH = re.compile(r"^\s*(?:-\s+)?\w+:\s+[^\"'\s{\[|>].*\s#")


def unquoted_hash_lines(path: Path) -> list[str]:
    return [f"{path.name}:{i}: quote this value, ' #' starts a YAML comment"
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if not line.lstrip().startswith("#") and UNQUOTED_HASH.match(line)]


def object_exists(sha: str, path: str) -> bool:
    return git("cat-file", "-e", f"{sha}:{path.rstrip('/')}").returncode == 0


def gh_json(*args: str) -> list[dict]:
    out = subprocess.run(["gh", *args], cwd=ROOT, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, encoding="utf-8", check=True)
    return json.loads(out.stdout)


def refresh_snapshot() -> dict:
    """Snapshot PR/issue titles and states so the site builds without network access."""
    prs = gh_json("pr", "list", "--state", "all", "--limit", "500", "--json",
                  "number,title,state,isDraft,headRefName,baseRefName,headRefOid,createdAt,mergedAt")
    issues = gh_json("issue", "list", "--state", "all", "--limit", "500", "--json",
                     "number,title,state,createdAt,closedAt")
    snapshot = {
        "prs": {str(p["number"]): {k: p[k] for k in ("title", "state", "isDraft", "headRefName",
                                                    "baseRefName", "headRefOid", "createdAt", "mergedAt")}
                for p in prs},
        "issues": {str(i["number"]): {k: i[k] for k in ("title", "state", "createdAt", "closedAt")}
                   for i in issues},
    }
    SNAPSHOT.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh-github", action="store_true",
                        help="Rewrite portal/records/github_snapshot.json from the GitHub API (needs gh).")
    args = parser.parse_args()

    snapshot = refresh_snapshot() if args.refresh_github else (
        json.loads(SNAPSHOT.read_text(encoding="utf-8")) if SNAPSHOT.exists() else {"prs": {}, "issues": {}})
    exp = yaml.safe_load((RECORDS / "experiments.yaml").read_text(encoding="utf-8"))
    legacy = yaml.safe_load((RECORDS / "legacy.yaml").read_text(encoding="utf-8"))
    failures: list[str] = []
    checked = 0
    for path in sorted(RECORDS.glob("*.yaml")):
        failures.extend(unquoted_hash_lines(path))

    for number, pr in exp["prs"].items():
        if len(pr["sha"]) != 40:
            failures.append(f"#{number}: sha must be the full 40-character commit")
        live = snapshot["prs"].get(str(number), {}).get("headRefOid")
        if live and live != pr["sha"]:
            failures.append(f"#{number}: recorded head {pr['sha'][:10]} != GitHub head {live[:10]} (update experiments.yaml)")

    ids = {e["id"] for e in exp["experiments"]}
    for item in exp["experiments"]:
        for pr in item["prs"]:
            if pr not in exp["prs"]:
                failures.append(f"{item['id']}: PR #{pr} missing from prs table")
        for ev in item["evidence"]:
            checked += 1
            sha = exp["prs"][ev["pr"]]["sha"]
            if not object_exists(sha, ev["path"]):
                failures.append(f"{item['id']}: {ev['path']} not found at #{ev['pr']} {sha[:10]}")
    for row in exp["forest"]:
        if row["exp"] not in ids:
            failures.append(f"forest row {row['label']!r}: unknown experiment {row['exp']}")
        if not row["lo"] <= row["est"] <= row["hi"]:
            failures.append(f"forest row {row['label']!r}: estimate outside its interval")

    for era in legacy["eras"]:
        for f in era.get("files", []):
            checked += 1
            if not (ROOT / f["path"]).exists():
                failures.append(f"archive {era['id']}: {f['path']} not in working tree")
        for number in era.get("prs", []):
            if snapshot["prs"] and str(number) not in snapshot["prs"]:
                failures.append(f"archive {era['id']}: PR #{number} not in snapshot")
        for number in era.get("issues", []):
            if snapshot["issues"] and str(number) not in snapshot["issues"]:
                failures.append(f"archive {era['id']}: issue #{number} not in snapshot")

    for line in failures:
        print("FAIL", line)
    print(f"checked {checked} paths; {len(failures)} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
