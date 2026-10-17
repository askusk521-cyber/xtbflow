"""MkDocs hook that renders the homepage figures and tables from portal/records/*.yaml.

Pages place markers such as ``<!-- viz:forest -->`` on a line of their own; this hook
replaces each marker with generated HTML/SVG before Markdown is parsed. Keeping the
numbers in YAML (not in page prose) gives one place to update when a result changes,
and lets scripts/check_portal.py verify every evidence path against git.

All charts use CSS custom properties defined in portal/assets/site.css, so the same
markup follows Material's light/dark scheme. Every chart ships with a text or table
equivalent on the same page (values are never conveyed by colour or hover alone).
"""
from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path

import yaml

RECORDS = Path(__file__).resolve().parents[1] / "portal" / "records"
MARKER = re.compile(r"^<!--\s*viz:([a-z0-9_-]+)(?::([a-z0-9_-]+))?\s*-->\s*$", re.M)

STATUS = {
    # status -> (css class, icon, default label). Icons keep state readable without colour.
    "go": ("go", "✓", "通过"),
    "nogo": ("nogo", "✕", "未通过"),
    "paused": ("paused", "‖", "暂缓"),
    "explore": ("explore", "◇", "探索性"),
    "running": ("running", "●", "运行中"),
}
LEVEL = {"gate": "预注册闸门", "exploratory": "探索性检查"}

_data: dict = {}
_prefix = ""  # relative path from the page being rendered to the site root


def site_link(target: str) -> str:
    """Turn a docs-relative Markdown path (e.g. paper/results.md#m0) into a URL from the current page.

    Raw HTML is not rewritten by MkDocs, so generated links must already be final URLs
    (use_directory_urls is on: paper/results.md -> paper/results/).
    """
    path, _, anchor = target.partition("#")
    if path.endswith("index.md"):
        url = path[: -len("index.md")]
    elif path.endswith(".md"):
        url = path[:-3] + "/"
    else:
        url = path
    return _prefix + url + (f"#{anchor}" if anchor else "")


def e(text) -> str:
    return html.escape(str(text), quote=True)


def _load() -> dict:
    data = {name: yaml.safe_load((RECORDS / f"{name}.yaml").read_text(encoding="utf-8"))
            for name in ("experiments", "progress", "references", "legacy")}
    snap = RECORDS / "github_snapshot.json"
    data["github"] = json.loads(snap.read_text(encoding="utf-8")) if snap.exists() else {"prs": {}, "issues": {}}
    for item in data["experiments"]["experiments"]:
        item["date"] = str(item["date"])  # YAML reads bare dates as datetime.date
    data["by_id"] = {x["id"]: x for x in data["experiments"]["experiments"]}
    return data


# ---------------------------------------------------------------- links

def repo() -> str:
    return _data["experiments"]["repo"]


def evidence_url(pr: int, path: str) -> str:
    sha = _data["experiments"]["prs"][pr]["sha"]
    kind = "tree" if path.endswith("/") else "blob"
    return f"{repo()}/{kind}/{sha}/{path.rstrip('/')}"


def main_url(path: str) -> str:
    kind = "tree" if path.endswith("/") else "blob"
    return f"{repo()}/{kind}/main/{path.rstrip('/')}"


def pr_link(n: int) -> str:
    title = _data["github"]["prs"].get(str(n), {}).get("title", "")
    return f'<a class="xf-pr" href="{repo()}/pull/{n}" title="{e(title)}">#{n}</a>'


def issue_link(n: int) -> str:
    title = _data["github"]["issues"].get(str(n), {}).get("title", "")
    return f'<a class="xf-pr" href="{repo()}/issues/{n}" title="{e(title)}">#{n}</a>'


def badge(status: str, label: str | None = None) -> str:
    cls, icon, default = STATUS[status]
    return f'<span class="xf-badge xf-{cls}"><span aria-hidden="true">{icon}</span>{e(label or default)}</span>'


def raw_paths(item: dict) -> str:
    if not item["raw"]:
        return '<span class="xf-muted">无独立输出目录</span>'
    root = _data["experiments"]["raw_root"]
    return " ".join(f'<code class="xf-path" title="n2:{e(root)}/{e(p)}">{e(p)}</code>' for p in item["raw"])


def evidence_links(item: dict) -> str:
    return "".join(f'<li><a href="{e(evidence_url(ev["pr"], ev["path"]))}">{e(ev["label"])}</a>'
                   f' <span class="xf-muted">{e(ev["path"])} · #{ev["pr"]}</span></li>'
                   for ev in item["evidence"])


# ---------------------------------------------------------------- homepage pieces

def viz_kpis(_arg) -> str:
    prog = _data["progress"]
    run = prog["running"][0]
    owner = sum(1 for n in prog["next"] if n["owner"])
    open_prs = [n for n, p in _data["github"]["prs"].items()
                if 101 <= int(n) <= 122 and p["state"] == "OPEN"]
    total = len(_data["experiments"]["experiments"])
    running = sum(1 for x in _data["experiments"]["experiments"] if x["status"] == "running")
    tiles = [
        ("当前阶段", prog["current_stage"]["name"], prog["current_stage"]["note"], ""),
        ("正在运行", f'{run["done"]} / {run["total"]}', run["title"], "progress.md#running"),
        ("待负责人决定", f"{owner} 项", "V1b 是否恢复、研究线去留、未合并 PR", "progress.md#next"),
        ("已记录的阶段与检查", f"{total} 项", f"其中 {running} 项运行中；{len(open_prs)} 个 PR（#101–#122）待审",
         "paper/results.md"),
    ]
    out = ['<div class="xf-kpis">']
    for label, value, sub, href in tiles:
        tag = f'a href="{site_link(href)}"' if href else "div"
        out.append(f'<{tag} class="xf-kpi"><div class="xf-kpi-label">{e(label)}</div>'
                   f'<div class="xf-kpi-value">{e(value)}</div><div class="xf-kpi-sub">{e(sub)}</div>'
                   f'</{tag.split()[0]}>')
    out.append("</div>")
    return "".join(out)


def _card(item: dict) -> str:
    return (f'<a class="xf-node xf-node-{STATUS[item["status"]][0]}" href="{e(site_link(item["paper"]))}">'
            f'<span class="xf-node-name">{e(item["name"])}</span>'
            f'<span class="xf-node-meta">{e(item["date"][5:])} · {badge(item["status"], item["verdict"])}</span></a>')


def viz_roadmap(_arg) -> str:
    rm = _data["experiments"]["roadmap"]
    by = _data["by_id"]
    out = ['<div class="xf-roadmap"><div class="xf-rm-label">主线（预注册闸门）</div><div class="xf-rm-main">']
    for i, key in enumerate(rm["main"]):
        if i:
            out.append('<span class="xf-arrow" aria-hidden="true">→</span>')
        out.append(_card(by[key]))
    out.append('</div><div class="xf-rm-label">10-08 起的探索性检查（V1a 开发集 / 已揭盲筛查集，不构成确认性证据）</div>'
               '<div class="xf-rm-lines">')
    for line in rm["lines"]:
        out.append(f'<div class="xf-rm-line"><div class="xf-rm-line-name">{e(line["name"])}</div>')
        out.extend(_card(by[k]) for k in line["ids"])
        out.append("</div>")
    out.append("</div></div>")
    return "".join(out)


# ---------------------------------------------------------------- charts

def viz_forest(_arg) -> str:
    rows = _data["experiments"]["forest"]
    by = _data["by_id"]
    lo, hi = -0.26, 0.20

    def x(v: float) -> str:
        # Percent coordinates keep circles round and text unscaled at any column width.
        return f"{3 + (v - lo) / (hi - lo) * 94:.2f}%"

    ticks = [-0.2, -0.1, 0.0, 0.1, 0.2]
    out = ['<figure class="xf-figure xf-forest" id="fig-forest">',
           '<div class="xf-legend"><span><i class="xf-key xf-s1"></i>预注册闸门（确认性）</span>'
           '<span><i class="xf-key xf-s2"></i>探索性检查</span>'
           '<span class="xf-muted">点为估计值，横线为双侧 95% 区间，空心点表示区间包含 0</span></div>',
           '<div class="xf-forest-grid" role="table" aria-label="各比较的效应量与区间">',
           '<div class="xf-fh" role="row"><span role="columnheader">比较与读法</span>'
           '<span role="columnheader">差值与 95% 区间（竖线为 0）</span><span role="columnheader">估计 [区间]</span></div>']
    for r in rows:
        level = by[r["exp"]]["level"]
        cls = "xf-s1" if level == "gate" else "xf-s2"
        crosses = r["lo"] <= 0 <= r["hi"]
        tip = f'{r["label"]}：{r["est"]:+.4f}，95% [{r["lo"]:+.4f}, {r["hi"]:+.4f}]；{r["favors"]}'
        grid = "".join(f'<line class="xf-grid" x1="{x(t)}" x2="{x(t)}" y1="0" y2="100%"/>' for t in ticks if t != 0)
        svg = (f'<svg class="xf-forest-svg" aria-hidden="true">{grid}'
               f'<line class="xf-zero" x1="{x(0)}" x2="{x(0)}" y1="0" y2="100%"/>'
               f'<line class="xf-ci {cls}" x1="{x(r["lo"])}" x2="{x(r["hi"])}" y1="50%" y2="50%"/>'
               f'<circle class="xf-dot {cls}{" xf-open" if crosses else ""}" cx="{x(r["est"])}" cy="50%" r="5.5"/></svg>')
        out.append(f'<div class="xf-fr" role="row" data-tip="{e(tip)}" tabindex="0">'
                   f'<span role="cell" class="xf-fl"><a href="{e(site_link(by[r["exp"]]["paper"]))}">{e(r["label"])}</a>'
                   f'<small>{e(r["favors"])}</small></span>'
                   f'<span role="cell" class="xf-fp">{svg}</span>'
                   f'<span role="cell" class="xf-fv">{r["est"]:+.3f}<br><small>[{r["lo"]:+.3f}, {r["hi"]:+.3f}]</small></span></div>')
    axis = "".join(f'<text x="{x(t)}" y="12" text-anchor="middle">{"0" if t == 0 else f"{t:+.1f}"}</text>' for t in ticks)
    out.append(f'<div class="xf-fr xf-faxis" aria-hidden="true"><span></span><span class="xf-fp">'
               f'<svg class="xf-forest-svg xf-axis-svg">{axis}</svg></span><span></span></div>')
    out.append('</div><figcaption><b>图 3.1</b>　各阶段主要配对比较的效应量。各行指标不同（Hit@32、AUC、机制量、Spearman ρ），'
               '只在同一行内读正负与区间是否跨 0，不能跨行比较大小。数字摘自所链报告；点击行名跳到对应小节。</figcaption></figure>')
    return "".join(out)


def viz_v1a_curves(_arg) -> str:
    cv = _data["experiments"]["v1a_curves"]
    budgets = cv["budgets"]
    w, h = 640, 310
    ml, mr, mt, mb = 48, 44, 30, 40

    def x(b: float) -> float:
        return ml + (math.log2(b) - 2) / 4 * (w - ml - mr)

    def y(v: float) -> float:
        return mt + (1 - v / 0.8) * (h - mt - mb)

    parts = [f'<svg viewBox="0 0 {w} {h}" class="xf-chart" role="img" aria-labelledby="v1a-cap">']
    for v in (0, 0.2, 0.4, 0.6, 0.8):
        parts.append(f'<line class="{"xf-axis" if v == 0 else "xf-grid"}" x1="{ml}" x2="{w - mr}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                     f'<text class="xf-tick" x="{ml - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.1f}</text>')
    for b in budgets:
        parts.append(f'<text class="xf-tick" x="{x(b):.1f}" y="{h - mb + 18}" text-anchor="middle">{b}</text>')
    parts.append(f'<text class="xf-axis-title" x="{(ml + w - mr) / 2:.0f}" y="{h - 4}" text-anchor="middle">'
                 f'预算（B0 等效候选数，对数刻度）</text>')
    parts.append(f'<text class="xf-axis-title" x="{ml - 8}" y="{mt - 14}" text-anchor="end">召回</text>')
    # End labels: sort by final value and push apart so close lines (A1 vs B1) stay legible.
    ends = sorted(((y(a["recall"][-1]), i) for i, a in enumerate(cv["arms"])))
    placed: list[tuple[float, int]] = []
    for yy, i in ends:
        if placed and yy - placed[-1][0] < 15:
            yy = placed[-1][0] + 15
        placed.append((yy, i))
    label_y = {i: yy for yy, i in placed}
    for i, arm in enumerate(cv["arms"]):
        s = f"xf-s{i + 1}"
        pts = " ".join(f"{x(b):.1f},{y(v):.1f}" for b, v in zip(budgets, arm["recall"]))
        parts.append(f'<polyline class="xf-line {s}" points="{pts}"/>')
        for b, v in zip(budgets, arm["recall"]):
            tip = f'{arm["arm"]}（{arm["desc"]}）预算 {b}：召回 {v:.3f}'
            parts.append(f'<circle class="xf-pt {s}" cx="{x(b):.1f}" cy="{y(v):.1f}" r="4"/>'
                         f'<circle class="xf-hit" cx="{x(b):.1f}" cy="{y(v):.1f}" r="12" data-tip="{e(tip)}"/>')
        ly = label_y[i]
        parts.append(f'<text class="xf-end xf-end-arm" x="{x(64) + 10:.1f}" y="{ly + 4:.1f}">{e(arm["arm"])}</text>')
    parts.append("</svg>")
    legend = "".join(f'<span><i class="xf-key xf-s{i + 1}"></i>{e(a["arm"])} {e(a["desc"])}</span>'
                     for i, a in enumerate(cv["arms"]))
    rows = "".join(f'<tr><td><b>{e(a["arm"])}</b> {e(a["desc"])}</td>'
                   + "".join(f"<td>{v:.3f}</td>" for v in a["recall"]) + f'<td><b>{a["auc"]:.4f}</b></td></tr>'
                   for a in cv["arms"])
    table = (f'<details class="xf-tableview"><summary>查看数据表</summary><table><thead><tr><th>臂</th>'
             + "".join(f"<th>@{b}</th>" for b in budgets) + "<th>AUC</th></tr></thead><tbody>"
             + rows + "</tbody></table></details>")
    return (f'<figure class="xf-figure" id="fig-v1a"><div class="xf-legend">{legend}</div>{"".join(parts)}'
            f'<figcaption id="v1a-cap"><b>图 3.2</b>　V1a 五臂成本曲线：最低已知参考通道的联合召回随预算的变化'
            f'（342 个母体 × 三个训练 seed 平均）。主比较是 B1 对 A2；无引导的 B0 在每个预算点都最高。'
            f'来源：<a href="{e(evidence_url(106, "docs/evidence/v1a/formal/RESULT_ZH.md"))}">V1a 正式结果</a>。'
            f'</figcaption>{table}</figure>')


# ---------------------------------------------------------------- progress

def viz_running(_arg) -> str:
    prog = _data["progress"]
    out = []
    for run in prog["running"]:
        pct = 100 * run["done"] / run["total"]
        items = "".join(
            f'<tr><td>{r["job"]}</td><td>{e(r["kind"])}</td><td><code>{e(r["item"])}</code></td>'
            f'<td>{badge("go" if r["outcome"] == "ok" else "nogo", "已判定" if r["outcome"] == "ok" else "失败")}'
            f' {e(r["detail"])}</td><td>{r["runtime_s"]:,} s</td></tr>' for r in run["items"])
        exp = _data["by_id"][run["exp"]]
        out.append(
            f'<div class="xf-run" id="run-{e(run["id"])}"><div class="xf-run-head">{badge("running")}'
            f'<b>{e(run["title"])}</b> {pr_link(run["pr"])}</div>'
            f'<div class="xf-progress" role="progressbar" aria-valuenow="{run["done"]}" aria-valuemin="0" '
            f'aria-valuemax="{run["total"]}"><div style="width:{pct:.1f}%"></div></div>'
            f'<div class="xf-run-meta"><span><b>{run["done"]} / {run["total"]}</b> 项已结束</span>'
            f'<span>开始于 {e(run["started"])}</span><span>{e(run["current"])}</span></div>'
            f'<dl class="xf-run-dl"><dt>GPU 用量</dt><dd>{e(run["gpu_note"])}</dd>'
            f'<dt>执行方式</dt><dd>{e(run["runner"])}</dd>'
            f'<dt>日志</dt><dd><code class="xf-path">n2:{e(run["log"])}</code></dd>'
            f'<dt>证据</dt><dd><ul class="xf-ev">{evidence_links(exp)}</ul></dd></dl>'
            f'<details class="xf-tableview"><summary>已结束的 {run["done"]} 项</summary><table><thead><tr>'
            f'<th>Slurm</th><th>类型</th><th>条目</th><th>结果</th><th>耗时</th></tr></thead><tbody>{items}'
            f'</tbody></table></details></div>')
    out.append(f'<p class="xf-muted xf-small">状态更新于 {e(prog["updated"])}，来源：{e(prog["source"])}。</p>')
    return "".join(out)


def viz_next(_arg) -> str:
    out = ['<ol class="xf-next">']
    for n in _data["progress"]["next"]:
        who = ('<span class="xf-badge xf-paused"><span aria-hidden="true">?</span>需负责人决定</span>'
               if n["owner"] else '<span class="xf-badge xf-explore"><span aria-hidden="true">→</span>执行中自然衔接</span>')
        refs = " ".join(pr_link(r) for r in n["refs"])
        out.append(f'<li><div class="xf-next-head"><b>{e(n["title"])}</b> {who}</div>'
                   f'<p>{e(n["detail"])}</p><p class="xf-muted xf-small">相关：{refs}</p></li>')
    out.append("</ol>")
    return "".join(out)


# ---------------------------------------------------------------- data catalog & evidence boxes

def viz_evidence(key) -> str:
    item = _data["by_id"][key]
    prs = " ".join(pr_link(p) for p in item["prs"])
    return (f'<div class="xf-evbox"><div class="xf-evbox-head">{badge(item["status"], item["verdict"])}'
            f'<span class="xf-muted">{e(LEVEL[item["level"]])} · {e(item["date"])} · PR {prs}</span></div>'
            f'<p><b>问题</b>　{e(item["question"])}</p><p><b>结论</b>　{e(item["finding"])}</p>'
            f'<div class="xf-evbox-links"><div><b>证据（固定到 PR 头提交）</b><ul class="xf-ev">{evidence_links(item)}</ul></div>'
            f'<div><b>原始输出（n2，未入库）</b><div>{raw_paths(item)}</div></div></div></div>')


def viz_catalog(_arg) -> str:
    out = ['<div class="xf-catalog"><input type="search" class="xf-filter" placeholder="筛选：输入实验名、PR 号、结论或路径…" '
           'aria-label="筛选数据目录" data-filter-target="xf-catalog-table">',
           '<div class="xf-scroll"><table id="xf-catalog-table"><thead><tr><th>实验</th><th>结论</th>'
           '<th>证据文件（永久链接）</th><th>原始输出（n2）</th><th>PR</th></tr></thead><tbody>']
    for item in sorted(_data["experiments"]["experiments"], key=lambda r: r["date"]):
        out.append(f'<tr><td><b>{e(item["name"])}</b><div class="xf-muted xf-small">{e(item["date"])} · '
                   f'{e(LEVEL[item["level"]])}</div></td><td>{badge(item["status"], item["verdict"])}</td>'
                   f'<td><ul class="xf-ev">{evidence_links(item)}</ul></td><td>{raw_paths(item)}</td>'
                   f'<td>{" ".join(pr_link(p) for p in item["prs"])}</td></tr>')
    out.append("</tbody></table></div></div>")
    return "".join(out)


# ---------------------------------------------------------------- references & archive

def viz_references(_arg) -> str:
    out, n = [], 0
    for group in _data["references"]["groups"]:
        out.append(f'<h2 id="ref-{e(group["id"])}">{e(group["name"])}</h2><ol class="xf-refs" start="{n + 1}">')
        for ref in group["refs"]:
            n += 1
            links = " · ".join(f'<a href="{e(l["url"])}">{e(l["label"])}</a>' for l in ref["links"])
            note = f'<div class="xf-ref-note">{e(ref["note"])}</div>' if ref.get("note") else ""
            cited = "；".join(ref.get("cited_in", []))
            out.append(f'<li id="ref-{e(ref["key"])}"><div>{e(ref["text"])}</div>'
                       f'{f"<div>{links}</div>" if links else ""}{note}'
                       f'<div class="xf-muted xf-small">出处：{e(cited)}</div></li>')
        out.append("</ol>")
    return "".join(out)


def _state_badge(info: dict) -> str:
    state = info.get("state", "?")
    if state == "MERGED":
        return '<span class="xf-chip xf-chip-merged">已合并</span>'
    if state == "CLOSED":
        return '<span class="xf-chip xf-chip-closed">已关闭</span>'
    return f'<span class="xf-chip xf-chip-open">{"草稿" if info.get("isDraft") else "开放"}</span>'


def viz_legacy(_arg) -> str:
    gh = _data["github"]
    out = ['<ol class="xf-timeline">']
    for era in _data["legacy"]["eras"]:
        files = "".join(f'<li><a href="{e(main_url(f["path"]))}">{e(f["label"])}</a> '
                        f'<span class="xf-muted">{e(f["path"])}</span></li>' for f in era.get("files", []))
        prs = "".join(f'<tr><td>{pr_link(n)}</td><td>{_state_badge(gh["prs"].get(str(n), {}))}</td>'
                      f'<td>{e(gh["prs"].get(str(n), {}).get("title", ""))}</td></tr>' for n in era.get("prs", []))
        issues = "".join(f'<tr><td>{issue_link(n)}</td><td>{_state_badge(gh["issues"].get(str(n), {}))}</td>'
                         f'<td>{e(gh["issues"].get(str(n), {}).get("title", ""))}</td></tr>' for n in era.get("issues", []))
        note = f'<p class="xf-muted xf-small">{e(era["note"])}</p>' if era.get("note") else ""
        details = ""
        if prs:
            details += (f'<details class="xf-tableview"><summary>PR（{len(era["prs"])} 个）</summary>'
                        f'<table><tbody>{prs}</tbody></table></details>')
        if issues:
            details += (f'<details class="xf-tableview"><summary>Issue（{len(era["issues"])} 个）</summary>'
                        f'<table><tbody>{issues}</tbody></table></details>')
        out.append(f'<li class="xf-era" id="era-{e(era["id"])}"><div class="xf-era-period">{e(era["period"])}</div>'
                   f'<div class="xf-era-body"><h3>{e(era["title"])}</h3><p>{e(era["summary"])}</p>{note}'
                   f'{f"<ul class=xf-ev>{files}</ul>" if files else ""}{details}</div></li>')
    out.append("</ol>")
    return "".join(out)


# ---------------------------------------------------------------- static figures

def viz_figure(name) -> str:
    return (Path(__file__).resolve().parents[1] / "portal" / "assets" / "figures" / f"{name}.svg").read_text(encoding="utf-8")


RENDERERS = {
    "kpis": viz_kpis, "roadmap": viz_roadmap, "forest": viz_forest, "v1a_curves": viz_v1a_curves,
    "running": viz_running, "next": viz_next, "evidence": viz_evidence, "catalog": viz_catalog,
    "references": viz_references, "legacy": viz_legacy, "figure": viz_figure,
}


def on_config(config, **kwargs):
    global _data
    _data = _load()
    return config


def on_page_markdown(markdown, page, config, files, **kwargs):
    global _prefix
    _prefix = "../" * page.url.count("/")
    def replace(match: re.Match) -> str:
        name, arg = match.group(1), match.group(2)
        if name not in RENDERERS:
            raise KeyError(f"{page.file.src_path}: unknown marker viz:{name}")
        # Raw HTML must sit on its own lines, separated by blank lines, to pass through Markdown untouched.
        return "\n\n" + RENDERERS[name](arg) + "\n\n"
    return MARKER.sub(replace, markdown)
