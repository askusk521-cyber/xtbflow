#!/usr/bin/env python3
"""Render a deterministic W13 evidence-status summary without plotting libraries."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from validate_claim_evidence import repository_root, validate_manifest


STATUS_COLORS = {
    "planned": "#6b7280",
    "observed": "#15803d",
    "blocked": "#b45309",
    "unresolved": "#b91c1c",
    "not_supported": "#7c3aed",
}


def _summary(manifest: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    rows = manifest["claims"]
    return {
        "schema": "xtbflow-claim-evidence-summary/v1",
        "manifest_id": manifest["manifest_id"],
        "manifest_status": manifest["status"],
        "claim_count": len(rows),
        "status_counts": validation["status_counts"],
        "evidence_level_counts": validation["evidence_level_counts"],
        "claims": [
            {
                "claim_id": row["claim_id"],
                "claim_type": row["claim_type"],
                "evidence_level": row["evidence_level"],
                "status": row["status"],
                "primary_metric": row["primary_metric"],
                "comparison_id": row["comparison_id"],
            }
            for row in sorted(rows, key=lambda item: item["claim_id"])
        ],
        "claim_limit": "summary only; no scientific result, method selection, or missing-evidence inference",
    }


def _svg(summary: dict[str, Any]) -> str:
    rows = summary["claims"]
    width = 960
    row_height = 34
    height = 80 + max(1, len(rows)) * row_height
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="28" font-family="sans-serif" font-size="18" font-weight="bold">Claim/evidence status</text>',
        f'<text x="24" y="51" font-family="sans-serif" font-size="12" fill="#374151">manifest={html.escape(summary["manifest_id"])} status={html.escape(summary["manifest_status"])}; summary only</text>',
    ]
    for index, row in enumerate(rows):
        y = 68 + index * row_height
        status = row["status"]
        color = STATUS_COLORS[status]
        claim = html.escape(row["claim_id"])
        label = html.escape(f'{row["claim_type"]} · {row["evidence_level"]} · {status}')
        out.extend(
            [
                f'<rect x="24" y="{y}" width="16" height="16" rx="3" fill="{color}"/>',
                f'<text x="52" y="{y + 13}" font-family="sans-serif" font-size="13">{claim}</text>',
                f'<text x="300" y="{y + 13}" font-family="sans-serif" font-size="12" fill="#374151">{label}</text>',
            ]
        )
    if not rows:
        out.append('<text x="24" y="90" font-family="sans-serif" font-size="13">no claims</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    parser.add_argument("--check-paths", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validation = validate_manifest(manifest, check_paths=args.check_paths, root=repository_root(args.manifest))
    summary = _summary(manifest, validation)
    for path in (args.output_json, args.output_svg):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_svg.write_text(_svg(summary), encoding="utf-8")
    print(json.dumps({"status": "completed", "claim_count": summary["claim_count"], "json": str(args.output_json), "svg": str(args.output_svg)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
