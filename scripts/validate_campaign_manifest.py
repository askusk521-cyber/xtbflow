#!/usr/bin/env python3
"""Validate a bounded W11 campaign manifest without running chemistry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from xtbflow.data.campaign import CampaignManifestError, load_campaign_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        manifest = load_campaign_manifest(args.manifest)
    except CampaignManifestError as exc:
        print(f"invalid campaign manifest: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "campaign_id": manifest.campaign_id,
                "status": manifest.status,
                "candidate_count": len(manifest.candidates),
                "comparison_count": len(manifest.comparisons),
                "input_fingerprint": manifest.input_fingerprint(),
                "claim_limit": manifest.claim_limit,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
