import argparse, json
from pathlib import Path
from xtbflow.m0.t1x_data import build_cache

p = argparse.ArgumentParser()
p.add_argument("--h5", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)          # 例如 $XTBFLOW_DATA/t1x/t1x_m0_v1.npz
p.add_argument("--limit", type=int, default=None)          # 调试时用 --limit 50
a = p.parse_args()
manifest = build_cache(a.h5, a.out, a.limit)
a.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
print(json.dumps(manifest, indent=2, sort_keys=True))
