from __future__ import annotations

import argparse
from pathlib import Path
import json

from src.labels.apply import apply_labels_to_flows_parquet


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply deterministic labels to flows parquet and write labeled outputs.")
    ap.add_argument("--dataset", choices=["vnat", "iscx"], required=True)
    ap.add_argument("--flows", type=str, required=True, help="Path to flows.parquet")
    ap.add_argument("--out", type=str, default=None, help="Optional output parquet path")
    ap.add_argument("--manifest", type=str, default=None, help="Optional manifest json path")

    args = ap.parse_args()

    flows = Path(args.flows)
    out = Path(args.out) if args.out else None
    manifest = Path(args.manifest) if args.manifest else None

    m = apply_labels_to_flows_parquet(
        flows_parquet=flows,
        dataset=args.dataset,
        out_parquet=out,
        out_manifest=manifest,
    )

    print(json.dumps({
        "dataset": m["dataset"],
        "output": m["output"],
        "derived_label_counts": m["derived"]["label_counts"],
        "consistency": m.get("consistency", None),
    }, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
