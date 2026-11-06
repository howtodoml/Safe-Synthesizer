"""Write cost.json for a NeMo run's split-back export, so notebook 04 can score it.

Usage:
    python nemo_cost.py RUN_PATH EXPORT_DIR TAG CONFIG WHEEL

The times are NeMo's own, from the "Pipeline Timing" record in the run's JSON-lines logs. Training goes in
"training" and generation in "sampling", the names notebook 04 sums. The row counts are the split-back's.
Nothing in NeMo seeds training, so "seeded" is false. It stops if the timing record isn't found, or if
cost.json exists.
"""

import hashlib
import json
import sys
from pathlib import Path

KEYS = ("total_time_sec", "pii_replacer_time_sec", "training_time_sec", "generation_time_sec", "evaluation_time_sec")


def find_timing(obj):
    if isinstance(obj, dict):
        if "training_time_sec" in obj and "generation_time_sec" in obj:
            return obj
        for v in obj.values():
            hit = find_timing(v)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = find_timing(v)
            if hit is not None:
                return hit
    return None


def main(run_path, export_dir, tag, config, wheel):
    run, out = Path(run_path), Path(export_dir)
    cost = out / "cost.json"
    assert not cost.exists(), f"{cost} exists"
    found = []
    for log in sorted(run.rglob("*.jsonl")):
        for line in log.read_text(errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            t = find_timing(rec)
            if t is not None:
                found.append((str(log.relative_to(run)), t))
    assert found, f"no Pipeline Timing record in {run}"
    where, t = found[-1]
    report = json.loads((out / "split_back.json").read_text())
    c = {
        "rows": {"account": report["accounts"], "trans": report["rows"]},
        "training": {"wall_seconds": round(float(t["training_time_sec"]), 1)},
        "sampling": {"wall_seconds": round(float(t["generation_time_sec"]), 1)},
        "settings": {
            "run": tag,
            "config": Path(config).name,
            "config_sha256": hashlib.sha256(Path(config).read_bytes()).hexdigest(),
            "wheel": Path(wheel).name,
            "seeded": False,
            "nemo_timing": {k: t.get(k) for k in KEYS},
            "timing_from": where,
            "timing_records_found": len(found),
        },
    }
    cost.write_text(json.dumps(c, indent=2, sort_keys=True) + "\n")
    print(json.dumps(c, indent=2, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    main(*sys.argv[1:])
