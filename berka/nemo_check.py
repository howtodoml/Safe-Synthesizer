"""Check a NeMo scoring run's results file against the earlier one, then print NeMo's figures.

Usage:
    python nemo_check.py OLD_RESULTS_JSON NEW_RESULTS_JSON

OLD is the earlier results file for the same ClavaDDPM and REaLTabFormer exports, without NeMo. NEW is notebook
04's results file with NEMO_RUN set. It checks, before printing any NeMo figure:

1. Every figure OLD holds for ClavaDDPM and REaLTabFormer is the same in NEW. A regression figure may differ by a
   relative 1e-9. The unseeded KS and TV figures and the timings are left out, and counted if they differ.
2. NEW has NeMo's results, the nemo_run block, and the second cardinality check for all three methods.
3. The real train accounts with transactions number 3,378.

It prints "all checks passed" and exits 0, or lists what failed and exits 1.
"""

import hashlib
import json
import math
import sys
from pathlib import Path


TOL = 1e-9
REAL_ACCOUNTS_WITH_TRANS = 3378
OLD_METHODS = ("ClavaDDPM", "REaLTabFormer")
UNSEEDED = {"avg pval", "pval err", "num sigs", "frac sigs", "sigs cols"}
widest = {"relative": 0.0, "where": None}
skipped = {"differing": 0, "where": set()}


def same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def is_regression(path):
    return "/utility_regression_difference/" in path or "/regression/" in path


def close(a, b, path):
    if same(a, b):
        return True
    if not is_regression(path) or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a, b)):
        return False
    rel = abs(a - b) / max(abs(a), abs(b))
    if rel <= TOL and rel > widest["relative"]:
        widest.update(relative=rel, where=path)
    return rel <= TOL


def differences(old, new, path=""):
    """Every place where new differs from old, over the keys old has. New keys are allowed."""
    if isinstance(old, dict):
        if not isinstance(new, dict):
            return [path or "(top)"]
        out = []
        for k, v in old.items():
            if path.endswith("/ks_tv") and k in UNSEEDED:
                if k in new and not same(v, new[k]):
                    skipped["differing"] += 1
                    skipped["where"].add(k)
                continue
            if k == "wall_seconds" and (path.endswith("/utility_regression_difference") or path.endswith("/regression")):
                continue
            if path.endswith("/whole_set_check") and k in ("regression_wall_seconds", "regression_largest_relative_difference"):
                continue
            out += differences(v, new[k], f"{path}/{k}") if k in new else [f"{path}/{k} missing"]
        return out
    if isinstance(old, list):
        if not isinstance(new, list) or len(old) != len(new):
            return [path]
        out = []
        for i, (a, b) in enumerate(zip(old, new)):
            out += differences(a, b, f"{path}/{i}")
        return out
    return [] if close(old, new, path) else [path]


def f4(x):
    return f"{x:.4f}"


def main(old_path, new_path):
    old_p, new_p = Path(old_path), Path(new_path)
    for p in (old_p, new_p):
        print("read", p.name, hashlib.sha256(p.read_bytes()).hexdigest()[:8])
    old, new = json.loads(old_p.read_text()), json.loads(new_p.read_text())
    failed = []

    for m in OLD_METHODS:
        d = differences(old["results"][m], new["results"].get(m, {}), f"/results/{m}")
        print(m, "same" if not d else f"DIFFERS at {len(d)}: {d[:8]}")
        failed += d
    for key in ("split_manifest_sha256", "split_rows", "clipped_category_codes", "costs"):
        d = differences({m: old[key][m] for m in OLD_METHODS} if key in ("clipped_category_codes", "costs") else old[key],
                        {m: new[key].get(m) for m in OLD_METHODS} if key in ("clipped_category_codes", "costs") else new[key],
                        f"/{key}")
        print(key, "same" if not d else f"DIFFERS: {d[:8]}")
        failed += d
    d = differences(old["scoring_additions"], new["scoring_additions"], "/scoring_additions")
    print("scoring_additions", "same" if not d else f"DIFFERS: {d[:8]}")
    failed += d
    print(f"unseeded KS and TV figures that differed: {skipped['differing']}, in {sorted(skipped['where'])}")
    print("largest regression difference allowed:", f"{widest['relative']:.2e}", widest["where"] or "")

    nemo = new["results"].get("NeMo")
    block = new.get("nemo_run")
    if nemo is None or block is None:
        failed.append("NeMo results or nemo_run block missing")
    for m in (*OLD_METHODS, "NeMo"):
        if "cardinality_accounts_with_trans" not in new["results"].get(m, {}):
            failed.append(f"{m}: cardinality_accounts_with_trans missing")
    if block is not None and block.get("real_train_accounts_with_trans") != REAL_ACCOUNTS_WITH_TRANS:
        failed.append(f"real accounts with trans {block.get('real_train_accounts_with_trans')}, not {REAL_ACCOUNTS_WITH_TRANS}")

    if failed:
        print("CHECKS FAILED:", failed[:20])
        sys.exit(1)
    print("all checks passed")

    print("NeMo folder:", block["folder"])
    print("NeMo split-back:", json.dumps(block["split_back"]))
    print("NeMo cost:", json.dumps(new["costs"]["NeMo"]))
    for m in (*OLD_METHODS, "NeMo"):
        r = new["results"][m]
        ap, reg = r["alpha_precision"], r["utility_regression_difference"]
        g = r["groups_mia_auc_min100"]
        print(f"== {m}")
        print("  referential integrity", f4(r["relational"]["referential_integrity"]["valid_rate"]),
              "| children-per-parent KS complement", f4(r["relational"]["cardinality"]["ks_complement"]),
              "| with transactions only", f4(r["cardinality_accounts_with_trans"]["ks_complement"]))
        print("  correlation difference, parent x child", f4(r["corr_cross_table_fro"]),
              "| alpha-precision / beta-recall", f4(ap["delta_precision_alpha_OC"]), "/", f4(ap["delta_coverage_beta_OC"]))
        print("  utility F1 difference", f4(r["utility_f1_difference"]["mean_f1_difference_holdout"]),
              "| R2 / explained variance difference", f4(reg["avg_r2_difference"]), "/",
              f4(reg["avg_explained_variance_difference"]))
        print("  median DCR vs train / vs holdout", f4(r["median_dcr_vs_train"]["median_dcr_score"]), "/",
              f4(r["median_dcr_vs_holdout"]["median_dcr_score"]), "| attack AUC", f4(r["mia_dcr"]["auc"]))
        print("  attack per group, 100-row minimum:", {k: (f4(v["auc"]), v["n_members"]) for k, v in g["scored"].items()},
              "| not scored", g["not_scored"])
        print("  rare slices:", {s: (f4(e["auc"]), e["n_members"]) if e["scored"] else "not scored"
                                 for s, e in r["rare_slices"]["slices"].items()})
        print("  utility per group (rows, F1, R2, EV):", {
            k: (e["holdout_rows"], f4(e["f1"]["mean_f1_difference_holdout"]),
                f4(e["regression"]["avg_r2_difference"]), f4(e["regression"]["avg_explained_variance_difference"]))
            for k, e in r["utility_per_group"]["groups"].items()})


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
