"""Split NeMo Safe-Synthesizer's one Berka table back into account and trans, as notebook 04 reads them.

NeMo writes each generated group's rows together, one group after another. It makes account_id the same within a
group, but two groups can get the same account_id. So an account here is a run of rows with the same account_id
in file order, not a value. Accounts are numbered from 0 in that order, and trans rows from 0, like the other
exports.

NeMo doesn't hold frequency and account_date the same within a group. Each account takes them from its first
row, as NeMo's own fix does for account_id (decided 6 Nov). How many accounts disagreed is recorded.

Values are kept as NeMo wrote them. Notebook 04 rounds and clips category codes itself.

Usage:
    python split_back_berka.py NEMO_CSV EXPORT_DIR

EXPORT_DIR gets account.csv, trans.csv and split_back.json. It must not hold any of them yet.
"""

import json
import sys
from pathlib import Path

import pandas as pd


GROUP_COL = "account_id"
PARENT_VALUES = ["frequency", "account_date"]
CHILD_VALUES = ["trans_date", "trans_type", "operation", "amount", "balance", "k_symbol", "bank", "account"]
JOINED_COLS = [GROUP_COL, *PARENT_VALUES, *CHILD_VALUES]
ACCOUNT_COLS = ["account_id", *PARENT_VALUES]
TRANS_COLS = ["trans_id", "account_id", *CHILD_VALUES]


def split_back(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return account, trans and a report from NeMo's one table."""
    assert sorted(table.columns) == sorted(JOINED_COLS), list(table.columns)
    assert len(table) > 0, "empty table"
    assert not table[GROUP_COL].isna().any(), "missing account_id"
    table = table[JOINED_COLS].reset_index(drop=True)
    run = (table[GROUP_COL] != table[GROUP_COL].shift()).cumsum().to_numpy() - 1
    first = ~pd.Series(run).duplicated().to_numpy()
    account = table.loc[first, PARENT_VALUES].reset_index(drop=True)
    account.insert(0, "account_id", range(len(account)))
    trans = table[CHILD_VALUES].copy()
    trans.insert(0, "account_id", run)
    trans.insert(0, "trans_id", range(len(trans)))
    by_run = table.assign(_run=run).groupby("_run", sort=True)
    disagree = {c: int((by_run[c].nunique(dropna=False) > 1).sum()) for c in PARENT_VALUES}
    any_disagree = int((by_run[PARENT_VALUES].nunique(dropna=False).max(axis=1) > 1).sum())
    runs_per_value = table.loc[first, GROUP_COL].value_counts()
    per_account = trans.groupby("account_id").size()
    report = {
        "rows": int(len(trans)),
        "accounts": int(len(account)),
        "rows_per_account": {"min": int(per_account.min()), "median": float(per_account.median()),
                             "p99": float(per_account.quantile(0.99)), "max": int(per_account.max())},
        "accounts_whose_parent_values_disagree": any_disagree,
        "accounts_disagreeing_by_column": disagree,
        "nemo_account_ids_used_by_more_than_one_account": int((runs_per_value > 1).sum()),
    }
    assert list(account.columns) == ACCOUNT_COLS and list(trans.columns) == TRANS_COLS
    return account, trans, report


def main(nemo_csv: str, export_dir: str) -> None:
    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("account.csv", "trans.csv", "split_back.json"):
        assert not (out / name).exists(), f"{out / name} exists"
    account, trans, report = split_back(pd.read_csv(nemo_csv))
    account.to_csv(out / "account.csv", index=False)
    trans.to_csv(out / "trans.csv", index=False)
    (out / "split_back.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
