"""Join Berka's account columns onto trans, as one table for NeMo Safe-Synthesizer.

NeMo Safe-Synthesizer reads one table. Each trans row gets its account's frequency and account_date by an inner
join on account_id. Accounts with no transactions can't be carried, so they drop out. Rows are sorted by
account_id, trans_date and then trans_id, so ties within an account have a fixed order, and trans_id is then
dropped. Codes stay as the split's integers.

Usage:
    python join_berka.py SPLIT_PART_DIR OUT_CSV

SPLIT_PART_DIR holds account.csv and trans.csv, for example work/split/train in berka_comparison.
"""

import json
import sys
from pathlib import Path

import pandas as pd


GROUP_COL = "account_id"
ORDER_COL = "trans_date"
ACCOUNT_COLS = ["account_id", "frequency", "account_date"]
TRANS_COLS = ["trans_id", "account_id", "trans_date", "trans_type", "operation", "amount", "balance", "k_symbol",
              "bank", "account"]
JOINED_COLS = ["account_id", "frequency", "account_date", "trans_date", "trans_type", "operation", "amount",
               "balance", "k_symbol", "bank", "account"]


def join_tables(account: pd.DataFrame, trans: pd.DataFrame) -> pd.DataFrame:
    """Return the one-table view NeMo reads: trans rows with their account's columns, trans_id dropped."""
    assert list(account.columns) == ACCOUNT_COLS, list(account.columns)
    assert list(trans.columns) == TRANS_COLS, list(trans.columns)
    assert account[GROUP_COL].is_unique, "account_id repeats in account"
    assert trans["trans_id"].is_unique, "trans_id repeats in trans"
    orphans = int((~trans[GROUP_COL].isin(account[GROUP_COL])).sum())
    assert orphans == 0, f"{orphans} trans rows have no account"
    joined = trans.merge(account, on=GROUP_COL, how="inner", validate="many_to_one")
    assert len(joined) == len(trans), (len(joined), len(trans))
    joined = joined.sort_values([GROUP_COL, ORDER_COL, "trans_id"], kind="mergesort").reset_index(drop=True)
    return joined[JOINED_COLS]


def summary(account: pd.DataFrame, trans: pd.DataFrame, joined: pd.DataFrame) -> dict:
    per_account = joined.groupby(GROUP_COL).size()
    ties = int(trans.duplicated([GROUP_COL, ORDER_COL], keep=False).sum())
    return {
        "rows": int(len(joined)),
        "columns": int(joined.shape[1]),
        "accounts": int(joined[GROUP_COL].nunique()),
        "accounts_in": int(len(account)),
        "accounts_dropped": int(len(account) - joined[GROUP_COL].nunique()),
        "rows_per_account": {"min": int(per_account.min()), "median": float(per_account.median()),
                             "p99": float(per_account.quantile(0.99)), "max": int(per_account.max())},
        "rows_sharing_account_and_date": ties,
        "dtypes": {c: str(t) for c, t in joined.dtypes.items()},
    }


def main(part_dir: str, out_csv: str) -> None:
    part = Path(part_dir)
    out = Path(out_csv)
    assert not out.exists(), f"{out} exists"
    account = pd.read_csv(part / "account.csv")
    trans = pd.read_csv(part / "trans.csv")
    joined = join_tables(account, trans)
    joined.to_csv(out, index=False)
    print(json.dumps(summary(account, trans, joined), indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
