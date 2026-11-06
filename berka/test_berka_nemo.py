"""Tests for the Berka join and split-back. Run with pytest, or as a script."""

import pandas as pd

from join_berka import join_tables
from split_back_berka import split_back


def toy():
    account = pd.DataFrame({"account_id": [7, 3, 9], "frequency": [0, 1, 2], "account_date": [10, 20, 30]})
    trans = pd.DataFrame({
        "trans_id": [5, 1, 4, 2, 3], "account_id": [3, 7, 3, 7, 3], "trans_date": [2, 1, 1, 1, 2],
        "trans_type": [0, 1, 2, 0, 1], "operation": [1, 1, 1, 1, 1], "amount": [1.5, 2.0, 3.0, 4.0, 5.0],
        "balance": [9.0, 8.0, 7.0, 6.0, 5.0], "k_symbol": [0, 0, 0, 0, 0], "bank": [0, 1, 0, 0, 0],
        "account": [0, 11, 0, 0, 0],
    })
    return account, trans


def test_join_order_and_drop():
    account, trans = toy()
    j = join_tables(account, trans)
    assert list(j["account_id"]) == [3, 3, 3, 7, 7]
    assert list(j["trans_date"]) == [1, 2, 2, 1, 1]
    assert list(j["amount"]) == [3.0, 5.0, 1.5, 2.0, 4.0]
    assert "trans_id" not in j.columns and 9 not in set(j["account_id"])


def test_round_trip_toy():
    account, trans = toy()
    a, t, report = split_back(join_tables(account, trans))
    assert list(a["account_id"]) == [0, 1] and list(a["frequency"]) == [1, 0]
    assert list(t["account_id"]) == [0, 0, 0, 1, 1] and list(t["trans_id"]) == [0, 1, 2, 3, 4]
    assert report["accounts_whose_parent_values_disagree"] == 0


def test_same_id_in_two_runs_is_two_accounts():
    account, trans = toy()
    j = join_tables(account, trans)
    j = pd.concat([j, j.iloc[:2]], ignore_index=True)
    a, t, report = split_back(j)
    assert len(a) == 3 and list(t["account_id"]) == [0, 0, 0, 1, 1, 2, 2]
    assert report["nemo_account_ids_used_by_more_than_one_account"] == 1


def test_first_row_wins_and_is_counted():
    account, trans = toy()
    j = join_tables(account, trans)
    j.loc[1, "account_date"] = 99
    j.loc[4, "frequency"] = 2
    a, _, report = split_back(j)
    assert list(a["account_date"]) == [20, 10] and list(a["frequency"]) == [1, 0]
    assert report["accounts_whose_parent_values_disagree"] == 2
    assert report["accounts_disagreeing_by_column"] == {"frequency": 1, "account_date": 1}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
