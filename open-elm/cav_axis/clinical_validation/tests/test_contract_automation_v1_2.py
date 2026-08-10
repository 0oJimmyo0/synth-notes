#!/usr/bin/env python3
"""Regression checks for the v1.2 structured parser."""

import importlib.util
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "contract_automation_v1_2.py"
spec = importlib.util.spec_from_file_location("contract_automation_v1_2", MODULE)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def test_numbered_items_preserve_decimals() -> None:
    items = module.parse_numbered_medications("1. Albuterol 2.5 mg PO q6h 2. Docusate 100 mg PO BID")
    assert [item[0] for item in items] == ["Albuterol 2.5 mg PO q6h", "Docusate 100 mg PO BID"]


def test_action_clauses_split_independent_actions() -> None:
    clauses, unresolved = module.parse_action_clauses("Continue wound care daily and follow up with orthopedics in two weeks.")
    assert not unresolved
    assert [item[0] for item in clauses] == ["Continue wound care daily", "follow up with orthopedics in two weeks."]


def test_no_decimal_as_list_item() -> None:
    items = module.parse_numbered_medications("Albuterol 0.083% solution 2.5 mg PO daily")
    assert len(items) == 1


def test_dangling_action_text_is_truncated() -> None:
    assert module.TRUNCATED.search("Please return to the emergency department or notify your")


if __name__ == "__main__":
    test_numbered_items_preserve_decimals()
    test_action_clauses_split_independent_actions()
    test_no_decimal_as_list_item()
    test_dangling_action_text_is_truncated()
    print("v1.2 parser regression checks passed")
