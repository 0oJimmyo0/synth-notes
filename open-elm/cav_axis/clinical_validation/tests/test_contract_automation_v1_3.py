#!/usr/bin/env python3
"""Regression checks for V1.3 instruction/follow-up atoms."""

import importlib.util
import sys
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "contract_automation_v1_3.py"
spec = importlib.util.spec_from_file_location("contract_automation_v1_3", MODULE)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_independent_action_predicates_split() -> None:
    atoms = module.atomize_actionable("Touch-down weight bear on the left leg and avoid alcohol.")
    assert [atom.action for atom in atoms] == ["weight bear", "avoid"]


def test_linked_follow_up_remains_one_atom() -> None:
    atoms = module.atomize_actionable("Follow up with Neurosurgery in one month with repeat cervical CT.")
    assert len(atoms) == 1
    assert atoms[0].section == "follow_up"


def test_unresolved_action_fragment_routes_manual() -> None:
    assert module.has_manual_fragment("You will see him as well as Dr.")
    assert module.has_manual_fragment("Call Dr.")


if __name__ == "__main__":
    test_independent_action_predicates_split()
    test_linked_follow_up_remains_one_atom()
    test_unresolved_action_fragment_routes_manual()
    print("v1.3 parser regression checks passed")
