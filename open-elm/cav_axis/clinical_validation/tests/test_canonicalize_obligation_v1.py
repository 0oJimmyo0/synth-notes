#!/usr/bin/env python3
"""Regression checks for the frozen safe canonicalizer."""

import importlib.util
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "canonicalize_obligation_v1.py"
spec = importlib.util.spec_from_file_location("canonicalize_obligation_v1", MODULE)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def test_safe_formatting_normalization() -> None:
    assert module.canonicalize_obligation_text(" Follow-up with Orthopedics in two weeks. ") == "follow up with orthopedics in 2 weeks"


def test_clinically_material_details_remain_distinct() -> None:
    assert module.canonicalize_obligation_text("Hold medication daily") != module.canonicalize_obligation_text("Hold medication twice daily")
    assert module.canonicalize_obligation_text("Follow up in 2 weeks") != module.canonicalize_obligation_text("Follow up in 1 month")


if __name__ == "__main__":
    test_safe_formatting_normalization()
    test_clinically_material_details_remain_distinct()
    print("canonicalizer regression checks passed")
