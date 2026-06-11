# Phase 1 Prep

This folder contains pre-generation and pre-audit scaffolding for the vanilla baseline only.

Contents:

- `run_audit_vanilla_generation.sh`
  - exact command to run `audit_vanilla_generation.py` after the official vanilla manifest exists
- `manual_review_rubric_50.csv`
  - blank manual review sheet for 50 sampled generated notes
- `*_template.csv`
  - empty result table templates for Phase 1 reporting
- `coverage_mapping_config.yaml`
  - coverage-mapping inputs and guards; do not run real-vs-synthetic coverage until generated notes are re-embedded and vanilla audit passes
- `factors.csv`
  - candidate metadata-factor spec for future CAV work; this is a planning scaffold, not a fitted axis-bank input yet

Phase 1 guardrail:

- Run vanilla audit first.
- Only if audit status is `PASS` or cautious `CAUTION` should coverage mapping proceed.
- Do not start CAV, LLM editor, or NER from this folder.
