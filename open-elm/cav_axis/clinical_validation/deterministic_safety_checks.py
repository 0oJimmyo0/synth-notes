#!/usr/bin/env python3
"""Apply conservative deterministic triage checks to generated notes.

The checks identify obvious defects for review prioritization. Passing this
script does not establish semantic factuality or clinical validity.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


SECTION_ALIASES = {
    "discharge_diagnosis": ("discharge diagnosis", "discharge diagnoses"),
    "hospital_course": ("brief hospital course", "hospital course"),
    "discharge_medications": ("discharge medications", "medications on discharge", "discharge medication"),
    "disposition": ("discharge disposition", "disposition"),
    "follow_up": ("followup instructions", "follow up instructions", "follow-up instructions", "follow up"),
    "instructions": ("discharge instructions", "instructions"),
}
NUMERIC_LIMITS = {
    "sodium": (100.0, 200.0),
    "potassium": (1.5, 8.0),
    "spo2": (0.0, 100.0),
    "o2sat": (0.0, 100.0),
    "oxygen saturation": (0.0, 100.0),
    "ph": (6.8, 7.8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic triage checks on generated-note manifests.")
    parser.add_argument("--input_path", required=True, help="JSONL manifest or CSV with generated_text.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_section_chars", type=int, default=40)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True) if path.suffix.lower() == ".jsonl" else pd.read_csv(path)


def normalized_lines(text: str) -> list[str]:
    return [line.strip() for line in str(text).replace("\r", "").split("\n")]


def substantive_sections(text: str, minimum_chars: int) -> dict[str, bool]:
    lines = normalized_lines(text)
    lowered = [line.lower() for line in lines]
    result: dict[str, bool] = {}
    for name, aliases in SECTION_ALIASES.items():
        start = next((index for index, line in enumerate(lowered) if any(line.startswith(alias + ":") or line == alias for alias in aliases)), None)
        if start is None:
            result[name] = False
            continue
        body = []
        for line in lines[start + 1 :]:
            if re.match(r"^[A-Za-z][A-Za-z /-]{2,80}:\s*$", line):
                break
            body.append(line)
        result[name] = len(re.sub(r"\W", "", " ".join(body))) >= minimum_chars
    return result


def impossible_numeric_values(text: str) -> list[str]:
    findings: list[str] = []
    for label, (lower, upper) in NUMERIC_LIMITS.items():
        pattern = re.compile(rf"\b{re.escape(label)}\b\s*[:=]?\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
        for match in pattern.finditer(str(text)):
            value = float(match.group(1))
            if value < lower or value > upper:
                findings.append(f"{label}={value:g}")
    return findings


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = read_table(input_path)
    if "generated_text" not in frame.columns:
        raise KeyError("input must contain generated_text")

    rows = []
    for index, row in frame.reset_index(drop=True).iterrows():
        text = str(row.get("generated_text", "") or "")
        sections = substantive_sections(text, args.min_section_chars)
        missing = [name for name, present in sections.items() if not present]
        numeric = impossible_numeric_values(text)
        lines = [line for line in normalized_lines(text) if line]
        repeated_line = any(lines.count(line) >= 3 and len(line) >= 30 for line in set(lines))
        terminal_fragment = bool(text.strip()) and not bool(re.search(r"[.!?]\s*$", text.strip()))
        phi_like = bool(re.search(r"\b\d{3}[-.)\s]\d{3}[-\s]\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", text))
        rows.append({
            "input_row_id": index,
            "candidate_id": row.get("candidate_id", row.get("blinded_note_id", "")),
            "missing_substantive_sections": "|".join(missing),
            "missing_substantive_section_count": len(missing),
            "impossible_numeric_values": "|".join(numeric),
            "impossible_numeric_flag": bool(numeric),
            "repeated_block_flag": repeated_line,
            "terminal_fragment_flag": terminal_fragment,
            "phi_like_pattern_flag": phi_like,
            "deterministic_triage_pass": not (missing or numeric or repeated_line or terminal_fragment or phi_like),
        })
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "deterministic_safety_checks.csv", index=False)
    summary = {
        "n_rows": int(len(results)),
        "triage_pass_rate": float(results.deterministic_triage_pass.mean()) if len(results) else 0.0,
        "missing_substantive_section_rate": float((results.missing_substantive_section_count > 0).mean()) if len(results) else 0.0,
        "impossible_numeric_rate": float(results.impossible_numeric_flag.mean()) if len(results) else 0.0,
        "warning": "This deterministic screen prioritizes review and cannot certify semantic source faithfulness.",
    }
    (output_dir / "deterministic_safety_checks_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
