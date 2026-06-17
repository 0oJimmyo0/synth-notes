#!/usr/bin/env python3
"""
Build subgroup metadata for the MIMIC-IV discharge-summary cohort.

Outputs:
- a whole-cohort table aligned to split_manifest_note_level_full.csv
- a filtered-only table for rows retained after long-sequence filtering

The output is designed to support:
- whole-real manifold discovery summaries
- later held-out subgroup coverage summaries
- future CAV factor selection
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FULL_MANIFEST_PATH = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/"
    "data_note_hadm_all/clinic_notes/1_task/leakage_audit/split_manifest_note_level_full.csv"
)
DEFAULT_PATIENTS_PATH = "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/hosp/patients.csv"
DEFAULT_ADMISSIONS_PATH = "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/hosp/admissions.csv"
DEFAULT_SERVICES_PATH = "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/hosp/services.csv"
DEFAULT_ICUSTAYS_PATH = "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/icu/icustays.csv"
DEFAULT_OUTPUT_DIR = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/"
    "data_note_hadm_all/clinic_notes/1_task/subgroup_metadata"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build subgroup metadata for the discharge-summary cohort.")
    parser.add_argument("--full_manifest_path", default=DEFAULT_FULL_MANIFEST_PATH)
    parser.add_argument("--patients_path", default=DEFAULT_PATIENTS_PATH)
    parser.add_argument("--admissions_path", default=DEFAULT_ADMISSIONS_PATH)
    parser.add_argument("--services_path", default=DEFAULT_SERVICES_PATH)
    parser.add_argument("--icustays_path", default=DEFAULT_ICUSTAYS_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_git_commit(script_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(script_dir.parent), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def normalize_gender(value: object) -> str:
    text = str(value).strip().upper()
    if text == "F":
        return "F"
    if text == "M":
        return "M"
    return "unknown"


def collapse_race(value: object) -> str:
    text = str(value).strip().upper()
    if text in {"", "NAN"}:
        return "unknown"
    if "HISPANIC" in text or "LATINO" in text:
        return "hispanic_latino"
    if text.startswith("WHITE"):
        return "white"
    if text.startswith("BLACK") or "AFRICAN" in text:
        return "black"
    if text.startswith("ASIAN"):
        return "asian"
    if text in {"UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"}:
        return "unknown"
    return "other"


def normalize_insurance(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"", "nan"}:
        return "unknown"
    if "medicare" in text:
        return "medicare"
    if "medicaid" in text:
        return "medicaid"
    if "private" in text:
        return "private"
    if "self pay" in text or "self-pay" in text or "no charge" in text:
        return "self_pay_or_no_charge"
    if "government" in text:
        return "government"
    if "other" in text:
        return "other"
    return re.sub(r"\s+", "_", text)


def normalize_admission_type(value: object) -> str:
    text = str(value).strip().upper()
    if text in {"EW EMER.", "DIRECT EMER."}:
        return "emergency"
    if text == "URGENT":
        return "urgent"
    if text in {"EU OBSERVATION", "OBSERVATION ADMIT", "DIRECT OBSERVATION", "AMBULATORY OBSERVATION"}:
        return "observation"
    if text in {"ELECTIVE", "SURGICAL SAME DAY ADMISSION"}:
        return "scheduled"
    if text in {"", "NAN"}:
        return "unknown"
    return "other"


def normalize_service(value: object) -> str:
    text = str(value).strip().upper()
    if text in {"", "NAN"}:
        return "unknown"
    if text in {"MED", "CMED", "OMED", "NMED"}:
        return "medicine"
    if text in {"SURG", "NSURG", "CSURG", "VSURG", "TSURG", "PSURG"}:
        return "surgery"
    if text == "OBS":
        return "observation"
    if text == "ORTHO":
        return "orthopedics"
    if text == "PSYCH":
        return "psychiatry"
    if text == "TRAUM":
        return "trauma"
    if text == "GYN":
        return "gynecology"
    if text == "GU":
        return "genitourinary"
    if text == "ENT":
        return "ent"
    if text == "DENT":
        return "dental"
    if text == "EYE":
        return "ophthalmology"
    return text.lower()


def bin_age(age_value: float) -> str:
    if pd.isna(age_value):
        return "unknown"
    if age_value < 40:
        return "18-39"
    if age_value < 65:
        return "40-64"
    if age_value < 80:
        return "65-79"
    return "80+"


def bin_los(days: float) -> str:
    if pd.isna(days):
        return "unknown"
    if days <= 2:
        return "0-2"
    if days <= 6:
        return "3-6"
    if days <= 13:
        return "7-13"
    return "14+"


def build_service_table(path: Path) -> pd.DataFrame:
    services = pd.read_csv(path)
    services["transfertime"] = pd.to_datetime(services["transfertime"], errors="coerce")
    services = services.sort_values(["hadm_id", "transfertime"])
    first_service = services.groupby("hadm_id", as_index=False).first()[["hadm_id", "curr_service"]].rename(
        columns={"curr_service": "first_service_raw"}
    )
    last_service = services.groupby("hadm_id", as_index=False).last()[["hadm_id", "curr_service"]].rename(
        columns={"curr_service": "last_service_raw"}
    )
    merged = first_service.merge(last_service, on="hadm_id", how="outer")
    merged["service"] = merged["last_service_raw"].fillna(merged["first_service_raw"])
    merged["service_group"] = merged["service"].map(normalize_service)
    return merged


def build_icu_table(path: Path) -> pd.DataFrame:
    icu = pd.read_csv(path, usecols=["hadm_id", "stay_id"])
    icu = icu.dropna(subset=["hadm_id"]).copy()
    icu["hadm_id"] = icu["hadm_id"].astype(int)
    summary = icu.groupby("hadm_id").agg(icu_stay_count=("stay_id", "nunique")).reset_index()
    summary["icu_flag"] = summary["icu_stay_count"] > 0
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_manifest = pd.read_csv(args.full_manifest_path)
    patients = pd.read_csv(args.patients_path, usecols=["subject_id", "gender", "anchor_age", "anchor_year"])
    admissions = pd.read_csv(
        args.admissions_path,
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "admission_type",
            "insurance",
            "race",
        ],
    )

    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")
    admissions["admit_year"] = admissions["admittime"].dt.year
    admissions["los_days"] = (admissions["dischtime"] - admissions["admittime"]).dt.total_seconds() / 86400.0
    admissions["admission_type_group"] = admissions["admission_type"].map(normalize_admission_type)
    admissions["insurance_group"] = admissions["insurance"].map(normalize_insurance)
    admissions["race_ethnicity"] = admissions["race"].map(collapse_race)
    admissions["los_bin"] = admissions["los_days"].map(bin_los)

    patients["sex_gender"] = patients["gender"].map(normalize_gender)

    services = build_service_table(Path(args.services_path))
    icu = build_icu_table(Path(args.icustays_path))

    merged = full_manifest.merge(admissions, on=["subject_id", "hadm_id"], how="left", validate="many_to_one")
    merged = merged.merge(patients, on="subject_id", how="left", validate="many_to_one")
    merged = merged.merge(services, on="hadm_id", how="left", validate="many_to_one")
    merged = merged.merge(icu, on="hadm_id", how="left", validate="many_to_one")

    merged["age_at_admit"] = merged["anchor_age"] + (merged["admit_year"] - merged["anchor_year"])
    merged["age_at_admit"] = merged["age_at_admit"].where(merged["age_at_admit"].notna(), merged["anchor_age"])
    merged["age_bin"] = merged["age_at_admit"].map(bin_age)
    merged["icu_stay_count"] = merged["icu_stay_count"].fillna(0).astype(int)
    merged["icu_flag"] = merged["icu_flag"].fillna(False).astype(bool)

    merged["insurance"] = merged["insurance_group"]
    merged["admission_type"] = merged["admission_type_group"]
    merged["service"] = merged["service_group"]

    keep_cols = [
        "source_row_id",
        "embedding_row_id",
        "split",
        "dataset_row_id_full",
        "dataset_row_id",
        "kept_after_filter",
        "note_id",
        "subject_id",
        "hadm_id",
        "charttime",
        "sex_gender",
        "age_at_admit",
        "age_bin",
        "race_ethnicity",
        "insurance",
        "admission_type",
        "service",
        "insurance_group",
        "admission_type_group",
        "service_group",
        "first_service_raw",
        "last_service_raw",
        "los_days",
        "los_bin",
        "icu_flag",
        "icu_stay_count",
        "race",
        "admittime",
        "dischtime",
    ]
    subgroup_df = merged[keep_cols].copy()

    full_output = output_dir / "subgroup_metadata_full.csv"
    filtered_output = output_dir / "subgroup_metadata_filtered.csv"
    summary_output = output_dir / "subgroup_metadata_summary.json"

    subgroup_df.to_csv(full_output, index=False)
    subgroup_df.loc[subgroup_df["kept_after_filter"] == True].copy().to_csv(filtered_output, index=False)

    summary = {
        "created_at": now_iso(),
        "script_path": str(Path(__file__).resolve()),
        "git_commit": get_git_commit(Path(__file__).resolve().parent),
        "full_manifest_path": str(Path(args.full_manifest_path).resolve()),
        "patients_path": str(Path(args.patients_path).resolve()),
        "admissions_path": str(Path(args.admissions_path).resolve()),
        "services_path": str(Path(args.services_path).resolve()),
        "icustays_path": str(Path(args.icustays_path).resolve()),
        "output_dir": str(output_dir.resolve()),
        "n_full_rows": int(len(subgroup_df)),
        "n_filtered_rows": int((subgroup_df["kept_after_filter"] == True).sum()),
        "missing_rates": subgroup_df[
            ["sex_gender", "age_at_admit", "race_ethnicity", "insurance", "admission_type", "service", "los_days", "icu_flag"]
        ]
        .isna()
        .mean()
        .to_dict(),
        "top_value_counts": {
            column: subgroup_df[column].fillna("missing").astype(str).value_counts().head(10).to_dict()
            for column in [
                "age_bin",
                "sex_gender",
                "race_ethnicity",
                "insurance",
                "admission_type",
                "service",
                "los_bin",
                "icu_flag",
            ]
        },
    }
    summary_output.write_text(json.dumps(summary, indent=2, default=str))

    print("Saved full subgroup metadata to:", full_output)
    print("Saved filtered subgroup metadata to:", filtered_output)
    print("Saved subgroup metadata summary to:", summary_output)


if __name__ == "__main__":
    main()
