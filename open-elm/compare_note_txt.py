#!/usr/bin/env python3
"""
Compare synthetic note .txt files that use the format:
=== Note 1 ===
<note body>
=== Note 2 ===
<note body>
...

Supports:
1) Comparing two files directly
2) Comparing all pairwise combinations in a directory
"""

import argparse
import difflib
import hashlib
import itertools
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


NOTE_PATTERN = re.compile(r"=== Note \d+ ===\n(.*?)(?=\n=== Note |\Z)", re.DOTALL)


@dataclass
class FileStats:
    path: str
    file_size_bytes: int
    note_count: int
    avg_note_chars: float
    median_note_chars: float


def parse_notes(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return [m.strip() for m in NOTE_PATTERN.findall(content)]


def normalized_note(note: str) -> str:
    # Normalize line endings and collapse extra whitespace for robust matching.
    note = note.replace("\r\n", "\n").replace("\r", "\n")
    note = re.sub(r"[ \t]+", " ", note)
    note = re.sub(r"\n{3,}", "\n\n", note)
    return note.strip()


def median(values: List[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def compute_file_stats(path: str, notes: List[str]) -> FileStats:
    lengths = [len(n) for n in notes]
    return FileStats(
        path=path,
        file_size_bytes=os.path.getsize(path),
        note_count=len(notes),
        avg_note_chars=(sum(lengths) / len(lengths)) if lengths else 0.0,
        median_note_chars=median(lengths),
    )


def hash_counts(notes: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for note in notes:
        h = hashlib.sha256(normalized_note(note).encode("utf-8")).hexdigest()
        counts[h] = counts.get(h, 0) + 1
    return counts


def multiset_intersection_size(a: Dict[str, int], b: Dict[str, int]) -> int:
    return sum(min(a.get(k, 0), b.get(k, 0)) for k in set(a) | set(b))


def compare_two_files(path_a: str, path_b: str, max_diff_samples: int) -> Dict:
    notes_a = parse_notes(path_a)
    notes_b = parse_notes(path_b)
    stats_a = compute_file_stats(path_a, notes_a)
    stats_b = compute_file_stats(path_b, notes_b)

    hashes_a = hash_counts(notes_a)
    hashes_b = hash_counts(notes_b)
    shared_exact_notes = multiset_intersection_size(hashes_a, hashes_b)

    min_len = min(len(notes_a), len(notes_b))
    index_exact_match = 0
    index_mismatch_samples: List[Dict[str, str]] = []

    for i in range(min_len):
        na = normalized_note(notes_a[i])
        nb = normalized_note(notes_b[i])
        if na == nb:
            index_exact_match += 1
            continue
        if len(index_mismatch_samples) < max_diff_samples:
            diff = "\n".join(
                difflib.unified_diff(
                    na.splitlines(),
                    nb.splitlines(),
                    fromfile=f"{os.path.basename(path_a)} note[{i + 1}]",
                    tofile=f"{os.path.basename(path_b)} note[{i + 1}]",
                    lineterm="",
                    n=2,
                )
            )
            index_mismatch_samples.append(
                {
                    "note_index_1_based": str(i + 1),
                    "diff_sample": diff[:4000],
                }
            )

    a_only_count = max(0, len(notes_a) - min_len)
    b_only_count = max(0, len(notes_b) - min_len)

    return {
        "file_a": vars(stats_a),
        "file_b": vars(stats_b),
        "pair_metrics": {
            "notes_compared_by_index": min_len,
            "index_exact_match_count": index_exact_match,
            "index_exact_match_ratio": (index_exact_match / min_len) if min_len else 0.0,
            "shared_exact_notes_any_order_count": shared_exact_notes,
            "shared_exact_notes_any_order_ratio_vs_smaller_file": (
                shared_exact_notes / min(len(notes_a), len(notes_b))
            )
            if min(len(notes_a), len(notes_b))
            else 0.0,
            "extra_notes_in_a_after_min_len": a_only_count,
            "extra_notes_in_b_after_min_len": b_only_count,
        },
        "index_mismatch_samples": index_mismatch_samples,
    }


def print_summary(result: Dict) -> None:
    a = result["file_a"]
    b = result["file_b"]
    m = result["pair_metrics"]

    print("=" * 80)
    print(f"A: {a['path']}")
    print(f"   size={a['file_size_bytes']} bytes, notes={a['note_count']}, "
          f"avg_chars={a['avg_note_chars']:.1f}, median_chars={a['median_note_chars']:.1f}")
    print(f"B: {b['path']}")
    print(f"   size={b['file_size_bytes']} bytes, notes={b['note_count']}, "
          f"avg_chars={b['avg_note_chars']:.1f}, median_chars={b['median_note_chars']:.1f}")
    print("-" * 80)
    print(f"Index exact matches: {m['index_exact_match_count']}/{m['notes_compared_by_index']} "
          f"({100*m['index_exact_match_ratio']:.2f}%)")
    print("Shared exact notes (any order): "
          f"{m['shared_exact_notes_any_order_count']} "
          f"({100*m['shared_exact_notes_any_order_ratio_vs_smaller_file']:.2f}% of smaller file)")
    print(f"Extra trailing notes in A: {m['extra_notes_in_a_after_min_len']}")
    print(f"Extra trailing notes in B: {m['extra_notes_in_b_after_min_len']}")

    if result["index_mismatch_samples"]:
        print("-" * 80)
        print("Mismatch samples:")
        for sample in result["index_mismatch_samples"]:
            print(f"\n[Note index {sample['note_index_1_based']}]")
            print(sample["diff_sample"])
    print("=" * 80)


def list_txt_files(directory: str, pattern: str) -> List[str]:
    regex = re.compile(pattern)
    files = []
    for name in os.listdir(directory):
        full = os.path.join(directory, name)
        if os.path.isfile(full) and name.endswith(".txt") and regex.search(name):
            files.append(full)
    files.sort()
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare clinical note .txt files")
    parser.add_argument("--file-a", type=str, help="Path to first .txt file")
    parser.add_argument("--file-b", type=str, help="Path to second .txt file")
    parser.add_argument("--dir", type=str, help="Directory containing .txt files")
    parser.add_argument(
        "--name-regex",
        type=str,
        default=r".*",
        help="Regex filter for filenames when using --dir",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="When using --dir, compare all file pairs",
    )
    parser.add_argument(
        "--max-diff-samples",
        type=int,
        default=2,
        help="Maximum number of per-note mismatch diff samples to print",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional output JSON path for machine-readable results",
    )
    args = parser.parse_args()

    results: List[Dict] = []

    if args.file_a and args.file_b:
        result = compare_two_files(args.file_a, args.file_b, args.max_diff_samples)
        print_summary(result)
        results.append(result)
    elif args.dir:
        files = list_txt_files(args.dir, args.name_regex)
        if len(files) < 2:
            raise ValueError(f"Need at least 2 matching .txt files in {args.dir}")
        pairs = itertools.combinations(files, 2) if args.all_pairs else [(files[0], files[1])]
        for file_a, file_b in pairs:
            result = compare_two_files(file_a, file_b, args.max_diff_samples)
            print_summary(result)
            results.append(result)
    else:
        raise ValueError("Use --file-a/--file-b OR --dir")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved JSON results to: {args.output_json}")


if __name__ == "__main__":
    main()
