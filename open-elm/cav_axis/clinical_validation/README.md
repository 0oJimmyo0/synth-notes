# Clinical Validation

This package supports review-calibrated evaluation and triage. It is not a
clinical-validity certifier: semantic factuality must be checked against the
verified ledger.

## Phase 3a: local medication-judge feasibility

The initial task is deliberately narrow: reconcile compact verified medication
facts with the generated transition-note medication claims. It is evaluated as
a calibration/feasibility study, not used as an automatic gate.

- `build_medication_judge_dataset.py` creates restricted task JSONL files from
  reviewed notes. Cluster 25 is development-only; cluster 11 remains locked
  held-out evaluation data.
- `build_medication_judge_candidate_review_pack.py` samples additional existing
  cluster-25 candidates for blinded medication-focused development labels; it
  never changes generation or touches held-out cluster-11 data.
- `medication_judge_schema.json` defines the evidence-first output contract.
- `medication_judge_prompt_v1.txt` is the versioned prompt template. Calibrate
  candidate prompt versions only on cluster-25 development data, then freeze one.
- `provision_medgemma_snapshot.py` authenticates with a Hugging Face read token
  in memory only, records an immutable gated MedGemma revision, downloads the
  model-only snapshot to scratch, and writes checksums plus a manifest.
- `verify_medgemma_install.py` validates offline local BF16 MedGemma loading
  before it sees any restricted task. Provisioned model weights may live on
  scratch; task JSONL, judge outputs, and reports must remain on approved
  project storage.
- `medgemma_bf16_verify.slurm` runs that model-only BF16 verification with the
  requested local GPU allocation. It exports offline-only settings and reads
  no MIMIC-derived data.
- `smoke_test_medgemma_medication_judge.py` and
  `medgemma_fabricated_smoke.slurm` run five invented medication-reconciliation
  cases through MedGemma's local chat template. They validate serving, JSON
  parsing, and memory before the model is given any restricted task JSONL.
- `build_fabricated_medication_judge_tasks.py` creates the same five invented
  cases in the production task format. Run these through
  `medgemma_medication_judge.slurm` before submitting a restricted calibration
  task, so the compact prompt/schema/runner path is verified end-to-end.
- `build_medication_judge_adjudication_pack.py` creates a restricted review pack
  for any judge-rejected, repeat-unstable, or schema-invalid development task.
  Prior human labels are excluded so reviewers can determine whether a route is
  a true discrepancy, a nonmaterial difference, or a judge overcall.
- `subset_medication_judge_tasks.py` creates a deterministic restricted runtime
  preflight subset while removing any development labels. Use it to establish
  throughput before launching the full calibration set.
- `medication_judge_compact_schema_v2.json` and
  `medication_judge_prompt_v2_compact.txt` define the bounded Phase 3a contract:
  up to 12 evidence-cited medication findings and a final reject decision. The
  runner supplies only `discharge_medications` and `instructions` ledger facts,
  rather than asking the model to recreate every verified fact.
- `medication_judge_compact_schema_v3.json` and
  `medication_judge_prompt_v3_action_aware.txt` separate supported material
  rejection from human-review routing. They were designed from the completed
  label-blind cluster-25 route adjudication and remain development-only.
- `medication_judge_prompt_v3_1_action_aware.txt` is the follow-up development
  prompt that makes the no-discrepancy JSON representation explicit. It must be
  evaluated in a new output directory and never merged with the partial V3 run.
- `build_adjudicated_medication_reference.py` merges original blinded labels
  with route adjudication into a derived, text-free material-discrepancy
  reference table for development calibration.
- `build_prospective_medication_judge_review_pack.py` creates a reviewer-blind
  prospective pack containing all routed notes and a deterministic sample of
  non-routed notes. The separate route key must remain unopened until clinical
  labels are final.
- `run_local_medication_judge.py` supports a local model's native chat template,
  BF16, repeat stability checks, medication-only ledger evidence, and records
  raw output plus parse metadata. It writes a small run manifest with the
  expected output count and completion state; partial runs are not calibration
  inputs.
  `medgemma_medication_judge.slurm` is the offline two-A40 launcher for the
  frozen development calibration only. Do not submit held-out or prospective
  tasks until the development prompt/configuration is frozen.

For MedGemma calibration, `max_new_tokens` caps only structured judge JSON, not
synthetic-note generation. The compact contract defaults to 1536 tokens; every
response records whether it reached the cap. Any cap-hit or schema-invalid
result is routed to review and must not be treated as a valid automated decision.
The launcher can decode same-repeat requests in small batches; `BATCH_SIZE=2`
is the conservative two-A40 setting. A per-batch time limit can also be set;
time-limited outputs are recorded and review-only rather than silently accepted.
The analysis requires every requested repeat. Missing repeats, invalid JSON, or
repeat disagreement are all human-review routes, never evidence of stability.
- `validate_judge_json.py` checks schema conformance without exporting note or
  ledger text.
- `analyze_medication_judge.py` compares frozen judge decisions to human
  labels. It reports counts, sensitivity, specificity, and false rejection.
- `ingest_manual_review_labels.py` summarizes completed detailed review sheets
  without exporting source-note text.
- `deterministic_safety_checks.py` detects objective issues such as missing
  substantive sections, unfinished output, extreme numeric values, repeated
  blocks, and PHI-like patterns.

All task JSONL files contain compact verified ledger text and synthetic notes.
Keep them on approved project storage. Do not send them to external APIs. The
generation checkpoint must not be used as the primary judge; provision an
independent approved local instruction-following model before execution.
