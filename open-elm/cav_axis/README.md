# Clinical Note Enrichment: Current Status

## Current Decision

| Status | Evidence |
| --- | --- |
| Positive | Closed-loop generate--embed--filter selection can improve final-output local-basin landing over fair same-anchor vanilla. |
| Negative | Current ELM decoding does not preserve the source clinical encounter reliably enough for clinical synthetic-note use. |
| Frozen | New input-space steering families, candidate-count scale-up, large corpus generation, and downstream NER. |
| Active | Source-fact-conditioned rescue feasibility, with source faithfulness as the primary endpoint. |

The latest cluster-29 pooled-basin pilot generated 2,048 candidates from 256 held-out anchors and selected 106 notes. The selected condition improved automated geometry and quality relative to a same-anchor, one-draw, 8192-token vanilla control. However, blinded and source-paired review found that automatic gates did not detect clinically important errors: missing substantive sections, medication and numeric corruption, story stitching, and near-universal loss of the principal source encounter.

Therefore, a note that lands in the intended embedding region is **not** considered clinically useful unless it also preserves the source record and passes privacy-risk evaluation.

## Current Workflow

```text
freeze current evidence
-> review-calibrated triage and evaluation
-> verified source-fact ledger
-> raw ELM vs fact-conditioned correction vs fact-only generation
-> source-faithfulness review
-> re-embed faithful outputs and assess geometry/privacy
-> decide whether ELM text is a useful scaffold
```

The three-arm rescue pilot is intentionally small (about 40--60 held-out cases):

- **A: Raw ELM candidate.** Frozen geometry-positive, faithfulness-negative baseline.
- **B: Source-fact-conditioned correction.** A local approved model receives the raw ELM candidate plus a verified source-fact ledger and may only retain or express supported facts.
- **C: Source-fact-only generation.** The same model receives the same ledger without the raw ELM candidate. This determines whether the ELM text is a useful scaffold or harmful narrative context.

No source note or ledger may be sent to an unapproved external service. Source-paired files and ledgers remain only on approved MIMIC-IV project storage.

## Primary Decision Gate

Before geometry is considered, the rescue condition must materially improve source-paired outcomes relative to raw ELM:

- principal diagnosis preservation: at least 90%;
- unsupported major claims: at most 10%;
- critical source omissions: at most 15%;
- sex contradiction: zero;
- disposition contradiction: zero;
- major procedure preservation when applicable: at least 80%;
- clinically usable notes: at least 70%;
- no meaningful increase in copying or PHI-like risk.

Only clinically faithful outputs are then evaluated for pooled-basin retention, centroid distance, nearest-cluster transition, source cosine, diversity, and patient-disjoint sensitivity.

## Folder Layout

- `closed_loop_output_enrichment.py`: frozen closed-loop selection baseline.
- `closed_loop_train_text_privacy_screen.py`: training-corpus privacy-risk screen. Shared boilerplate must be reported separately from material copying.
- `build_matched_vanilla_8192_control.py`: fair same-anchor vanilla control.
- `prepare_closed_loop_validation_pack.py` and `build_source_faithfulness_review.py`: blinded and source-paired review packs.
- `clinical_validation/`: manual-label ingestion and deterministic triage. These tools prioritize review and remove obvious failures; they do not certify clinical validity.
- `source_grounded_rescue/`: local-only fact-ledger construction and validation for the three-arm rescue pilot.
- Historical CAV, local-transport, decoder-adaptation, and coverage scripts remain as reproducible negative/partial findings. They are not the active scale-up path.

## Immediate Commands

Build a provisional ledger from a manually selected held-out anchor manifest:

```bash
python cav_axis/source_grounded_rescue/build_source_fact_ledger.py \
  --anchor_manifest_path /approved/path/pilot_anchor_manifest.csv \
  --dataset_path /approved/path/encoded_testing_filtered \
  --split_manifest_path /approved/path/split_manifest_note_level.csv \
  --output_dir /approved/path/source_grounded_rescue/ledger_v1
```

Every generated fact starts as `pending` manual verification. Run ledger validation only after reviewers mark each fact as verified, corrected, or rejected.

## Historical CAV Work

The axis-bank and local-transport scripts are retained because they established an important result: pre-decode embedding geometry does not guarantee identity preservation through `ELM_decode -> BGE_reembed`. This motivated closed-loop output selection, and the source-paired review then showed that final embedding placement alone is still insufficient for clinical use.
