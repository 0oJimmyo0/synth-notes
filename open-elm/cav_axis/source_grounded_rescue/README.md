# Source-Grounded Rescue

This package prepares a local-only, source-fact-conditioned feasibility study.
It deliberately separates fact extraction from generation so that failed
generation is not confused with an incorrect source ledger.

1. Select 40--60 held-out anchors from completed manual-review strata.
2. Build provisional ledgers with `build_source_fact_ledger.py`.
3. Manually verify every ledger fact and supporting span.
4. Validate ledger completeness with `validate_source_fact_ledger.py`.
5. Run three local-model arms: raw ELM, fact-conditioned correction, and
   fact-only generation.
6. Evaluate factuality first, then geometry and privacy.

The package does not yet invoke an editor/generator because no approved local
model path or inference interface has been selected. Do not send source notes
or ledgers to a third-party API.
