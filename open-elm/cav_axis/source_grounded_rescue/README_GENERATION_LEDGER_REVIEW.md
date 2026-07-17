# Concise Generation Ledger Review

Use `generation_ledger_smoke_review_template_RESTRICTED.csv` for the four-case
text-generation smoke test. This file is restricted because the reviewer column
contains source-derived facts.

For every usable fact row:

1. Read `source_fact_value_for_reviewer` only to verify the fact.
2. Write a concise, factual `generation_value` in your own compact wording.
3. Set `generation_value_review_status` to `verified`, `corrected`, or
   `omit`.
4. Do not include supporting spans, long source passages, dates, names, phone
   numbers, or unsupported inferences in `generation_value`.

The prompt will receive only `fact_id`, `field`, and the reviewed
`generation_value`; it will not receive source offsets, supporting spans, or
the full source note. Medication facts may retain a drug, dose, route, and
frequency only when those details are explicitly verified.
