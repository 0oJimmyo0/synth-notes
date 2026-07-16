# Source-Fact-Conditioned Correction Prompt

Use only with an approved local model after every ledger fact has been manually
verified. This prompt is a template, not an authorization to send MIMIC data to
an external service.

```text
You are revising a synthetic discharge summary using a verified source-fact
ledger. Preserve only claims supported by the ledger. Remove unsupported claims
from the draft. Do not invent diagnoses, procedures, complications, laboratory
values, medications, doses, routes, dates, demographics, disposition, or
follow-up. If the ledger lacks a fact, omit it rather than guessing.

Write a concise discharge summary with only substantive sections supported by
the ledger. Do not copy supporting spans verbatim unless needed to preserve a
short medication name or required clinical term.

VERIFIED FACT LEDGER:
{verified_ledger_json}

RAW ELM DRAFT (optional; omit entirely for the fact-only arm):
{raw_elm_draft}
```
