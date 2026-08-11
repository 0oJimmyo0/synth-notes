# Workload Study Protocol V1

## Objective

Measure whether selective, evidence-linked assistance reduces reviewer workload
while preserving a human-approved final transition contract. This is a
scalability study, not an autonomous clinical-decision study.

## Design

- Population: fresh utility-train cases with unique patients and no outcome access.
- Sample size: 80 cases, balanced across five source-note-length quintiles.
- Reviewer design: one qualified reviewer, stratified random allocation to 40
  `full_manual` and 40 `automation_assisted` cases, eight per condition within
  each length quintile.
- Blinding: the reviewer does not see manual gold, V1.3 mismatch labels, or
  held-out-gold data. The assisted condition shows source-span-linked candidate
  obligations and their `AUTO_ACCEPT` or `MANUAL_REVIEW` state only.
- Final safety boundary: both conditions require human signoff of the finalized
  contract; no candidate is a final clinical decision.

## Frozen Endpoints

Primary endpoint: reviewer total seconds per finalized contract-ready case.

Secondary safety endpoints:

- unsupported candidate obligations accepted by the reviewer;
- critical obligations absent after final review;
- incorrect medication identity, action, state, dose, route, frequency,
  duration, or condition after final review;
- incorrect final disposition after final review.

Secondary efficiency endpoints:

- candidate obligations shown, accepted unchanged, edited, rejected, and added;
- safe pre-population yield;
- review burden;
- reviewer actions and corrections per case;
- proportion of assisted cases faster than the stratum-specific manual median.

## Time Measurement

Start timing when the reviewer opens the assigned case pack. Stop after the
reviewer records the final contract-ready or manual-route outcome. Record start
and end timestamps plus total seconds in the review form. Pauses longer than
five minutes must be recorded and excluded from active-review time.

## Safety Stop

Suspend the assisted arm immediately if a reviewer accepts an unsupported
high-risk candidate obligation or an unsafe case is auto-cleared. Preserve the
case, source span, and review record for adjudication. There is no interim
efficacy stopping rule; the planned sample remains 80 cases.

## Sequence

Freeze policy, manifest, assignments, review form, and analysis plan before
opening source cases. Complete the workload study on fresh utility-train data.
Only then consider the one-time 38-case held-out selective-safety evaluation.
