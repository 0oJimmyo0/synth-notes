# Workload Study Analysis Plan V1

Analyze the frozen 80-case randomized workload manifest without outcome access.

Primary analysis: compare reviewer active seconds per finalized contract-ready
case between `automation_assisted` and `full_manual`. Report median, IQR,
median difference, and a bootstrap 95% confidence interval. Fit a sensitivity
linear model of log active seconds with workflow condition and source-note-length
quintile. The single-reviewer result is operational evidence, not a clinical
effect estimate.

Safety analysis: report counts and rates of unsupported accepted candidates,
critical final omissions, medication-state/regimen errors, and incorrect final
dispositions. Any safety-stop event is reported individually.

Automation analysis: report candidate precision after reviewer adjudication,
manual-route sensitivity, safe pre-population yield, review burden, and
accepted/edited/rejected/added obligation counts overall and by field.

Complexity analysis: summarize time and reviewer actions by source-note-length
quintile, active-medication count, instruction count, follow-up presence, and
candidate-route burden. These are exploratory unless the final review form is
complete for every case.

Do not compare outcomes or open the 38 `heldout_gold` cases in this phase.
