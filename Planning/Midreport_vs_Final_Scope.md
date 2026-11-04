# Mid-report and final-project scope

Updated November 4, 2026. Mid-report: due November 6, draft targeted November 5. The proposal lists December 6 for the final report/poster. This is a **recommended narrower final scope for team review**; some original commitments become conditional extensions.

The question is whether evaluating amino-acid arrangement in spatial patches adds information beyond composition and frozen ProteinMPNN. ProteinMPNN already uses geometry and sequence dependencies; its weaknesses must be demonstrated.

| | Mid-report: establish a defensible experiment | Final: test whether the proposed addition is useful |
| --- | --- | --- |
| Main question | How does frozen ProteinMPNN respond to rearrangements with unchanged composition and backbone? | Does a patch scorer add predictive or selection value on independent proteins? |
| Method | Validated pairs on several proteins; composition/geometry controls; a simple spatial descriptor; vanilla/soluble ProteinMPNN with verified shared decoding orders. | One scorer if justified; protein/family-separated training and evaluation; composition/geometry ablations and strong frozen-model baselines. |
| Outputs | Dataset table, patch illustration, per-protein score comparison, order sensitivity, failure analysis, reproducible artifacts and complete report PDF. | Held-out evaluation; reranking the same fixed candidate pools with equal selection budgets; bounded structural checks; independent measured-outcome evaluation where mapping is feasible. |
| Supported claims | Valid controls and model sensitivity to these arrangements. No demonstrated design improvement yet. | Added value only where comparisons support it. Folding and measured solubility/aggregation remain separate outcomes. |

The proposal expected a trained scorer by October 26 and property heads, region-specific scores and reranking by November 2. **Those milestones have not been achieved.** The mid-report must acknowledge the revised schedule. The first scored pair checks execution; the expanded analysis supplies the preliminary scientific result. Neither automatically establishes a need for another model.

The recommended final core establishes one scorer's added value and tests candidate selection against separate outcomes. Improving the same score used for ranking is expected by construction. Audit measured labels and sequence/structure mappings before committing to solubility/aggregation evaluation. Native-versus-shuffled pairs are **compatibility controls, not measured aggregation labels**. If measured outcomes cannot be assembled, keep the final claims about compatibility; structural confidence cannot substitute for those measurements.

Fine-tuning/DPO, multiple property heads, SKEMPI/interface prediction, region-specific optimization and broad binder benchmarks become **conditional extensions**. Reward-based fine-tuning needs a validated reward; SKEMPI needs its own task and evaluation.

## Work still needed for the mid-report

1. Finish/audit expanded scoring: mappings, exclusions, patch overlap, composition ties, decoding orders and per-protein results. Keep source-protein descendants together; this convenience panel is not a representative benchmark.
2. Inspect failures and order sensitivity. Distinguish sequence identity from structural recovery and arrangement sensitivity from added predictive value.
3. Audit one measured-outcome dataset: label meaning, access, mapping coverage, exclusions and feasible protein/family splits. Report feasibility honestly if validation remains unfinished.
4. Finish background, methods, actual results, limitations and revised timeline. Compile the report PDF and cross-check claims/figures against outputs.
5. Review final scope and deferrals after seeing the pilot and label audit. Keep local copies; PSC availability must not block report completion.

Remaining work will be divided at today's end; no new owners are assigned here. See the [detailed plan](Midreport_Plan_2026-11-04.md) and [instructor feedback](../Documentation/Feedback%20on%20Proposal.txt).
