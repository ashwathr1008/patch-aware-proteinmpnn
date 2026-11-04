# Controlled patch scoring: completed pilot

Frozen vanilla/soluble ProteinMPNN `v_48_020` use identical backbone coordinates, masks and eight verified decoding orders, evaluation mode and zero noise. Positive gaps mean **permuted minus original full-chain mean NLL**, in nats/residue. Arithmetic averaging across orders is not an order-marginalized joint likelihood. Patch-position terms are selected full-chain terms, not isolated patch likelihoods.

First ubiquitin pair: six changed positions within A:6–13.

| Checkpoint | Original NLL | Permuted NLL | Gap | Original preferred |
| --- | ---: | ---: | ---: | ---: |
| Vanilla | 1.3561 | 1.5739 | +0.2178 | 8/8 orders |
| Soluble | 1.2531 | 1.4818 | +0.2287 | 8/8 orders |

Its 66 forward calls verified actual decoding orders; null/reversed-presentation differences and first-order explicit/rank-derived differences were exactly zero. [First-pair provenance](../../Analysis/paired_first/run_record.json).

The expansion completed **85 pairs, 17 patches, seven sources and six conservative source groups**: 1,360 paired-order rows, 2,958 audited forward calls, 41.468 seconds. All 112 null/reversed-presentation audits and explicit/rank-derived checks agreed exactly. Independent recomputation from 2,720 per-residue arrays found zero arithmetic discrepancies. Fixed rules retained four fewer patches than requested; eight patches are discontinuous in sequence.

Both checkpoints favored original assignments in **84/85 order-averaged pairs**.

| Source | Split | Pairs | Vanilla gap | Soluble gap | Original: vanilla | Original: soluble |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1UBQ:A | Development | 15 | +0.1910 | +0.2046 | 15/15 | 15/15 |
| 1VII:A | Development | 15 | +0.2224 | +0.2251 | 15/15 | 15/15 |
| 1CRN:A | Development | 15 | +0.5741 | +0.5463 | 15/15 | 15/15 |
| 2CI2:I | Development | 15 | +0.2422 | +0.2296 | 15/15 | 15/15 |
| 1L2Y:A | Diagnostic | 5 | +0.7651 | +0.7357 | 5/5 | 5/5 |
| 1ENH:A | Diagnostic | 5 | +0.1602 | +0.1770 | 4/5 | 4/5 |
| 1SHG:A | Diagnostic | 15 | +0.2532 | +0.2400 | 15/15 | 15/15 |

The three diagnostic protein/group rates average **93.3%** equally; pooling gives 24/25 pairs (96%). Neither is an independent-sample inferential estimate. All 85 composition/geometry controls tied; the spatial descriptor changed in 41 pairs: 26 increases, 15 decreases and 44 ties. No descriptor direction is labeled favorable. [Tables](../../Analysis/paired_expanded/protein_summary.csv), [split summaries](../../Analysis/paired_expanded/split_summary.csv), [gap figure](../../Analysis/paired_expanded/paired_gaps.pdf), [descriptor figure](../../Analysis/paired_expanded/spatial_descriptor.pdf), [audits](../../Analysis/paired_expanded/run_record.json).

## Failure/edge case

`1ENH_A_patch_01_perm_005` swaps R/E at PDB A:18–19 (observed indices 16–17). Mean gaps are −0.009217 vanilla and −0.002422 soluble. Preference reverses across orders: original wins 1/8 and 2/8, respectively. Its hydrophobic descriptor ties. [Order values](edge_case_orders.csv), [order-sensitivity figure](../../Analysis/paired_expanded/order_sensitivity.pdf). This is a model-preference exception, not a biological negative. Follow-up should examine more independent proteins and stratify by changed-position count, with order settings frozen before evaluation.

Existing ProteinMPNN detects most rearrangements despite identical ingredient counts. Decoding blindness, added-scorer benefit and folding/aggregation improvement remain unestablished. Input limitations include designed 1L2Y, first NMR conformers, truncated domains and unconstrained disulfides. Original need not mean natural native. Pretraining overlap, dependent variants and tiny diagnostic groups limit generalization.
