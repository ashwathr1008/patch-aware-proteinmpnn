# Controlled patch scoring: first result and frozen expansion

Completed November 4, 2026. We scored the previously constructed ubiquitin pair with frozen vanilla and soluble ProteinMPNN `v_48_020` checkpoints. The original and permuted sequences differ at six positions within the eight-residue patch A:6–13. Backbone geometry, patch membership, amino-acid counts and identities outside the patch remain fixed in the computational input. Both sequences use the same eight explicitly recorded decoding orders, all-residue masks, evaluation mode and zero coordinate noise.

The primary score is full-chain mean negative log probability in **nats per residue**, averaged arithmetically across orders. The reported gap is permuted minus original; positive values favor the original assignment. This average is not the negative log of an order-marginalized joint probability. Each pair is scored within each checkpoint; gap magnitudes do not establish comparative checkpoint quality.

| Checkpoint | Original mean NLL | Permuted mean NLL | Mean gap | Original preferred |
| --- | ---: | ---: | ---: | ---: |
| Vanilla | 1.3561 | 1.5739 | +0.2178 | 8/8 orders |
| Soluble | 1.2531 | 1.4818 | +0.2287 | 8/8 orders |

Observed gaps ranged from +0.2044 to +0.2244 for vanilla and +0.2155 to +0.2402 for soluble. These are ranges across fixed orders, not confidence intervals. [Per-order scores](../../Analysis/paired_first/pair_scores.csv), [pair summaries](../../Analysis/paired_first/pair_summary.csv), [comparison figure](../../Analysis/paired_first/paired_gaps.pdf), and [order-sensitivity figure](../../Analysis/paired_first/order_sensitivity.pdf) preserve the measurements.

All **66 forward calls**, including audits, verified the actual decoding-order tensor used to construct the model's causal mask. Identical-sequence and reversed-presentation audits produced exactly zero differences for both checkpoints under every order. Explicit-order and rank-derived-order calculations also agreed exactly for each checkpoint's first order. Composition and shared-geometry controls tied. The simple hydrophobic-contact descriptor tied at zero for both arrangements across 15 eligible contacts. [Controls](../../Analysis/paired_first/pair_controls.csv) and [provenance/audits](../../Analysis/paired_first/run_record.json) are retained.

Secondary patch-position NLL-term gaps were +1.8701 for vanilla and +1.9898 for soluble. These are selected terms from full-chain autoregressive scoring, not isolated patch likelihoods; identity changes can also affect later outside-patch terms.

In plain language, shuffling the same eight amino-acid identities made this sequence less plausible to both existing models across every tested order. This establishes an arrangement-sensitive compatibility response in one example. It does not demonstrate a benefit from a new patch scorer, physical folding failure, improved solubility or reduced aggregation.

The [frozen expansion](../../Data/compatibility_pairs/metadata/summary.json) contains **85 pairs from 17 patches across seven source proteins and six conservative source groups**. Three diagnostic sources occupy three groups; the within-pilot similarity screen does not establish exclusion from pretraining. Fixed selection/overlap rules yielded four fewer patches than requested, with no threshold relaxation. **Expanded scoring is planned and has not yet been run in this report version.** Its purpose is to assess consistency and failures across proteins; variants and overlapping patches are dependent observations.
