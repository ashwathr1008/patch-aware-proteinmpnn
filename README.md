# Patch-aware protein design with ProteinMPNN

Course project for 02-741 Generative AI for Biomedicine, Fall 2026.

**Team:** Ashwath Raghuram, Kirti Hargunani, and Vani Nigam.

We investigate whether explicitly scoring local arrangements of residues on protein surfaces adds useful information beyond amino-acid composition and frozen ProteinMPNN. The eventual motivation is improved developability while preserving structural and functional compatibility. Native-versus-permuted patches provide compatibility controls; they are not measured aggregation labels.

## Mid-report scope

Due November 6; target a complete draft November 5 evening. Deliver a working baseline, a small controlled patch experiment, quantitative preliminary results, and a plan for independent solubility/developability validation. See the [detailed plan](Planning/Midreport_Plan_2026-11-04.md).

| Owner | Mid-report responsibility and completed work |
|---|---|
| Ashwath | **Completed:** local and PSC setup with verified retrieval; reproducible ProteinMPNN baseline (48 generated sequences across three structures and two checkpoints, plus a repeat check); first patch control and illustration; expanded, validated dataset (85 matched pairs, 17 patches, seven structures); controlled scoring with two checkpoints and eight shared decoding orders; composition/geometry controls, spatial-descriptor comparison, per-protein results/figures, and order-sensitive exception analysis; reproducible code, curated data/results, and [baseline methods](Documentation/Midreport/ashwath_section.md) and [controlled-scoring results](Documentation/Midreport/controlled_scoring_results.md) drafts. |
| Kirti | Expand and validate patches and matched permutations; comparison plot and patch illustration; experiment methods/results |
| Vani | Bounded motivation/related-work and validation-dataset investigation; evaluation and limitations text, depending on availability |

Each person writes the paragraphs for their own work. 

**November 4 update:** Ashwath's row now records completed work, including the dataset expansion, patch illustration, comparison figures, and experiment methods/results originally listed under Kirti. Kirti's and Vani's rows retain their initial provisional responsibilities; we will divide the remaining work at the end of today's session. Do not repeat completed tasks from the old handoff. Team review, the validation-dataset audit, remaining report text, and final report assembly are still outstanding.

## Current status — November 4

- PSC access, staged ProteinMPNN source/weights, and checksum-verified retrieval have been exercised.
- The first PSC test, job `50214872`, failed on a V100 because the existing PyTorch CUDA build could not execute on that GPU. Its 17 files were recovered and verified locally. This is an infrastructure finding, not a biological result.
- The local RTX 4090 Laptop GPU completed the development baseline: **48 generated sequences**, three structures, two frozen checkpoints, eight samples per combination. All six runs passed validation. A repeat of 1UBQ reproduced all 16 generated sequences and scores exactly on the same machine.
- A checked eight-residue 1UBQ patch permutation preserves geometry, membership, composition and outside-patch context; its mapping agrees with the ProteinMPNN input at all 76 residues and 304 backbone atoms.
- [Ashwath's report draft](Documentation/Midreport/ashwath_section.md), [results and figures](Analysis/README.md), and [Kirti's handoff](Planning/Kirti_Handoff_2026-11-04.md) contain the completed work and next steps.
- Controlled scoring of the previously illustrated ubiquitin pair is complete: both frozen checkpoints prefer the original arrangement under all eight explicit shared decoding orders. The [scoring results](Documentation/Midreport/controlled_scoring_results.md) explain the gaps and limitations.
- The frozen expansion is scored: **85 pairs, 17 patches, seven structures, seven homology clusters, two checkpoints and eight shared orders**. Both checkpoints prefer the original in 84/85 pairs by mean score, with one small, order-sensitive exception in 1ENH. [Expanded tables and figures](Analysis/paired_expanded/) retain every result. These are compatibility observations, not evidence of improved folding or solubility; no new patch scorer has been trained.
- PSC H100 smoke job `50216628` completed successfully; all 22 files were retrieved and checksum-verified locally. Its eight generated sequences are an infrastructure check separate from the baseline and paired experiment.
- [Mid-report versus final scope](Planning/Midreport_vs_Final_Scope.md) explains what the current study can claim and what the final project must establish. Remaining team assignments will be decided at the end of today's work.
- No training or experiment starts automatically by cloning this repository or running its tests.

## Next-step plan: supervised stability scorers

**Goal:** test whether learning stability-specific position preferences helps, and whether adding residue interactions helps further. This is a proposed folding-stability experiment; neither scorer is implemented yet, and stability is a separate endpoint from aggregation or binding.

| Scorer | One concrete method | What the comparison tests |
| --- | --- | --- |
| Supervised additive scorer | A small shared neural network scores each residue from its amino-acid identity and fixed backbone geometry; sum the residue scores. | Benefit of learning stability-specific position preferences. |
| Supervised interaction scorer | The same additive model plus a small symmetric pair network over nonadjacent Cα contacts within 8 Å. Pair inputs are both identities, distance and sequence separation; count each contact once. | Additional predictive benefit from explicit identity interactions. |

1. **Use measured stability data.** Use the pinned [Tsuboyama et al. (2023)](https://doi.org/10.1038/s41586-023-06328-6) [MegaScale release](https://zenodo.org/records/7992926), version `v2_230420`, described below. Audit quality and usable structure mappings before committing to a training-set size. Our original/shuffled pairs have no experimental stability labels and cannot supply these targets.
2. **Prepare examples and labels.** Use the published numeric `ddG_ML` target (positive means stabilizing, kcal/mol), rather than subtracting an arbitrary wild-type replicate. Each example needs `WT_name`, reference/variant sequences, mutation positions and a mapped parent backbone. Preserve uncertainty fields and exclude unusable/censored labels under the rules below. Keep all variants of a parent and close homologues in the same train/validation/test group.
3. **Build controlled features.** Use amino-acid identity and backbone-only features such as neighbor counts and contact distances. Keep geometry identical for a parent and its variants. The additive model must not see neighboring identities or sequence-conditioned embeddings. Train domain-level summed scores against domain-level labels; do not copy a whole-protein label onto individual patches. Use existing surface patches for subsequent inspection.
4. **Train both scorers the same way.** Predict the difference between variant and reference scores. Use Huber regression loss, AdamW and early stopping on validation proteins. Train small models locally, balance contributions across parent proteins, and use the same examples, splits and tuning budget. Keep ProteinMPNN frozen. Include a wider additive model as a capacity control.
5. **Test added value.** On held-out protein families, compare composition features, frozen ProteinMPNN, the additive scorer and the interaction scorer using per-protein Spearman correlation and MAE. Report variation across proteins and training seeds. Where matched single/double mutants exist, also test nonadditivity: double-mutant effect minus the two single-mutant effects. This directly challenges the additive model, whose predicted nonadditivity is zero on fixed geometry.
6. **Decide whether to proceed.** Use the interaction scorer for candidate reranking only if it adds held-out value over the additive and frozen-model baselines. Higher predicted stability alone is not proof of improved folding, solubility or binding; arbitrary shuffled sequences may be outside the training distribution.

**Deliverables:** versioned data/split manifest in `Data/`, scorer and training code in `Code/`, and held-out tables, plots and limitations in `Analysis/`. Acquisition is separate from training: the downloaded release still needs filtering, structure mapping and splits before any scorer training or evaluation.

### Exact dataset, locations and methods

**Source:** MegaScale / Tsuboyama 2023, Zenodo record **7992926**, version **v2_230420**, DOI **10.5281/zenodo.7992926**, license **CC BY 4.0**. Retain the paper citation and dataset attribution. Local archives are in **`Data/raw/megascale/`**; [Data/megascale_manifest.json](Data/megascale_manifest.json) records source URLs, exact sizes, published MD5 checksums, local SHA256 hashes and CSV schemas. Raw downloads remain ignored by Git.

| Local archive | Exact contents used | Role |
| --- | --- | --- |
| `Data/raw/megascale/Processed_K50_dG_datasets.zip` | `Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv` | Primary source of sequences and machine-learning stability labels. Use rows with valid numeric `ddG_ML`; the file also contains rows without usable mutation-effect targets. |
| Same archive | `Single_DMS_list.csv`, `Double_DMS_list.csv` (under the archive's dataset directory) | Metadata for reference domains and single/double-mutant scans. Confirm mappings using sequences, not filenames alone. |
| Same archive | `Tsuboyama2023_Dataset1_20230416.csv` | Broader source table retained for audit; not the default training table. |
| `Data/raw/megascale/AlphaFold_model_PDBs.zip` | Released parent/domain structure models | Backbone geometry; predicted structures must be distinguished from experimental structures and checked against reference sequences. |
| `Data/raw/megascale/Pipeline_figure_model.zip` | `Pipeline_figure_model/Data_quality_filtering_script.ipynb`, `Additive_model_Fig4.ipynb` | Authors' filtering and additive-analysis reference methods. Downloading does not execute these notebooks. Our proposed scorers are separate implementations. |

**How the source labels were obtained:** cDNA-display proteolysis with trypsin and chymotrypsin; five-minute digestion at room temperature; sequencing counts and Bayesian inference of protease resistance and folding stability. The authors screen scan quality and assay-disrupting mutants. Their `_ML` columns mark filtered values with `-`; `dG_ML` can also contain censored bounds such as `<-1` or `>5`. See the [original experimental and filtering methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC10412457/). These are folding-stability estimates, not aggregation or binding measurements.

**Proposed database preparation (not yet run):**

1. Stream the primary CSV inside the ZIP with Python `zipfile` + `csv`; no full in-memory load is needed. Read `aa_seq` as the domain sequence; keep `aa_seq_full` only for assay-context provenance because it contains flanking sequences.
2. Keep `name`, `WT_name`, `WT_cluster`, `mut_type`, `aa_seq`, `dG_ML`, `ddG_ML`, `deltaG_95CI_low`, `deltaG_95CI_high` and `pair_name`. Require finite numeric `dG_ML` and `ddG_ML`; do not replace `-` or bound strings with numbers, or recover filtered labels from unfiltered `deltaG` columns.
3. Start with canonical, same-length single/double substitutions. Verify mutation identities and one-based positions against the reference sequence; exclude insertions/deletions for this fixed-backbone pilot. Consolidate synonymous/repeated measurements of the same parent/sequence and retain replicate provenance.
4. Match `WT_name` to a sequence-verified parent structure. Use that same backbone for its variants, validate complete backbone coverage, and record mapping failures. Supplied AlphaFold models do not automatically cover every usable label.
5. Freeze parent/homology groups before splitting; use `WT_cluster` as a grouping aid, not proof of homology separation. Assign groups to approximately 70/15/15 train/validation/test partitions with seed 7. Report actual group and example counts. No mutant-level random split.

**Concrete initial training configuration (proposed):** small residue and symmetric pair networks with one 32-unit ReLU hidden layer; domain score differences predict `ddG_ML`. Use Huber loss (`delta=1` kcal/mol), AdamW (`lr=0.001`, `weight_decay=0.0001`), batches of 64 variant/reference pairs, at most 100 epochs and early stopping after 10 epochs without improved validation MAE. Balance parent-protein contributions, run seeds 7/17/27 and tune only on validation groups. Pair terms use each nonadjacent Cα contact within 8 Å once. The held-out comparison tests interactions beyond learned position preferences; it does not fine-tune ProteinMPNN.

Reproduce or verify acquisition from the repository root with a local Python 3.9+ interpreter:

```powershell
python Code/prepare_stability_data.py
```

The command downloads three selected archives, checks their published MD5 and local SHA256, and records the archive members and CSV headers. It does not download raw sequencing counts, run PSC jobs, filter training examples or start training.

### Pair-geometry bias for the contextual interaction scorer (proposed)

This module belongs to the sequence-conditioned attention head described in the patch-scorer architecture, rather than the additive/regression baseline above. It lets each attention head learn how residue proximity should influence attention. It is not a physical interaction energy, and it has not yet been trained.

| Step | Calculation | Output |
| --- | --- | --- |
| 1. Measure geometry | For every pair in the patch-plus-context region, compute Cα distance `d_ij` in Å and sequence-index separation `abs(i-j)`. Use the same parent backbone for both assignments. | Two fixed measurements per pair. |
| 2. Encode distance | Eight radial features `r_m = exp(-(d_ij-c_m)^2 / (2*sigma^2))`, centers `c_m = 2,4,...,16 Å`, width `sigma = 2 Å`. | Eight distance channels. |
| 3. Add separation/contact | Append `min(abs(i-j)/32, 1)` and `1[d_ij <= 8 Å]`. | Ten geometry features per pair. |
| 4. Learn per-head bias | Shared network `Linear(10,16) -> ReLU -> Linear(16,4)`. | Four scalar biases: one for each attention head. |
| 5. Adjust attention | For head h, add `b_ij,h` to `Q_i,h · K_j,h / sqrt(16)`, then softmax over valid key residues j. | Geometry-adjusted attention weights. |

For B assignments and T patch/context residues, geometry features have shape **`[B,T,T,10]`**. The network produces biases **`[B,T,T,4]`**, rearranged to **`[B,4,T,T]`** to match attention logits. Four heads each have 16 channels in a 64-channel token representation.

**Example:** if two residues are 6 Å apart and 12 positions apart in sequence, the appended channels are `12/32 = 0.375` and contact `1`; the distance feature centered at 6 Å equals `1`. The network learns how these features affect each head's attention. Positive bias increases a pair's relative attention logit; negative bias decreases it.

**Initialization and training:** initialize the final linear layer's weights and bias to zero, so the module starts with no geometry adjustment. Train the bias network jointly with the small attention head using measured swap-ranking loss, keeping ProteinMPNN frozen. Backbone geometry and its encoded features are fixed; only the mapping from geometry to attention bias is learned. Symmetric pair inputs give `b_ij = b_ji`, although the complete Q/K attention need not be symmetric. Compute biases for all region pairs, including self-pairs; the 8 Å indicator is a feature, not a hard attention cutoff. Mask padded key tokens before softmax and exclude padded tokens from pooling.

**Required comparison:** train an otherwise identical head with the geometry bias fixed at zero. Compare on the same held-out clusters, with matched training settings and seeds, to test whether explicit geometry bias adds value. The settings above are proposed starting values, not validated optimum values.

An editable process diagram and its reproducible builder are in [Analysis/path_scorer/figures/](Analysis/path_scorer/figures/). This folder follows the requested `path_scorer` spelling and is separate from `Analysis/patch_scorer/`.

## Repository layout

| Directory | Contents |
|---|---|
| `Compute/psc/` | Run preparation, Slurm test, verified archive/transfer tools and tests |
| `Code/` | Development-input preparation, frozen baseline and patch-permutation tools |
| `Data/` | Dataset manifests and data-preparation notes; large inputs remain local |
| `Analysis/` | Actual pilot tables/figures, mapping checks and reproducible analysis scripts |
| `Documentation/` | Proposal, instructor feedback, project guidelines and LaTeX template |
| `Planning/` | Mid-report plan and literature notes |
| `Papers/` | Reference links; downloaded PDFs remain local |
| `Runs/`, `Results/` | Local/generated artifacts, excluded from Git |

## Start here

Clone the repository and run commands **from its root** (the local `Project/` folder is the repository):

```bash
git clone https://github.com/ashwathr1008/patch-aware-proteinmpnn.git
cd patch-aware-proteinmpnn
python -m unittest discover -s Compute/psc -p test_bundle.py
```

Archive tests and preparation utilities use the Python 3.9+ standard library. No model packages or GPU are needed for these tests.

For the completed local pipeline, follow [environment setup and execution](Code/README.md). Small result tables and figures are included in Git; the full local run is `Results/baseline-local-20261104`.

To prepare a new small engineering test, including pinned upstream source, two model checkpoints and the public 1UBQ structure:

```bash
python Compute/psc/prepare_smoke.py --output Runs/my-smoke-test
```

This downloads inputs and creates files; it does not submit a job. Use a new output directory for each preparation. Model execution requires NumPy and a PyTorch installation compatible with the actual hardware. The original HW1 environment is a reusable PSC option for Ashwath, but its CUDA build failed on V100; do not assume `torch.cuda.is_available()` proves compatibility. See [PSC setup and retrieval](Compute/psc/README.md) for submission, deadlines and local fallback.

The PSC account/path settings in those scripts currently describe Ashwath's setup. Teammates must use their own authorized identities and environment paths. Do not commit credentials or copy another person's authentication material.

## Work together

Use short branches such as `ashwath/baseline`, `kirti/patch-controls`, and `vani/related-work`, then open pull requests to `main`. Keep changes small and include the command/check that verifies them. See [CONTRIBUTING.md](CONTRIBUTING.md).

Commit code, configurations, structure/split manifests, report source, and small reviewed result tables/figures. Downloaded literature PDFs, model checkpoints, raw datasets, environments and complete run archives stay outside Git. Share large artifacts separately with a manifest of their source, version and hashes. GitHub is not the only backup of PSC data.

Mid-report runs should finish and be copied locally November 5. Optional later PSC work should finish November 7 with verified local copies by November 8 noon, ahead of the November 9 outage.

## References and third-party materials

See the [reference list](Papers/README.md), [proposal](Documentation/Patch%20aware%20protein%20design%20with%20ProteinMPNN.pdf), and [instructor feedback](Documentation/Feedback%20on%20Proposal.txt). The report template retains its original attribution and [CC BY-SA 4.0 license](Documentation/CMU_LTI_Paper_Template__2025_10_8_/LICENSE). ProteinMPNN is downloaded from its official repository with its upstream license; it is not vendored into Git here.
