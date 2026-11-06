# Patch-aware ProteinMPNN: November 4–6 midpoint plan

Prepared November 4, 2026. These are proposed tasks and acceptance criteria, not completed experiments or predicted results. The proposal (PDF and DOCX), feedback, project guidelines, report template, and relevant sections of all eleven local papers were reviewed. Dates use America/New_York time.

**Execution update:** the 48-sequence frozen baseline, same-machine repeat, first patch example and controlled 85-pair expansion are complete. The expansion covers 17 patches/seven sources/seven homology clusters, with both checkpoints preferring the original in 84/85 order-averaged pairs and one order-sensitive exception. See [baseline draft](../Documentation/Midreport/ashwath_section.md), [controlled results](../Documentation/Midreport/controlled_scoring_results.md), and [current scope](Midreport_vs_Final_Scope.md). Ashwath took forward the expanded work; remaining owners will be decided at today's end. H100 smoke job 50216628 succeeded and all 22 files are verified locally. The original targets and provisional division below remain historical planning context.

Aim for a small, reproducible **patch-compatibility pilot**: a working frozen ProteinMPNN baseline, composition-matched patch controls, quantitative preliminary analysis, and a credible path to independent developability validation. The [guidelines](../Documentation/Project%20Guidelines.txt) request preliminary results and a remaining-work timeline; they do not require completed fine-tuning or all proposed property heads. The [feedback](../Documentation/Feedback%20on%20Proposal.txt) explicitly distinguishes compatibility from aggregation.

Keep aggregation/solubility as the motivation and eventual validation question. Native-versus-permuted patches supply compatibility labels, not measured aggregation labels. A null pilot result is reportable when the experiment is sound and it determines the next step.

## Research question and corrections

**Does a spatial patch representation provide useful information beyond amino-acid composition, and what does frozen ProteinMPNN already capture about that information?**

Separate pipeline validity, sequence–geometry compatibility, and prediction of independent measured developability. Success at the first two does not establish the third.

Revise these proposal claims:

- ProteinMPNN models a joint sequence distribution through autoregressive conditionals. Unknown future neighbours do not make the joint configuration unconstrained. Decoding-order sensitivity is an empirical hypothesis.
- Composition matching removes a residue-count shortcut. Discrimination can still reflect packing, exposure, sequence context or perturbation artefacts; it does not establish aggregation mechanisms.
- BindCraft already treats interface and non-interface redesign differently. Novelty must come from the patch objective, matched controls or additional predictive value, rather than region separation alone.
- A patch heatmap does not automatically establish causal residue-level credit assignment. ResiDPO is relevant prior work for local rewards.
- Keep SKEMPI binding mutation effects as a separately defined future task; do not mix their labels with native/permuted patch labels.

## What to have by November 5

These quantities are workload caps for an exploratory pilot, not statistically powered sample-size claims.

| Deliverable | Target | Acceptance criterion |
|---|---|---|
| Working baseline | Frozen vanilla ProteinMPNN inference and scoring | Saved identifiers, checkpoint/version, configuration, seeds, sequences, scores, runtime and failures; one run reproduced |
| Patch dataset | 10–20 clean, nonredundant proteins; approximately 100–300 matched pairs | Source/chain/residue mapping, family groups, development/holdout assignment, patch membership and pair validation |
| Control experiment | Composition-only control, simple spatial descriptor and frozen ProteinMPNN compatibility | Per-protein paired results, tie handling, score definitions and documented decoding-order treatment |
| Visual explanation | One structure with highlighted patches and a matched arrangement example | Clear account of what changes and what stays fixed, including the surface approximation |
| Evaluation readiness | One experimental-solubility dataset audited | Actual label definition, access, identifiers, structure coverage, exclusions and planned split |
| Report | Complete draft with references, limitations and revised schedule | PDF compiles; every preliminary-result statement corresponds to an actual output |

Use existing benign benchmark structures. Prefer a prefiltered nonredundant set to a large new PDB collection. Start with monomers; a simple complex can illustrate regional masks if already easy to process. Binder performance need not be demonstrated by the midpoint.

After the control experiment works, generate roughly 8–16 candidates on 3–5 backbones and demonstrate reranking if time permits. Clean pairs and quantitative baseline comparisons take priority over candidate volume.

If only one dedicated day per person is available, reduce to 5–10 structures, 50–100 pairs and 2–3 inference examples. Preserve the controls, analysis and complete report. Show individual-protein results and explicitly call this a feasibility pilot.

## How to make the pilot informative

**Patch definition.** Use simple residue neighbourhoods on a fixed backbone and document the surface-membership approximation. A residue graph is adequate for the pilot; a full surface mesh adds substantial setup. Visually inspect several patches. Do not call a backbone exposure approximation measured solvent-accessible area.

**Grouping and splitting.** Group proteins, complexes and close homologues before extracting patches or generating variants. All descendants of a source remain in its group. With no new model training, use a development subset and a frozen diagnostic subset; do not force tiny train/validation/test partitions. If training later, add proper training, validation and test groups. The new split does not establish that structures were absent from ProteinMPNN pretraining.

**Exact matched pairs.** Hold geometry, patch membership, size and outside-patch context fixed while rearranging amino-acid assignments. Verify counts exactly; exclude unchanged rearrangements or document their treatment. Randomize presentation order. Ensure filenames, IDs and padding cannot expose labels.

A deterministic composition-only model should give identical scores within every pair. Report ties and count them as half-correct for paired ranking accuracy. Geometry-only scores should also tie when geometry is unchanged. These are pipeline sanity checks, not biological discoveries.

Do not reuse native side-chain chemistry as an input after changing residue identities. Use fixed backbone geometry plus current residue assignments. Distinguish annotations used for bookkeeping from model features.

**Three comparisons.**

1. Composition-only summary: confirms that matching removes the count shortcut.
2. Simple spatial descriptor, such as local hydrophobic clustering: tests arrangement sensitivity. Native patches need not have a better value. Show changes/distributions without treating the heuristic's direction as biological ground truth. Do not call it SAP or an aggregation predictor unless that method is actually implemented and validated.
3. Frozen ProteinMPNN: compare otherwise matched full sequences on the same backbone. This is sequence–backbone compatibility, not a patch-specific aggregation endpoint.

For ProteinMPNN comparisons, use the same predefined decoding orders for both members of a pair and summarize scores across those orders. Match masks and context. Equal seeds alone do not prove equal orders; verify what the runner uses. Record order variability on a subset. If explicit order control is unavailable, document the limitation and make no mechanistic claim about decoding.

Report paired score differences and ranking results by protein. Overlapping patches are not independent biological replicates. With few proteins, show observations and their spread rather than narrow intervals based on hundreds of patches. Do not tune a descriptor or its preferred direction on the diagnostic subset.

**Optional reranking.** Compare random selection, ProteinMPNN-score selection and patch-score selection from the same candidate pool, keeping the number selected equal. Track selected identities, diversity, composition/charge shifts and scores not used by each selection rule. Improvement in the ranking score is expected by construction and is not independent validation. A second computational predictor is still a proxy. If using a complex illustration, preserving interface residues does not establish retained affinity.

## Proposed division of labor

The proposal names Ashwath Raghuram, Kirti Hargunani and Vani Nigam but does not map individuals to its A/B/C workstreams. Ashwath expects to start first and prefers work he can substantially delegate to the coding assistant. He should therefore own the initial setup and handoff package, then baseline integration. Kirti and Vani's assignments remain provisional and can be exchanged according to experience. Each person owns a technical output and writes the corresponding report paragraphs.

| Person | November 4 | November 5 | Writing ownership |
|---|---|---|---|
| Ashwath | Delegate environment, runner, output schema and one small patch example to the assistant; review and hand off a working example before others begin | Run agreed baseline set; integrate outputs; optional small generation/reranking demo | Baseline method, compute/reproducibility, generation results |
| Kirti | Starting this evening: build on the starter, curate structures/groups, and extend patches and matched permutations | Validate pairs; produce the patch illustration and per-protein comparison with Ashwath's baseline outputs | Data, controls, experiment methods/results and limitations |
| Vani | Availability uncertain: take a bounded, independent motivation/related-work and label-dataset audit task | Supply scientific framing and evaluation/limitations text as time allows | Motivation, related work and independent-validation plan |

Allow approximately 6–8 focused hours for each main technical workstream, with optional second-day time for extensions/debugging. These are effort estimates, not measured model runtimes. Vani's availability is unknown, so essential experiments and report assembly must be covered by Ashwath and Kirti. Reduce data volume before sacrificing the writing buffer. Each person provides finished paragraphs, captions and references for their contribution.

Within the first hour, settle a shared example format: protein ID, chain/residue mapping, group/split, patch ID, original/current identities, geometry reference, pair ID, candidate ID and score metadata. Ashwath's first handoff should include one valid real example; aim for the first roughly two hours, but treat setup failures as a reason to simplify. Kirti then owns curation and validation of the expanded set. Others can initially use a clearly marked synthetic fixture for code development; fixtures are not biological results.

Ashwath's assistant-delegable package should contain an isolated environment specification, a reproducible frozen-model runner, one end-to-end example with actual logs, a minimal patch/permutation implementation with count checks, agreed output schemas, and a short handoff README. It need not include a trained model, full dataset or finished study. The assistant can also prepare plots and draft technical descriptions from actual outputs. Ashwath's role is to review the scope and outputs, coordinate the handoff, and ensure the team understands the experiment and its limits. This plan does not imply that those scripts or model runs have already been completed.

## Two-day schedule and cutoffs

**November 4, first 60–90 minutes:** freeze question, roles and definitions; start the report immediately. Ashwath delegates the starter package and reviews the first inference attempt. If teammates start later, prepare their inputs now; once available they can proceed independently. Keep alternate-cluster investigation brief.

**Next 2–4 hours:** exchange a validated patch example and baseline output; exercise composition/geometry controls; confirm shared mappings. Expand curation only after this handoff works.

**End of November 4:** aim for an actual inference output, checked patch pairs, draft result table, report skeleton and an endpoint-audit decision. If missing, shrink scale and cancel optional models. Avoid relying on a long overnight job.

**November 5, first work block:** complete the fixed pilot, plots, failure inspection and result paragraphs. Add soluble ProteinMPNN if straightforward; its weights are already available in the official repository.

**November 5, mid-afternoon:** freeze experiments and figures. Each person delivers their section/captions. Use the remaining hours for interpretation, references, reproducibility checks, compilation and cross-review.

**November 5, evening:** have a submission-ready PDF. November 6 is a buffer for final checks and submission; the supplied documents do not specify a deadline time of day.

Stop rules:

- After roughly 90 minutes of CUDA trouble, attempt the smallest CPU run while diagnosing the GPU separately.
- If surface preprocessing is difficult, simplify to a documented backbone/residue representation.
- If a score is weak, report and investigate it rather than searching indefinitely for a positive result.
- If independent labels cannot be mapped, state that aggregation efficacy is untested and specify the next mapping task.
- A coherent baseline/control pilot is a valid submission target. Never replace missing results with expected numbers.

## Extensions to prioritize or defer

Optional, in order: soluble ProteinMPNN comparison; small reranking demonstration; controlled-order experiment; then one small learned compatibility model and a geometry ablation, only after expanding independent proteins and validating the split. Tiny-set training is debugging, not generalization evidence.

Structural prediction is optional only if already easy to run. Defer multi-head learning, SKEMPI training, MaSIF reproduction, backbone generation, broad binder benchmarks, DPO/RL and large folding campaigns. A new scorer is not necessary for the midpoint. Fine-tuning should depend on a useful, validated reward.

## Compute

Read-only inspection found an **RTX 4090 Laptop GPU with 16,376 MiB VRAM**, about **63.7 GiB RAM**, 32 logical CPUs and about 573 GiB free on the workspace drive. PyTorch was absent from the default Python and existing workspace `.venv` checked. Hardware detection is not a successful CUDA/model test.

Use a hybrid workflow: keep the midpoint runnable locally, and use PSC when an available slot saves time. The user confirmed permission beyond the midpoint. V100 job 50214872 failed because the installed CUDA build could not execute on that GPU; all 17 files were retrieved. Subsequent local inference and H100 smoke job 50216628 succeeded; all 22 H100 files are verified locally. Use a verified compatible GPU/environment combination. See [PSC run and retrieval instructions](../Compute/psc/README.md). Finish/download mid-report results by November 5 evening; optional further PSC preparation can finish November 7 with verified downloads by November 8 noon.

Frozen ProteinMPNN and small patch analyses are plausible local workloads; measure runtime and memory before scaling. Keep the local project environment separate from homework. The proposal's 200–400 GPU-hours concerns a much broader training/folding campaign.

The [official ProteinMPNN repository](https://github.com/dauparas/ProteinMPNN) provides ordinary and soluble checkpoints. Its [runner](https://github.com/dauparas/ProteinMPNN/blob/main/protein_mpnn_run.py) uses CUDA when available and otherwise CPU, making CPU a useful small-run fallback.

[PSC's migration notice](https://www.psc.edu/b2-migration-2026/), checked directly again on November 4, lists a **November 9, 2026** start and approximately **4–6 weeks** of outage, with a tentative December return. Compute, storage, data movement and VMs are included. Four weeks from November 9 is December 7, after the proposal's December 6 final deadline. Allocation permission extends past the midpoint, but post-November-8 work must be independent of Bridges-2. Retain verified local copies of each completed batch.

Lane cluster hardware, permissions and queue availability were not verified. Use it only if someone can quickly demonstrate a usable job. For larger final-stage work, investigate faculty/course-sponsored resources after the midpoint. [CMU ORCHARD](https://www.cmu.edu/rcd/faq.html) offers H100 resources through a governed access process; that does not establish immediate team access. No need for paid cloud compute has been demonstrated for this pilot.

## Independent validation and remaining work

Audit [eSOL](https://dbarchive.biosciencedbc.jp/data/esol/LATEST/README_e.html) as one candidate: it contains experimental cell-free protein-solubility measurements and structure identifiers where available. Check sequence/structure mapping and usable coverage before committing. These are whole-protein labels under particular conditions, not patch labels or therapeutic-formulation measurements. Transfer to miniprotein binders is unestablished. Define protein-level aggregation of patch scores without tuning on test labels.

A useful midpoint artifact records records examined, available measurements, successful structure matches, exclusions and a planned split. If usable labels are available early, a descriptive comparison is a valuable extension. Otherwise, complete mapping next. A second predicted score does not replace experimental validation.

| Dates | Revised milestone |
|---|---|
| November 6–10 | Finalize scope; secure independent labels, mapping and split; obtain extra compute only if needed |
| November 11–17 | One scorer if justified; frozen/simple baselines; geometry/composition controls |
| November 18–24 | Held-out reranking and measured-endpoint evaluation; fine-tuning only if reward validation succeeds |
| November 25–December 1 | Fixed evaluations, ablations, failure analysis and limited structural checks |
| December 2–5 | Final report/poster and cross-review; proposal lists December 6 deadlines |

If independent developability validation cannot be assembled or fails, finish with a narrower compatibility study. Do not substitute structural confidence for aggregation evidence.

## Report contents

Use the supplied LaTeX template, adapting the conclusion area to remaining work and adding explicit limitations. The supplied guidelines specify contents but no page limit; check any separate course instructions.

Include motivation and the narrowed question; closest related work; patches, controls, grouping and scoring; actual preliminary results; analysis of confounds/failures; remaining timeline; limitations. Explicitly mention small sample size, proxy labels, surface approximation, possible pretraining overlap and any unmeasured aggregation, binding or folding outcomes.

Minimum visual set: **one dataset table, one patch illustration and one quantitative paired-comparison plot**. Add a reranking table only if completed. Exact composition invariance alone is insufficient as the central result: pair it with actual ProteinMPNN outputs and a substantive compatibility/spatial comparison.

## Paper digest

Page numbers below refer to PDF pages. Relevant claims, methods/evaluation and selected supplementary material were reviewed; no experiments were reproduced.

| Local paper | Implication for this project |
|---|---|
| [ProteinMPNN](https://doi.org/10.1126/science.add2187), pp. 1–3, 7 | Direct baseline and decoding reference. Joint autoregression already captures dependencies. Recovery, predicted folding and laboratory evidence are distinct endpoints. Read closely. |
| [BindCraft](https://doi.org/10.1038/s41586-025-09429-6), pp. 2–3, 9, 11–12 | Already preserves an interface during soluble ProteinMPNN redesign. Separates computational filtering from measured binding and discusses false positives. Read relevant passages closely. |
| [RFdiffusion](https://doi.org/10.1038/s41586-023-06415-8), pp. 2, 9, 11 | Separates backbone generation, sequence design, screening and experiments. Reuse backbones; new generation is unnecessary now. |
| [AlphaFold2](https://doi.org/10.1038/s41586-021-03819-2), pp. 3, 6, 9 | Confidence and structural agreement concern structural accuracy, not aggregation validation. |
| [AlphaFold3](https://doi.org/10.1038/s41586-024-07487-w), pp. 6–7 | Complex predictions remain imperfect and static structures do not describe solution behaviour. Defer implementation. |
| [Chai-2](https://doi.org/10.1101/2025.07.05.663018), pp. 6–7, 11, 15 | Distinguishes experimental binding from computational developability screening and calls for biophysical assays. Patch descriptors are useful background, not validation of permutation labels. |
| [RFdiffusion3](https://doi.org/10.1101/2025.09.18.676967), pp. 5–6, 15 | Retains sequence redesign with ProteinMPNN/LigandMPNN. Computational benchmarks and selected experimental applications provide different evidence. |
| [EnzyGen](https://arxiv.org/abs/2405.08205), pp. 6–10 | Geometry-aware modeling and ablations are relevant principles; its proxies do not supply aggregation labels. Low reading priority. |
| [EnzyGen2](https://doi.org/10.64898/2026.03.02.709205), pp. 5–13 | Separates computational results, ablations and experimentally measured function. Its co-design task is beyond this scope. |
| [ESM3](https://doi.org/10.1126/science.ads0018), pp. 4–7 | Computational alignment results and measured fluorescence are separate evidence streams. It does not validate patch compatibility as an aggregation reward. |
| [ESM Cambrian/ESMFold2](https://doi.org/10.64898/2026.06.03.729735), pp. 2, 8–9 | Large generation/ranking campaigns followed by physical characterization illustrate separate validation. Inappropriate scale for this midpoint. |

Three close references named in the proposal are absent from the local paper folder:

- [MaSIF paper](https://www.nature.com/articles/s41592-019-0666-6) and [implementation](https://github.com/LPDI-EPFL/masif): surface patch learning is established; task-specific features and training objectives matter. Arbitrary cross-protein negatives are not an exact reproduction. Full preprocessing is too much setup for the core pilot.
- [ProtAlign/MoMPNN paper](https://arxiv.org/html/2603.06748v1) and [checkpoints](https://github.com/Qivon7/MoMPNN): relevant whole-sequence property alignment; uses computational property proxies. Checkpoint availability does not make training reproduction necessary now.
- [ResiDPO/EnhancedMPNN](https://arxiv.org/abs/2506.00297): relevant residue-level preference signals and computational designability evaluation. Avoid claiming all prior approaches lack local rewards.

Prioritize the feedback, ProteinMPNN, relevant BindCraft passages and these closest comparisons. Do not assign a teammate a full day of reading all eleven papers while implementation waits.
