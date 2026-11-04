# Kirti's mid-report handoff — November 4

Ashwath's setup, frozen-model pilot and first patch example are complete. Your main task is the **controlled patch comparison**, using the existing tools and mappings. There is no trained patch model yet. Aim for a small defensible experiment and your methods/results paragraphs by November 5; reduce the number of proteins before dropping the controls.

## What is ready

- [Setup and commands](../Code/README.md): local GPU execution succeeded; CPU mode is available in the wrapper. The two checkpoints total about 13 MB; PyTorch itself is a much larger installation.
- [Development input manifest](../Data/development_manifest.json): three small proteins with hashes and limitations. These are debugging examples, not held-out evaluation data.
- [Baseline results](../Analysis/README.md): 48 generated sequences, actual score/recovery tables, a figure and portable provenance. Full raw runs are on Ashwath's laptop under `Results/`; the small shared artifacts suffice to inspect the baseline.
- [Patch schema/tool](../Code/patches/README.md), [actual example](../Analysis/handoff_20261104/patch_example/manifest.json), and [residue table](../Analysis/handoff_20261104/patch_example/residues.csv): center A:11, positions A:6–13, seed 7, eight patch residues, six changed identities. The same directory contains both full-chain FASTAs. All seven invariants pass.
- [Model-mapping verification](../Analysis/handoff_20261104/handoff_checks.json): all 76 residues/304 backbone atoms agree. Keep PDB residue identifiers distinct from one-based sequence indices; 1VII numbering starts at 41.
- [Ashwath's report text and captions](../Documentation/Midreport/ashwath_section.md) can be integrated without rewriting the experiment history.

## Work order for this evening and tomorrow

1. Reproduce the one-patch example and inspect the figure. Confirm counts, changed identities, mappings and exclusions before expanding. Use new output directories. The starter currently chooses **one center per structure**; changing its seed only changes the permutation at that center. Multiple centers require an explicit extension, not repeated seeds mislabeled as new patches.
2. Freeze a modest additional set of clean, benign proteins and record provenance/exclusions. Target roughly 5–10 independent proteins and 50–100 validated pairs if time permits. Group source proteins and close homologues before deriving variants. Current convenience-set group names do not implement homology clustering. Keep exploratory and fixed diagnostic choices distinct; do not call them a trained-model test split.
3. Compute exact composition-only controls and an explicitly defined simple spatial descriptor. Composition-only scores must tie within every pair. A hydrophobic-neighbor descriptor may respond to arrangement, but neither direction is an aggregation label. Use fixed backbone geometry and current identity assignments, without native side-chain chemistry leaking into permuted features.
4. Add frozen-model **paired rescoring** of each original/permuted full chain on the same backbone. Check native/mutant indexing against the model parser first. Reuse the same explicit decoding order within each pair, optionally averaging over a small predeclared set of shared orders. Record order IDs/seeds and score definition. Merely setting one random seed for successive score calls does not guarantee matched orders; verify the actual orders. Distinguish whole-sequence compatibility from an explicitly defined patch contribution. The generation-run CSV cannot answer this experiment.
5. Summarize by source protein: pair counts, invariant failures/exclusions, composition ties, spatial-score changes and paired model-score differences. If using paired ranking accuracy, count ties as half-correct and state score direction. Do not treat overlapping patches from one protein as independent evidence. Report null or mixed results without tuning the setup until native always wins.
6. Write methods, actual results, one comparison figure and limitations. Link each numerical claim to saved CSVs and configurations. Preserve the small-sample, surface-proxy and proxy-label caveats. Push a branch such as `kirti/patch-controls` and open a PR; keep raw datasets, weights and full run archives outside Git.

## Definition of done

A versioned input/pair manifest; validated mappings and invariants; exact composition ties; actual spatial and paired ProteinMPNN results on more than one independent protein; a per-protein comparison figure; and report-ready methods/results. If time is tight, reduce pair count and document the limitation. A learned scorer, fine-tuning and folding campaigns are outside the critical path.

No PSC job is required for the existing baseline: six model subprocesses totaled 39.4 seconds locally. Ashwath's separate H100 smoke test checks a compute option and is not your controlled experiment. Coordinate additional submissions with Ashwath and retrieve outputs before the outage.
