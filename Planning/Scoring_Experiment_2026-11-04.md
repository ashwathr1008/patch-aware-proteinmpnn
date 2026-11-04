# Controlled scoring experiment — frozen before score inspection

Prepared November 4, 2026. This specification extends the completed setup pilot. No scoring outcomes were available when this file was first written. Work ownership will be reassigned with the team when Ashwath finishes today's work.

## Question

Does frozen ProteinMPNN assign different sequence–backbone compatibility to original and shuffled local arrangements when amino-acid composition, backbone coordinates and outside-patch identities are held fixed? How consistent is the paired difference across proteins and decoding orders? This is a compatibility diagnostic, with no aggregation or folding labels.

## First checked pair

Use the saved 1UBQ chain A example, center 11, positions 6–13, permutation seed 7. Score both full sequences with vanilla and soluble `v_48_020`, no training, evaluation mode and zero added backbone noise. Use eight explicit shared decoding orders generated with seed 137. Record actual permutations and pass them directly to the model; a common RNG seed alone is insufficient.

Primary endpoint: mean per-residue negative log probability over the whole observed chain, averaged over eight orders. Define the paired gap as **permuted NLL minus original NLL**; positive means the model prefers the original. Secondary endpoint: the corresponding average of conditional terms at patch positions, labeled as context/order-dependent terms, not a standalone patch likelihood. No absolute score comparison between checkpoints is interpreted as biological superiority.

Audit identical-sequence comparisons (zero difference), reversal of presentation order, unchanged inputs/masks and explicit-order use. Use a numerical tie tolerance of 1e-6 NLL units. Preserve every order's result, including sign changes.

## Expansion selected before outcomes

Retain 1UBQ, 1VII and 1CRN as **development** sources. Attempt four additional diagnostic sources: 1L2Y (Trp-cage), 1ENH (engrailed homeodomain), 1SHG (spectrin SH3), and 2CI2 (barley chymotrypsin inhibitor 2). Selection is for small, familiar structural examples, not measured solubility or expected score direction. Keep the first deposited model and one explicitly recorded chain. Exclude incomplete/gapped/noncanonical model inputs rather than silently repairing them; record exclusions. These remain a convenience sample, not established as absent from model pretraining.

Define source/family groups and inspect sequence similarity before deriving pairs. All descendants of one source stay together. A basic within-pilot alignment screen is not comprehensive homology detection or proof of population independence. Keep diagnostic settings fixed after viewing scores.

Select up to three distinct low-neighbor-count patches per structure: 10 Å CA neighbor count, bottom quartile including ties, 8 Å patch radius, minimum eight residues/two identities, membership overlap limited to Jaccard <=0.5. Visit eligible centers by increasing neighbor count and sequence index, retaining those satisfying the overlap cap. Generate up to five distinct nonidentity permutations per patch using seed 7. Record shortfalls under these settings; do not relax thresholds in response to results. This caps the expansion at 105 pairs across seven source proteins.

## Baselines and interpretation

- Exact 20-amino-acid composition vectors must match within every pair. Any deterministic function of those vectors ties by construction.
- Geometry-only summaries also tie because coordinates and membership are fixed.
- Exploratory spatial descriptor: among within-patch CA contacts <=8 Å with nonadjacent sequence positions, the fraction joining two members of the fixed set A,V,I,L,M,F,W,Y. Normalize by all eligible contacts; record zero-contact cases separately. Report signed changes without declaring either direction biologically better. This is a simple arrangement descriptor, not SAP or an aggregation predictor.
- Frozen ProteinMPNN provides the main compatibility comparison. Average orders within each pair, then summarize pairs within each source protein. Show every protein and diagnostic/development labels. Count ranking ties as half-correct. Do not treat overlapping patches, shuffles or orders as independent biological replicates.

Retain original/permuted sequences, residue/patch mappings, input/model/code hashes, all scores, controls, failure/exclusion accounting and figures. Do not select a favorable subset or tune a descriptor direction on scored diagnostic examples.

## Mid-report versus final project

The mid-report should establish a reproducible baseline, a valid controlled comparison across multiple proteins, preliminary observations and limitations, and an audited plan for independent measured validation. It need not claim a new model or improved protein properties.

The final project should test whether an explicit patch score contributes useful information beyond frozen ProteinMPNN and composition. A practical core is a defined patch score, fair candidate reranking on independent protein groups, and evaluation using a measured endpoint where defensible mappings exist. Training a small patch scorer is justified only with adequate data and controls. Fine-tuning ProteinMPNN, residue-level preference optimization, broad binder design and multiple property heads are optional extensions, contingent on a validated objective and available time/compute. If measured developability validation cannot be completed, final claims must remain a compatibility study.
