## Stage 1: Prepare experimental labels

Filter MegaScale `_ML` measurements, verify mutation annotations, consolidate replicates, and normalize ΔΔG to kcal/mol with positive meaning destabilizing.  
Added: `Code/scorer/prepare_training.py`, `Code/scorer/requirements-preparation.txt`; produced: `Data/processed/vani_training_v2/measurements.sqlite` and `stage1_labels.json` (510,756 accepted rows; 497,198 consolidated measurements).

## Stage 2: Map structures and split proteins

Verify complete reference backbones and group parents before splitting, using ≥40% global edit similarity plus identical measured sequences; remote homology below that threshold is not ruled out.  
Added: mapping/clustering in `Code/scorer/prepare_training.py`; produced: `Data/processed/vani_training_v2/structures/` and `stage2_splits.json` (337 parents, 142 groups; train/validation/test: 236/50/51 parents).

## Stage 3: Build the training manifest

Build experimentally labelled reference–variant and measured reverse-swap pairs, preserve measurement provenance, and validate the manifest without training.  
Added: pair generation in `Code/scorer/prepare_training.py`, `Code/scorer/test_prepare_training.py`, and measured-swap support in `Code/scorer/data.py`; produced: `Data/processed/vani_training_v2/stability_pairs.json`, `stage3_manifest.json`, and `summary.json` (465,825 pairs, including 51,898 composition-preserving swaps).
