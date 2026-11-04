# Patch-aware protein design with ProteinMPNN

Course project for 02-741 Generative AI for Biomedicine, Fall 2026.

**Team:** Ashwath Raghuram, Kirti Hargunani, and Vani Nigam.

We investigate whether explicitly scoring local arrangements of residues on protein surfaces adds useful information beyond amino-acid composition and frozen ProteinMPNN. The eventual motivation is improved developability while preserving structural and functional compatibility. Native-versus-permuted patches provide compatibility controls; they are not measured aggregation labels.

## Mid-report scope

Due November 6; target a complete draft November 5 evening. Deliver a working baseline, a small controlled patch experiment, quantitative preliminary results, and a plan for independent solubility/developability validation. See the [detailed plan](Planning/Midreport_Plan_2026-11-04.md).

| Owner | Mid-report responsibility |
|---|---|
| Ashwath | Working setup and first example; ProteinMPNN baseline outputs; reproducible handoff and baseline-methods text |
| Kirti | Expand and validate patches and matched permutations; comparison plot and patch illustration; experiment methods/results |
| Vani | Bounded motivation/related-work and validation-dataset investigation; evaluation and limitations text, depending on availability |

Each person writes the paragraphs for their own work. The technical sections should be ready for assembly even if Vani's available time is limited.

## Current status — November 4

- PSC access, staged ProteinMPNN source/weights, and checksum-verified retrieval have been exercised.
- The first PSC test, job `50214872`, failed on a V100 because the existing PyTorch CUDA build could not execute on that GPU. Its 17 files were recovered and verified locally. This is an infrastructure finding, not a biological result.
- The local RTX 4090 Laptop GPU completed the development baseline: **48 generated sequences**, three structures, two frozen checkpoints, eight samples per combination. All six runs passed validation. A repeat of 1UBQ reproduced all 16 generated sequences and scores exactly on the same machine.
- A checked eight-residue 1UBQ patch permutation preserves geometry, membership, composition and outside-patch context; its mapping agrees with the ProteinMPNN input at all 76 residues and 304 backbone atoms.
- [Ashwath's report draft](Documentation/Midreport/ashwath_section.md), [results and figures](Analysis/README.md), and [Kirti's handoff](Planning/Kirti_Handoff_2026-11-04.md) contain the completed work and next steps.
- Replacement PSC H100 test `50216628` was submitted and was still pending at the last check. A local collector is watching it; the successful baseline is already stored locally. See the compute notes for status and retrieval details.
- The expanded patch dataset, controlled paired scoring experiment and learned scorer remain future work. The three structures are development examples, not an independent benchmark.
- No training or experiment starts automatically by cloning this repository or running its tests.

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
