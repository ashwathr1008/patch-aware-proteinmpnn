# Team workflow

Start each work session from an updated `main`, then create a branch:

```bash
git switch main
git pull --ff-only
git switch -c kirti/patch-controls
```

Choose your own branch name. Stage the intended files, inspect the changes, and push:

```bash
git status --short
git add Code/ Data/ Analysis/
git diff --cached
git commit -m "Describe the concrete change"
git push -u origin HEAD
```

Open a pull request into `main` and have another teammate review it when possible. Write what changed, how it was checked, and any remaining limitation. Avoid force-pushing shared branches and coordinate before editing the same report section.

For each experiment, record input identifiers, protein/homology grouping, code/model versions, configuration, seeds, outputs, failures and runtime. Keep related patches/variants in the same split. Label exploratory results and distinguish predicted compatibility from measured biological outcomes.

Large inputs, downloaded model weights, raw output folders and private local configuration are ignored. Put small reviewed figures/tables in `Analysis/` and keep their generating code and provenance alongside them. Do not add whole `Runs/` or `Results/` trees to a commit.

The current infrastructure check is:

```bash
python -m unittest discover -s Compute/psc -p test_bundle.py
```

The archive source-symlink test may skip on Windows without symlink privileges. Model execution is a separate check and requires a compatible PyTorch environment.
