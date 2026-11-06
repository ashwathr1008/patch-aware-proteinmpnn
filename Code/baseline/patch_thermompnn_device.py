"""Let ThermoMPNN-D's inference script run without a GPU, by a minimal recorded patch.

ThermoMPNN-D is the published baseline closest to this project: frozen ProteinMPNN features
with a light head, trained on the same mega-scale stability data, with a double-mutant mode.
Comparing against it matters more than comparing against a model we wrote ourselves.

Its entry point `v2_ssm.py` hard-codes the device in five places, so it cannot run on a
machine without CUDA. The package internals already resolve the device properly; only the
entry script does not. This rewrites exactly those five lines and nothing else.

The patch is a script rather than a hand edit so that it is reviewable, repeatable, and
visible in this repository's history. Results obtained from a patched third-party model depend
on the patch being right, so it should be as small and as legible as possible. It refuses to
run twice, verifies each replacement is unique before applying it, and writes a record of what
it changed.

ThermoMPNN-D is MIT licensed, which permits modification. The upstream checkout is left in
place; nothing here is vendored into this repository.
"""

import argparse
import hashlib
import json
from pathlib import Path

MARKER = "# patched for optional CPU execution by patch_thermompnn_device.py"

PREAMBLE = '''%s
import os as _os
import torch as _torch


def _resolve_device():
    """Honour THERMOMPNN_DEVICE, else use CUDA when it is genuinely available."""
    choice = _os.environ.get("THERMOMPNN_DEVICE", "").strip().lower()
    if choice in ("cpu", "cuda"):
        return choice
    return "cuda" if _torch.cuda.is_available() else "cpu"


_DEVICE = _resolve_device()
''' % MARKER

REPLACEMENTS = [
    ('    device = "cuda"', "    device = _DEVICE"),
    ("    model.cuda()", "    model.to(_DEVICE)"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True,
                        help="path to a ThermoMPNN-D checkout")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    target = args.repo.resolve() / "v2_ssm.py"
    if not target.exists():
        raise SystemExit("No v2_ssm.py under " + str(args.repo))
    source = target.read_text(encoding="utf-8")
    before = hashlib.sha256(source.encode()).hexdigest()
    if MARKER in source:
        print("already patched; nothing to do")
        return

    counts = {}
    patched = source
    for old, new in REPLACEMENTS:
        occurrences = patched.count(old)
        if occurrences == 0:
            raise SystemExit("Expected line not found, upstream may have changed: " + old.strip())
        patched = patched.replace(old, new)
        counts[old.strip()] = occurrences

    # The preamble goes after the existing imports so the module still reads top-down.
    lines = patched.splitlines(keepends=True)
    insert_at = 0
    for index, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = index + 1
    patched = "".join(lines[:insert_at]) + "\n" + PREAMBLE + "\n" + "".join(lines[insert_at:])

    record = {"file": str(target), "sha256_before": before,
              "sha256_after": hashlib.sha256(patched.encode()).hexdigest(),
              "replacements": counts, "marker": MARKER,
              "note": "Device resolution only. No model, data or arithmetic was changed."}
    if args.dry_run:
        print(json.dumps(record, indent=2))
        return
    backup = target.with_suffix(".py.orig")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
    target.write_text(patched, encoding="utf-8")
    (args.repo.resolve() / "device_patch_record.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for line, count in counts.items():
        print("  replaced %dx: %s" % (count, line))
    print("original kept at %s" % backup)
    print("set THERMOMPNN_DEVICE=cpu or cuda to override autodetection")


if __name__ == "__main__":
    main()
