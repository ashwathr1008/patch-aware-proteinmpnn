"""Render a development patch pair with shared geometry, camera, and colors."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path


# Categorical identity colors: these do not encode any biochemical score.
AMINO_ACID_COLORS = dict(zip(
    "ACDEFGHIKLMNPQRSTVWY",
    ("#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
     "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
     "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
     "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"),
))


def validate_manifest(manifest):
    if manifest.get("schema_version") != 1 or manifest.get("status") != "development_demo_only":
        raise ValueError("Expected a schema-1 development_demo_only patch manifest.")
    if not manifest.get("checks") or not all(value is True for value in manifest["checks"].values()):
        raise ValueError("The manifest must report passing invariant checks.")
    residues = manifest.get("residues", [])
    patch = [row for row in residues if row["in_patch"]]
    if not patch or any(row["backbone_coordinates"]["CA"] is None for row in patch):
        raise ValueError("Patch must contain CA coordinates for every selected residue.")
    if Counter(row["original_aa"] for row in patch) != Counter(row["permuted_aa"] for row in patch):
        raise ValueError("Patch amino acid compositions differ.")
    if any(row["original_aa"] != row["permuted_aa"] for row in residues if not row["in_patch"]):
        raise ValueError("An identity outside the patch has changed.")
    for row in residues:
        for key in ("original_aa", "permuted_aa"):
            if row[key] not in AMINO_ACID_COLORS:
                raise ValueError("Unsupported amino acid identity: {}".format(row[key]))
        ca = row["backbone_coordinates"]["CA"]
        if ca is not None and (len(ca) != 3 or not all(math.isfinite(value) for value in ca)):
            raise ValueError("CA coordinates must be finite XYZ triples.")
    return residues, patch


def annotate_patch(axis, patch, identity_key, figure):
    """Place labels on two sides of the projected patch, with short leader lines."""
    from mpl_toolkits.mplot3d import proj3d

    projected = []
    for row in patch:
        xyz = row["backbone_coordinates"]["CA"]
        x, y, _ = proj3d.proj_transform(*xyz, axis.get_proj())
        pixel_x, pixel_y = axis.transData.transform((x, y))
        projected.append((pixel_x, pixel_y, x, y, row))
    ordered = sorted(projected, key=lambda item: (item[0], item[1]))
    split = max(1, len(ordered) // 2)
    for side, group in ((-1, ordered[:split]), (1, ordered[split:])):
        if not group:
            continue
        group = sorted(group, key=lambda item: item[1])
        target_x = (min(item[0] for item in projected) - 22 if side < 0
                    else max(item[0] for item in projected) + 22)
        middle_y = sum(item[1] for item in group) / len(group)
        spacing = 17 * figure.dpi / 72
        for position, (pixel_x, pixel_y, x, y, row) in enumerate(group):
            target_y = middle_y + (position - (len(group) - 1) / 2) * spacing
            label = "{}{}{}".format(row[identity_key], row["resseq"], row["insertion_code"])
            axis.annotate(
                label, xy=(x, y), xycoords="data",
                xytext=((target_x - pixel_x) * 72 / figure.dpi,
                        (target_y - pixel_y) * 72 / figure.dpi),
                textcoords="offset points", ha="right" if side < 0 else "left", va="center",
                fontsize=9, fontweight="bold", color="#24313d", annotation_clip=False,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1},
                arrowprops={"arrowstyle": "-", "color": "#697887", "lw": 0.7,
                            "shrinkA": 2, "shrinkB": 5},
            )


def render_manifest(manifest, output):
    """Write patch_pair.png and patch_pair.pdf into a new output directory."""
    output = Path(output)
    if output.exists():
        raise ValueError("Output already exists; choose a new directory: {}".format(output))
    residues, patch = validate_manifest(manifest)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ca_rows = [row for row in residues if row["backbone_coordinates"]["CA"] is not None]
    coordinates = [row["backbone_coordinates"]["CA"] for row in ca_rows]
    lower = [min(xyz[dimension] for xyz in coordinates) for dimension in range(3)]
    upper = [max(xyz[dimension] for xyz in coordinates) for dimension in range(3)]
    mid = [(a + b) / 2 for a, b in zip(lower, upper)]
    half_width = max(max(b - a for a, b in zip(lower, upper)) * 0.56, 1.0)
    figure = plt.figure(figsize=(13, 7.2), facecolor="white", dpi=150)
    figure.subplots_adjust(left=0.02, right=0.98, top=0.81, bottom=0.20, wspace=0.04)
    axes = [figure.add_subplot(1, 2, index + 1, projection="3d") for index in range(2)]
    source_name = Path(manifest["source"]["pdb_path"]).stem.upper()
    chain = manifest["source"]["chain"].strip() or "blank"
    changed = sum(row["original_aa"] != row["permuted_aa"] for row in patch)
    figure.suptitle("A fixed patch, two identity assignments", y=0.97, fontsize=19, fontweight="bold", color="#172b3a")
    figure.text(0.5, 0.91,
                "{} | chain {} | {} patch residues | {} identities reassigned | seed {}".format(
                    source_name, chain, len(patch), changed, manifest["parameters"]["seed"]),
                ha="center", fontsize=11, color="#52616d")
    try:
        for axis, identity_key, title in zip(axes, ("original_aa", "permuted_aa"),
                                             ("Original assignments", "Permuted assignments")):
            axis.set_title(title, fontsize=14, pad=8, color="#24313d")
            axis.set_proj_type("ortho")
            axis.view_init(elev=22, azim=-60)
            axis.set_box_aspect((1, 1, 1))
            axis.set_xlim(mid[0] - half_width, mid[0] + half_width)
            axis.set_ylim(mid[1] - half_width, mid[1] + half_width)
            axis.set_zlim(mid[2] - half_width, mid[2] + half_width)
            axis.set_axis_off()
            # Do not draw a fictitious connection across missing CA atoms or long gaps.
            for left, right in zip(residues, residues[1:]):
                a, b = left["backbone_coordinates"]["CA"], right["backbone_coordinates"]["CA"]
                if a is None or b is None or sum((x - y) ** 2 for x, y in zip(a, b)) > 4.5 ** 2:
                    continue
                axis.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#c4cbd1", lw=1.3, alpha=0.85)
            axis.scatter(*zip(*coordinates), c="#d4d9de", s=9, alpha=0.8, depthshade=False)
            patch_xyz = [row["backbone_coordinates"]["CA"] for row in patch]
            axis.scatter(*zip(*patch_xyz), c=[AMINO_ACID_COLORS[row[identity_key]] for row in patch],
                         s=115, edgecolors="#35434f", linewidths=0.7, depthshade=False)
        figure.canvas.draw()
        for axis, identity_key in zip(axes, ("original_aa", "permuted_aa")):
            annotate_patch(axis, patch, identity_key, figure)
        identities = sorted({row["original_aa"] for row in patch})
        legend = [Line2D([0], [0], marker="o", color="none", markerfacecolor=AMINO_ACID_COLORS[aa],
                         markeredgecolor="#35434f", markersize=8, label=aa) for aa in identities]
        figure.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.125),
                      ncol=min(10, len(legend)), frameon=False, title="Amino acid identity (same colors in both panels)",
                      fontsize=10, title_fontsize=10, handletextpad=0.35, columnspacing=1.3)
        figure.text(0.5, 0.055,
                    "Backbone geometry, patch membership, and amino acid composition are identical.\n"
                    "Only patch identity assignments change. Development demonstration; no performance claim.",
                    ha="center", va="center", fontsize=10, color="#52616d", linespacing=1.5)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
        figure.savefig(output / "patch_pair.png", dpi=240, facecolor="white")
        figure.savefig(output / "patch_pair.pdf", facecolor="white",
                       metadata={"Title": "Fixed-geometry patch permutation development demonstration"})
    finally:
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        render_manifest(manifest, args.output)
    except (ValueError, KeyError, OSError, ImportError) as exc:
        parser.exit(2, "error: {}\n".format(exc))
    print("Saved {} and {}".format(args.output / "patch_pair.png", args.output / "patch_pair.pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
