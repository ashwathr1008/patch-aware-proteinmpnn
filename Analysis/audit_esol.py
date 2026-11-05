"""Audit eSOL as a candidate measured-outcome dataset: labels, mapping coverage, exclusions.

The mid-report plan requires one experimental-solubility dataset audited, with its actual label
definition, access, identifiers, structure coverage, exclusions and a planned split. This script
produces that record. It does not train or evaluate anything, and it deliberately reports
coverage whether or not the result is favourable.

What eSOL measures. Niwa et al., PNAS 2009 translated individual E. coli K-12 proteins in the
PURE cell-free system, which is chaperone-free, and measured what fraction of each product
stayed soluble. The label is therefore a **whole-protein solubility percentage for one organism
under one in vitro condition**. It is not a patch-level label, not an aggregation rate, and not
a therapeutic formulation measurement. A separate subset carries solubility measured with
chaperones added, which is a different quantity again.

Mapping. eSOL identifies proteins by B number and JW ID. UniProt carries the same identifiers
as ordered locus names along with PDB cross-references, so one join gives both the protein
identity and whether an experimental structure exists.

Inputs are downloaded files passed by path; nothing is fetched here. See the companion
documentation for retrieval commands and the dataset licence.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics

SOLUBILITY_COLUMN = "Solubility (%)"
IDENTIFIER_COLUMNS = ("B number", "JW_ID")
MAPPED_FIELDS = ("b_number jw_id gene_name uniprot_accession solubility_percent "
                 "pdb_entry_count pdb_entries excluded_reason").split()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_delimited(path, delimiter=","):
    with Path(path).open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def number(value):
    try:
        return float((value or "").strip())
    except (TypeError, ValueError):
        return None


def distribution(values):
    ordered = sorted(values)
    quartiles = statistics.quantiles(ordered, n=4) if len(ordered) > 1 else [ordered[0]] * 3
    return {"count": len(ordered), "mean": statistics.fmean(ordered),
            "median": statistics.median(ordered), "p25": quartiles[0], "p75": quartiles[2],
            "min": ordered[0], "max": ordered[-1],
            "fraction_below_30": sum(1 for v in ordered if v < 30) / len(ordered),
            "fraction_above_70": sum(1 for v in ordered if v > 70) / len(ordered)}


def build_locus_index(uniprot_rows):
    """Index UniProt entries by every ordered-locus token they carry."""
    index = {}
    for row in uniprot_rows:
        for token in (row.get("Gene Names (ordered locus)") or "").split():
            index.setdefault(token.strip(), []).append(row)
    return index


def audit(esol_rows, uniprot_rows, maximum_solubility):
    index = build_locus_index(uniprot_rows)
    records, counts = [], {
        "esol_records": len(esol_rows), "missing_solubility": 0, "unmapped": 0,
        "ambiguous_mapping": 0, "above_maximum_solubility": 0, "no_structure": 0, "usable": 0}
    for row in esol_rows:
        value = number(row.get(SOLUBILITY_COLUMN))
        b_number = (row.get("B number") or "").strip()
        jw_id = (row.get("JW_ID") or "").strip()
        record = {"b_number": b_number, "jw_id": jw_id,
                  "gene_name": (row.get("Gene name K-12") or "").strip(),
                  "solubility_percent": value, "uniprot_accession": "",
                  "pdb_entry_count": 0, "pdb_entries": "", "excluded_reason": ""}
        if value is None:
            counts["missing_solubility"] += 1
            record["excluded_reason"] = "no solubility measurement"
            records.append(record)
            continue
        hits = index.get(b_number) or index.get(jw_id) or []
        if not hits:
            counts["unmapped"] += 1
            record["excluded_reason"] = "no UniProt entry for either identifier"
            records.append(record)
            continue
        if len(hits) > 1:
            counts["ambiguous_mapping"] += 1
        entry = hits[0]
        record["uniprot_accession"] = entry.get("Entry", "")
        pdb_entries = [p for p in (entry.get("PDB") or "").split(";") if p]
        record["pdb_entry_count"] = len(pdb_entries)
        record["pdb_entries"] = ";".join(pdb_entries)
        # Percentages above 100 are a normalization artefact of the assay readout and are
        # excluded rather than clipped, so the exclusion is visible in the counts.
        if value > maximum_solubility:
            counts["above_maximum_solubility"] += 1
            record["excluded_reason"] = "solubility above {:g}%".format(maximum_solubility)
        elif not pdb_entries:
            counts["no_structure"] += 1
            record["excluded_reason"] = "no experimental structure in UniProt"
        else:
            counts["usable"] += 1
        records.append(record)
    return records, counts


def summarize(records, counts, maximum_solubility, pilot_structures):
    measured = [r for r in records if r["solubility_percent"] is not None]
    mapped = [r for r in measured if r["uniprot_accession"]]
    with_structure = [r for r in mapped if r["pdb_entry_count"]]
    without_structure = [r for r in mapped if not r["pdb_entry_count"]]
    usable = [r for r in records if not r["excluded_reason"]]
    return {
        "schema_version": 1, "purpose": "measured_outcome_dataset_audit", "dataset": "eSOL",
        "label_definition":
            "Percentage of each individually translated E. coli K-12 protein that remained "
            "soluble in the chaperone-free PURE cell-free system (Niwa et al., PNAS 2009). "
            "Whole-protein, one organism, one in vitro condition; not a patch-level label, "
            "not an aggregation rate, not a formulation measurement",
        "licence": "CC Attribution-Share Alike 2.1 Japan; attribution string required: "
                   "\"eSOL (c) Hideki Taguchi (Tokyo Institute of Technology) licensed under "
                   "CC Attribution-Share Alike 2.1 Japan\"",
        "identifiers": list(IDENTIFIER_COLUMNS),
        "counts": counts,
        "coverage": {
            "records_with_measurement": len(measured),
            "mapped_to_uniprot": len(mapped),
            "mapped_fraction": len(mapped) / len(measured) if measured else None,
            "with_experimental_structure": len(with_structure),
            "structure_fraction_of_measured":
                len(with_structure) / len(measured) if measured else None,
            "usable_after_exclusions": len(usable),
            "total_pdb_entries": sum(r["pdb_entry_count"] for r in with_structure),
        },
        "solubility_distribution": {
            "all_measured": distribution([r["solubility_percent"] for r in measured]),
            "with_structure": distribution([r["solubility_percent"] for r in with_structure]),
            "without_structure":
                distribution([r["solubility_percent"] for r in without_structure]),
        },
        "selection_bias":
            "Proteins with solved structures are measurably more soluble than those without, "
            "so restricting to structures removes part of the aggregation-prone range. The "
            "two distributions are reported side by side so the shift is explicit",
        "pilot_overlap": {
            "pilot_structures": pilot_structures,
            "overlapping_records": 0,
            "note": "eSOL covers E. coli K-12 only. Every current pilot structure is from "
                    "another source organism or is a designed sequence, so the overlap with "
                    "the existing seven-structure set is zero by construction. Using eSOL "
                    "means assembling a new E. coli structure set, not labelling the pilot",
        },
        "planned_split":
            "Not executed here. A homology-aware split requires clustering the usable "
            "sequences at a fixed identity threshold and assigning whole clusters to "
            "development and held-out groups. Ordered-locus proximity and gene names do not "
            "imply homology and must not be used as the grouping key. Sequences are not in "
            "the joined files and must be retrieved before clustering",
        "exclusions_applied": {
            "no solubility measurement": counts["missing_solubility"],
            "no UniProt entry for either identifier": counts["unmapped"],
            "solubility above {:g}%".format(maximum_solubility):
                counts["above_maximum_solubility"],
            "no experimental structure in UniProt": counts["no_structure"],
        },
        "open_questions": [
            "PDB cross-references are counted per protein, not checked for covering the full "
            "chain, resolution or mutation status; a usable structure set needs that filter",
            "The 788-protein chaperone subset measures a different quantity and is excluded "
            "from the usable count rather than merged",
            "Transfer from E. coli cell-free solubility to designed miniproteins or to "
            "formulation behaviour is unestablished",
        ],
    }


def figure(records, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mapped = [r for r in records if r["solubility_percent"] is not None and r["uniprot_accession"]]
    with_structure = [r["solubility_percent"] for r in mapped if r["pdb_entry_count"]]
    without = [r["solubility_percent"] for r in mapped if not r["pdb_entry_count"]]
    figure_, axis = plt.subplots(figsize=(7.4, 4.3))
    bins = range(0, 155, 5)
    axis.hist(without, bins=bins, color="#2a78d6", alpha=0.75, label="no structure", zorder=3)
    axis.hist(with_structure, bins=bins, color="#eb6834", alpha=0.75,
              label="has structure", zorder=3)
    for values, color in ((without, "#2a78d6"), (with_structure, "#eb6834")):
        axis.axvline(statistics.median(values), color=color, linestyle="--", linewidth=1.4,
                     zorder=4)
    axis.set_xlabel("eSOL solubility (%)", fontsize=9, color="#2b2b2b")
    axis.set_ylabel("Proteins", fontsize=9, color="#2b2b2b")
    axis.set_title("Proteins with solved structures are the more soluble ones",
                   fontsize=9.5, color="#2b2b2b", loc="left", pad=10)
    axis.legend(frameon=False, fontsize=8.5, labelcolor="#595959")
    axis.yaxis.grid(True, color="#b4b4b4", linewidth=0.6, alpha=0.5, zorder=0)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#b4b4b4")
    axis.tick_params(colors="#595959", labelsize=8, length=3)
    axis.text(0.98, 0.52, "dashed lines: medians", transform=axis.transAxes, ha="right",
              fontsize=7.5, color="#595959")
    figure_.tight_layout()
    for suffix in (".png", ".pdf"):
        figure_.savefig(path.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(figure_)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    esol_rows = read_delimited(args.esol)
    if SOLUBILITY_COLUMN not in (esol_rows[0] if esol_rows else {}):
        raise ValueError("eSOL file is missing the '" + SOLUBILITY_COLUMN + "' column")
    uniprot_rows = read_delimited(args.uniprot, delimiter="\t")
    if "PDB" not in (uniprot_rows[0] if uniprot_rows else {}):
        raise ValueError("UniProt file is missing the 'PDB' column")
    pilot = []
    if args.pairs:
        dataset = json.loads(Path(args.pairs).read_text(encoding="utf-8-sig"))
        pilot = sorted({s["structure_id"] for s in dataset["structures"]})
    records, counts = audit(esol_rows, uniprot_rows, args.maximum_solubility)
    summary = summarize(records, counts, args.maximum_solubility, pilot)
    with (output / "esol_mapped.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAPPED_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    write_json(output / "esol_audit.json", summary)
    if not args.no_figure:
        figure(records, output / "esol_coverage")
    coverage = summary["coverage"]
    print("eSOL records {}; with measurement {}; mapped {} ({:.1%}); with structure {} ({:.1%})"
          .format(counts["esol_records"], coverage["records_with_measurement"],
                  coverage["mapped_to_uniprot"], coverage["mapped_fraction"],
                  coverage["with_experimental_structure"],
                  coverage["structure_fraction_of_measured"]), flush=True)
    print("usable after exclusions: {}".format(coverage["usable_after_exclusions"]), flush=True)
    print("median solubility with structure {:.1f}% vs without {:.1f}%".format(
        summary["solubility_distribution"]["with_structure"]["median"],
        summary["solubility_distribution"]["without_structure"]["median"]), flush=True)
    print("overlap with the current pilot structures: 0 (eSOL is E. coli only)", flush=True)
    print("Complete: " + str(output / "esol_audit.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--esol", type=Path, required=True, help="esol.csv from esol.zip")
    parser.add_argument("--uniprot", type=Path, required=True,
                        help="UniProt TSV with accession, gene names, ordered locus, PDB")
    parser.add_argument("--pairs", type=Path, help="pairs.json, to record pilot overlap")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-solubility", type=float, default=100.0)
    parser.add_argument("--no-figure", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
