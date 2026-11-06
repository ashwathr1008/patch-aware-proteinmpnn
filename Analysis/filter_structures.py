"""Filter candidate proteins to those with a usable experimental structure.

The eSOL audit counted UniProt's PDB cross-references per protein. That count is an upper
bound and nothing more: it says a structure exists, not that the structure covers the protein,
resolves it well enough to define a surface, or represents the wild-type sequence. This applies
the criteria that make a structure usable for fixed-backbone patch work, and reports how many
candidates survive.

Criteria, fixed before the metadata was examined:

  method      X-ray diffraction, electron microscopy or solution NMR
  resolution  at most the declared cutoff for diffraction and microscopy; NMR produces no
              resolution figure and is judged on coverage alone
  coverage    the entity's construct length within a declared band of the UniProt length, so
              that fragments and large fusion constructs are both excluded
  mutations   no engineered mutation recorded on the entity

A protein passes if any of its entries passes. Among passing entries one is chosen by a fixed
rule, best resolution then lowest identifier, so the choice never depends on inspection.

Entries are matched to the protein through the entity whose reference sequence identifier is
that protein's UniProt accession. An entry holding several chains therefore contributes only
the chain that actually belongs to the protein in question.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

DIFFRACTION = {"X-RAY DIFFRACTION", "ELECTRON MICROSCOPY", "ELECTRON CRYSTALLOGRAPHY",
               "NEUTRON DIFFRACTION"}
NMR = {"SOLUTION NMR", "SOLID-STATE NMR"}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def entity_for(entry, accession):
    """The polymer entity in this entry that is the protein we asked about."""
    for entity in entry.get("polymer_entities") or []:
        identifiers = entity.get("rcsb_polymer_entity_container_identifiers") or {}
        for reference in identifiers.get("reference_sequence_identifiers") or []:
            if (reference.get("database_name") == "UniProt"
                    and reference.get("database_accession") == accession):
                return entity
    return None


def judge(entry, accession, uniprot_length, max_resolution, coverage_band):
    """Decide one entry, returning the verdict and the reason behind it."""
    methods = [m.get("method") for m in (entry.get("exptl") or []) if m.get("method")]
    method = methods[0] if methods else None
    verdict = {"entry": entry.get("rcsb_id"), "method": method, "resolution": None,
               "coverage": None, "mutation": None, "passes": False, "reason": ""}
    if method is None:
        verdict["reason"] = "no experimental method recorded"
        return verdict
    if method not in DIFFRACTION and method not in NMR:
        verdict["reason"] = "unsupported method"
        return verdict
    resolutions = ((entry.get("rcsb_entry_info") or {}).get("resolution_combined")) or []
    resolution = min(resolutions) if resolutions else None
    verdict["resolution"] = resolution
    if method in DIFFRACTION:
        if resolution is None:
            verdict["reason"] = "no resolution reported"
            return verdict
        if resolution > max_resolution:
            verdict["reason"] = "resolution above cutoff"
            return verdict
    entity = entity_for(entry, accession)
    if entity is None:
        verdict["reason"] = "no entity maps to this accession"
        return verdict
    mutation = (entity.get("rcsb_polymer_entity") or {}).get("pdbx_mutation")
    verdict["mutation"] = mutation
    if mutation:
        verdict["reason"] = "engineered mutation"
        return verdict
    length = (entity.get("entity_poly") or {}).get("rcsb_sample_sequence_length")
    if not length or not uniprot_length:
        verdict["reason"] = "no construct length"
        return verdict
    coverage = length / uniprot_length
    verdict["coverage"] = coverage
    low, high = coverage_band
    if not low <= coverage <= high:
        verdict["reason"] = "construct length outside coverage band"
        return verdict
    verdict["passes"] = True
    verdict["reason"] = "ok"
    return verdict


def choose(passing):
    """Fixed selection rule: best resolution first, NMR last, then lowest identifier."""
    return sorted(passing, key=lambda v: (v["resolution"] is None,
                                          v["resolution"] if v["resolution"] is not None else 0.0,
                                          v["entry"]))[0]


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    band = (args.min_coverage, args.max_coverage)
    kept, dropped, reasons = [], [], Counter()
    for candidate in candidates:
        verdicts = []
        for identifier in candidate["pdb"]:
            entry = metadata.get(identifier)
            if entry is None:
                verdicts.append({"entry": identifier, "passes": False,
                                 "reason": "no metadata returned"})
                continue
            verdicts.append(judge(entry, candidate["acc"], candidate["len"],
                                  args.max_resolution, band))
        passing = [v for v in verdicts if v["passes"]]
        record = {**{k: candidate[k] for k in ("b", "acc", "gene", "len", "sol")},
                  "entries_considered": len(verdicts), "entries_passing": len(passing)}
        if passing:
            record["chosen"] = choose(passing)
            kept.append(record)
        else:
            # Attribute the protein to the reason its best entry failed on.
            order = ["engineered mutation", "construct length outside coverage band",
                     "resolution above cutoff", "no entity maps to this accession",
                     "no resolution reported", "no construct length", "unsupported method",
                     "no experimental method recorded", "no metadata returned"]
            seen = {v.get("reason") for v in verdicts}
            record["reason"] = next((r for r in order if r in seen), "unknown")
            reasons[record["reason"]] += 1
            dropped.append(record)
    summary = {
        "schema_version": 1, "purpose": "structure_quality_filter",
        "criteria": {"methods_allowed": sorted(DIFFRACTION | NMR),
                     "max_resolution_angstrom": args.max_resolution,
                     "coverage_band": list(band),
                     "mutations_allowed": False,
                     "declared_before_inspection": True},
        "selection_rule": "best resolution, NMR last, then lowest entry identifier",
        "candidates": len(candidates), "kept": len(kept), "dropped": len(dropped),
        "drop_reasons": dict(reasons.most_common()),
    }
    if kept:
        solubility = [r["sol"] for r in kept]
        lengths = [r["len"] for r in kept]
        summary["survivors"] = {
            "solubility_median": statistics.median(solubility),
            "solubility_sd": statistics.pstdev(solubility),
            "solubility_below_60": sum(1 for v in solubility if v < 60),
            "length_median": statistics.median(lengths),
            "resolution_median": statistics.median(
                [r["chosen"]["resolution"] for r in kept if r["chosen"]["resolution"]] or [0]),
        }
    write_json(output / "structure_filter.json", summary)
    write_json(output / "filtered_candidates.json", kept)
    print("candidates {} -> kept {} ({:.0%})".format(
        len(candidates), len(kept), len(kept) / max(1, len(candidates))), flush=True)
    for reason, count in reasons.most_common():
        print("  dropped {:4d}  {}".format(count, reason), flush=True)
    if kept:
        s = summary["survivors"]
        print("  survivors: solubility median {:.0f}%, SD {:.1f}, {} below 60%; length median "
              "{:.0f}".format(s["solubility_median"], s["solubility_sd"],
                              s["solubility_below_60"], s["length_median"]), flush=True)
    print("Complete: " + str(output / "structure_filter.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True, help="JSON list from the eSOL audit")
    parser.add_argument("--metadata", required=True, help="JSON map of PDB entry metadata")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-resolution", type=float, default=2.5)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--max-coverage", type=float, default=1.25)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
