"""A fixed-backbone SAP surrogate: burial-weighted hydrophobicity on a frozen backbone.

Spatial Aggregation Propensity (Chennamsetty et al., PNAS 2009) weights each residue's
hydrophobicity by how exposed it is, and sums that over a spatial neighbourhood. The published
statistic uses side-chain solvent-accessible area from a packed structure. This module computes
a surrogate that needs no repacking, which matters here for two reasons.

First, the staged inputs in this project carry backbone atoms only (N, CA, C, O), so side-chain
area is not available to compute. Second, and more importantly, the burial term *must not*
depend on residue identity. These experiments compare composition-matched assignments on one
frozen backbone, so any identity-dependent burial term would let native side-chain chemistry
leak into the permuted features and would score the arrangement by construction rather than by
measurement. Deriving burial from the backbone plus a virtual CB keeps it identity-independent:
the same geometric scaffold serves every assignment on that structure.

  burial   relative accessibility of the five-atom set {N, CA, C, O, CB} at each position,
           normalized by the same residue's accessibility in isolation, so it lies in [0, 1]
  sap_simple   sum_i burial_i * h(a_i)
  sap_window   sum_i sum_{j within R angstrom of i} burial_j * h(a_j)

`sap_simple` is the form a per-position burial-weighted composition takes. `sap_window` is the
spatially averaged form, which is what makes SAP a patch statistic rather than a per-residue
one: it rewards hydrophobic residues that are exposed *and clustered together*.

h is the Black and Mould (1991) normalized hydrophobicity scale, shifted so that glycine sits
at zero, which is the convention SAP uses. Hydrophilic residues therefore contribute negative
values and hydrophobic ones positive.

The virtual CB formula is the one ProteinMPNN itself uses for its neighbour graph, so burial
here is defined on the same geometric construct the model sees.

Accessibility is computed with the Shrake-Rupley algorithm: a sphere of test points around each
atom, counting the fraction not occluded by neighbouring atoms.
"""

import math

import numpy as np

# Black and Mould (1991) normalized hydrophobicity. Glycine is 0.501 on this scale.
BLACK_MOULD = {
    "A": 0.616, "R": 0.000, "N": 0.236, "D": 0.028, "C": 0.680,
    "Q": 0.251, "E": 0.043, "G": 0.501, "H": 0.165, "I": 0.943,
    "L": 0.943, "K": 0.283, "M": 0.738, "F": 1.000, "P": 0.711,
    "S": 0.359, "T": 0.450, "W": 0.878, "Y": 0.880, "V": 0.825,
}
GLYCINE_REFERENCE = BLACK_MOULD["G"]
HYDROPHOBICITY = {letter: value - GLYCINE_REFERENCE for letter, value in BLACK_MOULD.items()}
ALPHABET = "".join(sorted(BLACK_MOULD))

# Van der Waals radii in angstrom, and the water probe.
ATOM_RADII = {"N": 1.55, "CA": 1.70, "C": 1.70, "O": 1.52, "CB": 1.70}
BACKBONE_ATOMS = ("N", "CA", "C", "O")
RESIDUE_ATOMS = BACKBONE_ATOMS + ("CB",)
PROBE_RADIUS = 1.4
DEFAULT_SPHERE_POINTS = 960
DEFAULT_WINDOW_RADIUS = 10.0


def sphere_points(count):
    """Near-uniform points on the unit sphere by the golden-angle spiral."""
    if count < 1:
        raise ValueError("A positive number of sphere points is required")
    indices = np.arange(count, dtype="float64")
    z = 1.0 - (2.0 * indices + 1.0) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = indices * math.pi * (3.0 - math.sqrt(5.0))
    return np.stack([radius * np.cos(theta), radius * np.sin(theta), z], axis=1)


def virtual_cb(backbone):
    """Ideal CB positions from N, CA and C, using ProteinMPNN's own coefficients."""
    backbone = np.asarray(backbone, dtype="float64")
    if backbone.ndim != 3 or backbone.shape[1:] != (4, 3):
        raise ValueError("Backbone must have shape (length, 4, 3) ordered N, CA, C, O")
    if not np.isfinite(backbone).all():
        raise ValueError("Backbone holds nonfinite coordinates")
    n, ca, c = backbone[:, 0], backbone[:, 1], backbone[:, 2]
    b, d = ca - n, c - ca
    a = np.cross(b, d)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * d + ca


def residue_atoms(backbone):
    """Stack the five-atom representation, returning coordinates and radii."""
    backbone = np.asarray(backbone, dtype="float64")
    coordinates = np.concatenate([backbone, virtual_cb(backbone)[:, None, :]], axis=1)
    radii = np.array([ATOM_RADII[name] for name in RESIDUE_ATOMS], dtype="float64")
    return coordinates, np.tile(radii, (backbone.shape[0], 1))


def atom_sasa(coordinates, radii, points):
    """Per-atom solvent-accessible area by Shrake-Rupley, in square angstrom."""
    coordinates = np.asarray(coordinates, dtype="float64").reshape(-1, 3)
    radii = np.asarray(radii, dtype="float64").reshape(-1)
    if coordinates.shape[0] != radii.shape[0]:
        raise ValueError("Coordinate and radius counts differ")
    expanded = radii + PROBE_RADIUS
    areas = np.empty(coordinates.shape[0], dtype="float64")
    separation = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
    for index in range(coordinates.shape[0]):
        # Only atoms whose expanded spheres can reach this one can occlude it.
        reach = expanded[index] + expanded
        neighbours = np.flatnonzero((separation[index] < reach) & (np.arange(len(radii)) != index))
        shell = coordinates[index] + expanded[index] * points
        if neighbours.size:
            distances = np.linalg.norm(
                shell[:, None, :] - coordinates[neighbours][None, :, :], axis=-1)
            exposed = np.all(distances >= expanded[neighbours][None, :], axis=1)
        else:
            exposed = np.ones(shell.shape[0], dtype=bool)
        areas[index] = (4.0 * math.pi * expanded[index] ** 2) * exposed.mean()
    return areas


def burial_profile(backbone, sphere_point_count=DEFAULT_SPHERE_POINTS):
    """Relative accessibility per residue in [0, 1], independent of residue identity.

    Each position's five-atom area in the assembled chain is divided by the same five atoms'
    area in isolation, so the reference uses identical geometry and no amino-acid lookup.
    """
    coordinates, radii = residue_atoms(backbone)
    length = coordinates.shape[0]
    points = sphere_points(sphere_point_count)
    context = atom_sasa(coordinates.reshape(-1, 3), radii.reshape(-1), points)
    context = context.reshape(length, len(RESIDUE_ATOMS)).sum(axis=1)
    isolated = np.array([
        atom_sasa(coordinates[i], radii[i], points).sum() for i in range(length)])
    if np.any(isolated <= 0):
        raise ValueError("An isolated residue has no accessible area")
    relative = context / isolated
    if np.any(relative < -1e-9) or np.any(relative > 1.0 + 1e-6):
        raise ValueError("Relative accessibility fell outside [0, 1]")
    return np.clip(relative, 0.0, 1.0)


def hydrophobicity_vector(sequence):
    unknown = set(sequence) - set(HYDROPHOBICITY)
    if unknown:
        raise ValueError("Sequence holds noncanonical residues: " + "".join(sorted(unknown)))
    return np.array([HYDROPHOBICITY[letter] for letter in sequence], dtype="float64")


def window_matrix(backbone, radius=DEFAULT_WINDOW_RADIUS):
    """Boolean neighbourhood over virtual CB positions, including each residue itself."""
    if radius <= 0:
        raise ValueError("The window radius must be positive")
    centres = virtual_cb(backbone)
    separation = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1)
    return separation <= radius


def sap_profile(burial, sequence, window=None):
    """Per-residue SAP contributions, optionally summed over a spatial neighbourhood."""
    burial = np.asarray(burial, dtype="float64")
    if burial.ndim != 1 or burial.shape[0] != len(sequence):
        raise ValueError("Burial profile and sequence lengths differ")
    weighted = burial * hydrophobicity_vector(sequence)
    if window is None:
        return weighted
    window = np.asarray(window, dtype=bool)
    if window.shape != (len(sequence), len(sequence)):
        raise ValueError("Window matrix shape does not match the sequence")
    return window @ weighted


def score(burial, sequence, window=None):
    """Total SAP surrogate for one assignment on a fixed backbone."""
    return float(sap_profile(burial, sequence, window).sum())


def descriptors(burial, sequence, window, positions=None):
    """Simple and windowed SAP totals, over the chain and over selected positions."""
    simple = sap_profile(burial, sequence)
    windowed = sap_profile(burial, sequence, window)
    result = {"sap_simple": float(simple.sum()), "sap_window": float(windowed.sum())}
    if positions is not None:
        index = np.asarray(list(positions), dtype=int)
        if index.size and (index.min() < 0 or index.max() >= len(sequence)):
            raise ValueError("Patch position out of range")
        result["sap_simple_patch"] = float(simple[index].sum())
        result["sap_window_patch"] = float(windowed[index].sum())
        result["burial_patch_mean"] = float(np.asarray(burial)[index].mean())
    return result
