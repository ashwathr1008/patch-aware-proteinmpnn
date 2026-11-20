"""Frozen, teacher-forced ProteinMPNN features and a trainable stability head."""
import hashlib
import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

VOCABULARY = "ACDEFGHIKLMNPQRSTVWYX"  # pinned ProteinMPNN run-script vocabulary


def weight_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_backbone(mpnn_dir, checkpoint_path, device="cpu"):
    """Infer architecture from a standard full-backbone ProteinMPNN checkpoint."""
    from baseline.run_baseline import UPSTREAM_HASHES, sha256
    source = Path(mpnn_dir) / "protein_mpnn_utils.py"
    # Git on Windows may check out CRLF; compare identical source normalized to LF.
    source_digest = hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if source_digest != UPSTREAM_HASHES["protein_mpnn_utils.py"]:
        raise ValueError("ProteinMPNN source differs from repository-pinned implementation")
    spec = importlib.util.spec_from_file_location("stability_mpnn_utils", source)
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = checkpoint["model_state_dict"]
    layers = lambda prefix: len({int(k.split(".")[1]) for k in state if k.startswith(prefix + ".")})
    model = utils.ProteinMPNN(
        num_letters=state["W_out.weight"].shape[0],
        node_features=state["W_v.weight"].shape[0] if "W_v.weight" in state else state["W_e.weight"].shape[0],
        edge_features=state["W_e.weight"].shape[1],
        hidden_dim=state["W_out.weight"].shape[1],
        num_encoder_layers=layers("encoder_layers"), num_decoder_layers=layers("decoder_layers"),
        augment_eps=0.0, k_neighbors=checkpoint["num_edges"], ca_only=False).to(device)
    model.load_state_dict(state, strict=True)
    if model.W_out.out_features != len(VOCABULARY):
        raise ValueError("Unsupported checkpoint vocabulary; do not guess token mapping")
    return FrozenDecoder(model), utils


class FrozenDecoder(nn.Module):
    def __init__(self, model, vocabulary=VOCABULARY):
        super().__init__()
        self.model = model
        self.vocabulary = vocabulary
        self.model.requires_grad_(False)
        self.model.augment_eps = 0.0
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        self.model.eval()
        return self

    @property
    def width(self):
        return self.model.W_out.in_features + len(self.vocabulary)

    def forward(self, features, sequences, order):
        self.model.eval()
        self.model.augment_eps = 0.0
        length = features["X"].shape[1]
        if sorted(order) != list(range(length)):
            raise ValueError("Decoding order must be a complete zero-based permutation")
        if any(len(s) != length or set(s) - set(self.vocabulary.replace("X", "")) for s in sequences):
            raise ValueError("Sequences must be complete aligned canonical sequences")
        device = features["X"].device
        tokens = torch.tensor([[self.vocabulary.index(a) for a in s] for s in sequences], device=device)
        count = len(sequences)
        expand = lambda x: x.expand(count, *([-1] * (x.ndim - 1))).contiguous()
        captured = []
        # Capture the actual projection input; do not reimplement decoder masks or layers.
        hook = self.model.W_out.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        try:
            with torch.no_grad():
                self.model(expand(features["X"]), tokens, expand(features["mask"]),
                           expand(features["chain_M"]), expand(features["residue_idx"]),
                           expand(features["chain_encoding_all"]), torch.zeros(count, length, device=device),
                           use_input_decoding_order=True,
                           decoding_order=torch.tensor([order], device=device).expand(count, -1))
                if len(captured) != 1 or not torch.isfinite(captured[0]).all():
                    raise ValueError("Expected one finite final decoder-state tensor")
                identities = F.one_hot(tokens, len(self.vocabulary)).to(captured[0].dtype)
                return torch.cat((captured[0], identities), dim=-1), expand(features["mask"])
        finally:
            hook.remove()


class StabilityHead(nn.Module):
    def __init__(self, input_width, hidden_width=64):
        super().__init__()
        self.residue_mlp = nn.Sequential(nn.Linear(input_width, hidden_width), nn.ReLU(),
                                         nn.Linear(hidden_width, hidden_width), nn.ReLU())
        # A shared output bias cancels in pair differences and cannot be learned.
        self.output = nn.Linear(hidden_width, 1, bias=False)

    def forward(self, features, mask):
        if (mask.sum(dim=1) <= 0).any():
            raise ValueError("Cannot pool an empty backbone")
        values = self.residue_mlp(features)
        pooled = (values * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        return self.output(pooled).squeeze(-1)


def pair_predictions(extractor, head, features, reference, variants, orders):
    """Variant minus reference, averaged over shared fixed orders; no sampling."""
    if not variants or not orders:
        raise ValueError("Candidates and decoding orders must be nonempty")
    predictions = []
    for order in orders:
        vectors, mask = extractor(features, [reference] + list(variants), order)
        scores = head(vectors, mask)
        predictions.append(scores[1:] - scores[0])
    return torch.stack(predictions).mean(0)


@torch.no_grad()
def score_swapper_candidates(extractor, head, features, reference, candidates, orders):
    """Accept existing swapper pair records; returns predictions in kcal/mol, positive destabilizing."""
    from collections import Counter
    variants = []
    for candidate in candidates:
        if candidate["original_sequence"] != reference:
            raise ValueError("Swapper reference differs from aligned reference")
        variant = candidate["permuted_sequence"]
        patch = {i - 1 for i in candidate["patch_indices"]}
        if not patch or min(patch) < 0 or max(patch) >= len(reference):
            raise ValueError("Invalid one-based swapper patch indices")
        if len(variant) != len(reference) or Counter(variant) != Counter(reference):
            raise ValueError("Candidate must preserve reference composition")
        if any(a != b and i not in patch for i, (a, b) in enumerate(zip(reference, variant))):
            raise ValueError("Candidate changes identities outside patch")
        variants.append(variant)
    head.eval()
    values = pair_predictions(extractor, head, features, reference, variants, orders).cpu().tolist()
    return [{"pair_id": c["pair_id"], "ddg_kcal_mol": v} for c, v in zip(candidates, values)]
