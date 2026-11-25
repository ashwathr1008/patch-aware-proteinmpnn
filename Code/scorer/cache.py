"""Bounded-memory, resumable SQLite cache of frozen sequence/order features."""
from collections import OrderedDict, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3

import torch
from torch.nn.utils.rnn import pad_sequence
from scorer.data import backbone_features


class FeatureCache:
    def __init__(self, root, extractor, identity, feature_batch_size=8, memory_entries=256):
        if feature_batch_size < 1 or memory_entries < 0:
            raise ValueError("Invalid feature batch/cache size")
        self.extractor = extractor
        self.feature_batch_size = feature_batch_size
        self.memory_entries = memory_entries
        metadata = {"format": 1, "identity": identity, "vocabulary": extractor.vocabulary,
                    "width": extractor.width, "torch": str(torch.__version__)}
        namespace = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        directory = Path(root) / namespace
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "features.sqlite"
        metadata_path = directory / "metadata.json"
        if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("Cache metadata mismatch")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        self.db = sqlite3.connect(self.path)
        self.db.execute("CREATE TABLE IF NOT EXISTS features(key TEXT PRIMARY KEY,length INTEGER,width INTEGER,vectors BLOB,mask BLOB,digest TEXT)")
        self.memory = OrderedDict()
        self.stats = {"hits": 0, "misses": 0, "extractor_calls": 0}

    def close(self):
        self.db.close()

    def _remember(self, key, value):
        if self.memory_entries:
            self.memory[key] = value
            self.memory.move_to_end(key)
            while len(self.memory) > self.memory_entries:
                self.memory.popitem(last=False)

    def get_many(self, features, sequences, order):
        digest = hashlib.sha256()
        for name, tensor in sorted(features.items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}".encode())
            digest.update(value.numpy().tobytes())
        prefix = digest.hexdigest() + "|" + json.dumps(order)
        keys = {s: hashlib.sha256((prefix + "|" + s).encode()).hexdigest() for s in dict.fromkeys(sequences)}
        found, missing = {}, []
        for sequence, key in keys.items():
            value = self.memory.get(key)
            if value is not None:
                self.memory.move_to_end(key)
            else:
                row = self.db.execute("SELECT length,width,vectors,mask,digest FROM features WHERE key=?", (key,)).fetchone()
                if row:
                    length, width, vectors, mask, checksum = row
                    if width != self.extractor.width or length != len(sequence) or hashlib.sha256(vectors + mask).hexdigest() != checksum:
                        raise ValueError("Corrupt/incompatible frozen feature cache entry")
                    value = (torch.frombuffer(bytearray(vectors), dtype=torch.float32).reshape(length, width).clone(),
                             torch.frombuffer(bytearray(mask), dtype=torch.float32).clone())
            if value is None:
                missing.append(sequence)
                self.stats["misses"] += 1
            else:
                found[sequence] = value
                self._remember(key, value)
                self.stats["hits"] += 1
        for start in range(0, len(missing), self.feature_batch_size):
            chunk = missing[start:start + self.feature_batch_size]
            vectors, masks = self.extractor(features, chunk, order)
            self.stats["extractor_calls"] += 1
            for sequence, vector, mask in zip(chunk, vectors, masks):
                value = (vector.detach().cpu().float().contiguous().clone(), mask.detach().cpu().float().contiguous().clone())
                if not all(torch.isfinite(t).all() for t in value):
                    raise ValueError("Cannot cache nonfinite features")
                blobs = [t.numpy().tobytes() for t in value]
                self.db.execute("INSERT OR IGNORE INTO features VALUES (?,?,?,?,?,?)",
                                (keys[sequence], value[0].shape[0], value[0].shape[1], *blobs,
                                 hashlib.sha256(blobs[0] + blobs[1]).hexdigest()))
                found[sequence] = value
                self._remember(keys[sequence], value)
            self.db.commit()  # completed batches survive interruption
        return [found[s] for s in sequences]


def cached_predictions(cache, head, parents, pairs, orders, order_indices, device):
    """Batch variable-length pairs; each pair uses one shared reference/variant order."""
    if not pairs or len(order_indices) != len(pairs):
        raise ValueError("Nonempty pairs and one order index per pair required")
    groups = defaultdict(list)
    for i, (pair, index) in enumerate(zip(pairs, order_indices)):
        groups[(pair["parent_id"], index)].append(i)
    references, variants = [None] * len(pairs), [None] * len(pairs)
    for (pid, index), positions in groups.items():
        sequences = [pairs[i][field] for i in positions for field in ("reference_sequence", "variant_sequence")]
        features = backbone_features(parents[pid], device)
        values = cache.get_many(features, sequences, orders[pid][index])
        for offset, i in enumerate(positions):
            references[i], variants[i] = values[2 * offset:2 * offset + 2]
    values = references + variants
    vectors = pad_sequence([v[0] for v in values], batch_first=True).to(device)
    masks = pad_sequence([v[1] for v in values], batch_first=True).to(device)
    scores = head(vectors, masks)
    return scores[len(pairs):] - scores[:len(pairs)]


def precache(cache, parents, pairs, orders, device):
    sequences = defaultdict(set)
    for pair in pairs:
        sequences[pair["parent_id"]].update((pair["reference_sequence"], pair["variant_sequence"]))
    for pid, entries in sorted(sequences.items()):
        values = sorted(entries)
        for order in orders[pid]:
            # Bound return-list memory even for a parent with many variants.
            for start in range(0, len(values), cache.feature_batch_size):
                cache.get_many(backbone_features(parents[pid], device), values[start:start + cache.feature_batch_size], order)
        print(json.dumps({"cached_parent": pid, **cache.stats}), flush=True)
