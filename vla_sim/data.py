from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = ["<pad>", "<unk>"]


def select_state(state, mode):
    if mode == 'full':
        return state
    if mode == 'robot':
        return state[..., [0, 1, 2, 8]]
    raise ValueError(f'Unknown state mode: {mode}')


def tokenize(text: str) -> list[str]:
    return text.lower().replace(",", " ").replace(".", " ").split()


def build_vocab(instructions: list[str]) -> dict[str, int]:
    words = sorted({word for text in instructions for word in tokenize(text)})
    return {word: i for i, word in enumerate(SPECIAL_TOKENS + words)}


def encode_instruction(text: str, vocab: dict[str, int], length: int = 16) -> np.ndarray:
    ids = [vocab.get(word, vocab["<unk>"]) for word in tokenize(text)][:length]
    return np.asarray(ids + [vocab["<pad>"]] * (length - len(ids)), dtype=np.int64)


def create_episode_split(
    episode_records: list[dict], seed: int = 42, train_ratio: float = 0.8, val_ratio: float = 0.1
) -> dict:
    """Create a deterministic task-stratified episode split."""
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1")
    rng = np.random.default_rng(seed)
    by_task: dict[str, list[int]] = {}
    for record in episode_records:
        by_task.setdefault(str(record["task"]), []).append(int(record["episode"]))

    partitions = {"train": [], "val": [], "test": []}
    task_counts: dict[str, dict[str, int]] = {}
    for task, ids in sorted(by_task.items()):
        ids_array = np.asarray(sorted(ids), dtype=np.int64)
        rng.shuffle(ids_array)
        n_train = int(len(ids_array) * train_ratio)
        n_val = int(len(ids_array) * val_ratio)
        task_parts = {
            "train": ids_array[:n_train].tolist(),
            "val": ids_array[n_train : n_train + n_val].tolist(),
            "test": ids_array[n_train + n_val :].tolist(),
        }
        for name, values in task_parts.items():
            partitions[name].extend(values)
        task_counts[task] = {name: len(values) for name, values in task_parts.items()}

    for values in partitions.values():
        values.sort()
    validate_episode_split(partitions, [int(r["episode"]) for r in episode_records])
    return {
        "format_version": 1,
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "task_counts": task_counts,
        **partitions,
    }


def validate_episode_split(split: dict, expected_ids: list[int] | None = None) -> None:
    sets = {name: set(map(int, split[name])) for name in ("train", "val", "test")}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise ValueError("Episode leakage detected: train/val/test are not disjoint")
    if expected_ids is not None and set(expected_ids) != set().union(*sets.values()):
        raise ValueError("Split does not contain every dataset episode exactly once")


class EpisodeDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        chunk_size: int = 16,
        split: str | None = None,
        split_file: str | Path | None = None,
        state_mode: str = 'full',
    ):
        self.root = Path(root)
        self.state_mode = state_mode
        select_state(np.zeros(9), state_mode)
        all_files = sorted(self.root.glob("episode_*.npz"))
        all_ids = [int(path.stem.split("_")[-1]) for path in all_files]
        if split is not None:
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Unknown split {split!r}; expected train, val, or test")
            split_path = Path(split_file) if split_file else self.root / "split.json"
            split_data = json.loads(split_path.read_text(encoding="utf-8"))
            validate_episode_split(split_data, all_ids)
            selected = set(map(int, split_data[split]))
            self.files = [path for path, episode_id in zip(all_files, all_ids) if episode_id in selected]
        else:
            self.files = all_files
        if not self.files:
            raise FileNotFoundError(f"No episodes found for split={split!r} in {self.root}")
        self.episode_ids = [int(path.stem.split("_")[-1]) for path in self.files]
        self.chunk_size = chunk_size
        self.episodes = [dict(np.load(path, allow_pickle=False)) for path in self.files]
        meta = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        self.vocab: dict[str, int] = meta["vocab"]
        self.samples = [(episode_idx, t) for episode_idx, ep in enumerate(self.episodes) for t in range(len(ep["actions"]))]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ep_idx, t = self.samples[index]
        ep = self.episodes[ep_idx]
        actions = ep["actions"][t : t + self.chunk_size]
        valid = len(actions)
        if valid < self.chunk_size:
            padding = np.repeat(actions[-1:], self.chunk_size - valid, axis=0)
            actions = np.concatenate([actions, padding], axis=0)
        mask = np.zeros(self.chunk_size, dtype=np.bool_)
        mask[valid:] = True
        image = torch.from_numpy(ep["images"][t]).permute(2, 0, 1).float() / 255.0
        return {
            "image": image,
            "state": torch.from_numpy(select_state(ep["states"][t], self.state_mode)).float(),
            "tokens": torch.from_numpy(ep["tokens"]).long(),
            "actions": torch.from_numpy(actions).float(),
            "padding_mask": torch.from_numpy(mask),
        }
