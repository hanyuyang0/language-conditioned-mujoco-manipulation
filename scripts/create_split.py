from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_sim.data import create_episode_split, validate_episode_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Create task-balanced episode-level train/val/test splits")
    parser.add_argument("--data", default="datasets/mujoco_language_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.data)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    split = create_episode_split(metadata["episodes"], args.seed, args.train_ratio, args.val_ratio)
    episode_ids = [int(record["episode"]) for record in metadata["episodes"]]
    validate_episode_split(split, episode_ids)
    output = Path(args.output) if args.output else root / "split.json"
    output.write_text(json.dumps(split, indent=2), encoding="utf-8")

    print(f"saved: {output.resolve()}")
    print(f"train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")
    for task, counts in split["task_counts"].items():
        print(f"{task}: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    print("overlap check: passed")


if __name__ == "__main__":
    main()
