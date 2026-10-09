from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vla_sim.checkpoint import save_checkpoint
from vla_sim.data import EpisodeDataset
from vla_sim.models import make_policy
from vla_sim.losses import gripper_positive_weight, policy_loss, aggregate_metrics


def main():
    parser = argparse.ArgumentParser(description="Train BC or compact ACT")
    parser.add_argument("--data", default="datasets/mujoco_language_v2")
    parser.add_argument("--split-file", default=None,
                        help="Episode split JSON (default: <data>/split.json)")
    parser.add_argument("--output", default="runs/act")
    parser.add_argument("--model", choices=["bc", "act", "bc_gripper_cls"], default="act")
    parser.add_argument('--state-mode', choices=['full', 'robot'], default='full')
    parser.add_argument('--visual-mode', choices=['global', 'spatial'], default='global')
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gripper-loss-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.gripper_loss_weight <= 0:
        parser.error('--gripper-loss-weight must be positive')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    split_file = args.split_file or str(Path(args.data) / "split.json")
    train_set = EpisodeDataset(args.data, args.chunk_size, split="train", split_file=split_file, state_mode=args.state_mode)
    val_set = EpisodeDataset(args.data, args.chunk_size, split="val", split_file=split_file, state_mode=args.state_mode)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, args.batch_size, num_workers=args.num_workers)
    model = make_policy(args.model, len(train_set.vocab), args.chunk_size,
                        state_dim=4 if args.state_mode == 'robot' else 9,
                        visual_mode=args.visual_mode).to(device)
    positive_weight = None
    gripper_counts = None
    if args.model == 'bc_gripper_cls':
        weight, gripper_counts = gripper_positive_weight(train_set)
        positive_weight = torch.tensor(weight, device=device)
        print(f'Gripper training labels: {gripper_counts}, pos_weight={weight:.4f}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    print(
        f"device={device}, parameters={sum(p.numel() for p in model.parameters()):,}, "
        f"train_episodes={len(train_set.episode_ids)}, val_episodes={len(val_set.episode_ids)}, "
        f"train_frames={len(train_set)}, val_frames={len(val_set)}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train(); train_losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            image, state, tokens, target, mask = [batch[k].to(device) for k in
                ["image", "state", "tokens", "actions", "padding_mask"]]
            loss, metrics = policy_loss(model, image, state, tokens, target, mask, positive_weight,
                                       args.gripper_loss_weight, return_metrics=True)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_losses.append(metrics)
        model.eval(); val_losses = []
        with torch.inference_mode():
            for batch in val_loader:
                image, state, tokens, target, mask = [batch[k].to(device) for k in
                    ["image", "state", "tokens", "actions", "padding_mask"]]
                _, metrics = policy_loss(model, image, state, tokens, target, mask, positive_weight,
                                         args.gripper_loss_weight, return_metrics=True)
                val_losses.append(metrics)
        row = {'epoch': epoch}
        for prefix, batches in [('train', train_losses), ('val', val_losses)]:
            row.update({f'{prefix}_{key}': value for key, value in aggregate_metrics(batches).items()})
        history.append(row); print(row)
        if row["val_loss"] < best:
            best = row["val_loss"]
            save_checkpoint(output / "best.pt", model, model_name=args.model, vocab=train_set.vocab,
                            chunk_size=args.chunk_size, state_mode=args.state_mode, visual_mode=args.visual_mode,
                            extra={"args": vars(args), "history": history, "split_file": split_file,
                                   "gripper_counts": gripper_counts,
                                   "gripper_pos_weight": float(positive_weight) if positive_weight is not None else None})
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"best validation loss={best:.6f}; checkpoint={output / 'best.pt'}")


if __name__ == "__main__":
    main()
