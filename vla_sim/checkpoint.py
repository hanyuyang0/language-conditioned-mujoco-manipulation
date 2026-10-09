from __future__ import annotations

from pathlib import Path

import torch

from vla_sim.models import make_policy


def save_checkpoint(path: str | Path, model, *, model_name: str, vocab: dict[str, int], chunk_size: int, extra=None, state_mode='full', visual_mode='global'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(), "model_name": model_name,
        "vocab": vocab, "chunk_size": chunk_size, "extra": extra or {}, 'state_mode': state_mode,
        'visual_mode': visual_mode,
    }, path)


def load_checkpoint(path: str | Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    mode = payload.get('state_mode', 'full')
    if mode not in ('full', 'robot'):
        raise ValueError(f'Invalid checkpoint state mode: {mode}')
    model = make_policy(payload["model_name"], len(payload["vocab"]), payload["chunk_size"],
                        state_dim=4 if mode == 'robot' else 9,
                        visual_mode=payload.get('visual_mode', 'global'))
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, payload
