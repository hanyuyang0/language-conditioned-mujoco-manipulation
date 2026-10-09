from __future__ import annotations

import torch
from torch import nn


class VisualEncoder(nn.Module):
    def __init__(self, output_dim: int = 128, mode: str = 'global'):
        super().__init__()
        if mode not in ('global', 'spatial'):
            raise ValueError(f'Unknown visual encoder: {mode}')
        grid = 4 if mode == 'spatial' else 1
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(grid), nn.Flatten(), nn.Linear(128 * grid * grid, output_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class LanguageEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        mask = tokens.ne(0).unsqueeze(-1)
        return (embedded * mask).sum(1) / mask.sum(1).clamp_min(1)


class BCPolicy(nn.Module):
    def __init__(self, vocab_size: int, state_dim: int = 9, action_dim: int = 4, chunk_size: int = 1, visual_mode: str = 'global'):
        super().__init__()
        self.chunk_size = chunk_size
        self.visual = VisualEncoder(128, mode=visual_mode)
        self.language = LanguageEncoder(vocab_size, 64)
        self.head = nn.Sequential(
            nn.Linear(128 + 64 + state_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, chunk_size * action_dim), nn.Tanh(),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([self.visual(image), self.language(tokens), state], dim=-1)
        return self.head(fused).view(image.shape[0], self.chunk_size, -1)


class BCGripperClassifier(BCPolicy):
    """Same BC backbone: bounded XYZ regression and an unbounded grip logit."""

    def __init__(self, vocab_size: int, chunk_size: int = 1, state_dim: int = 9, visual_mode: str = 'global'):
        super().__init__(vocab_size, chunk_size=chunk_size, state_dim=state_dim, visual_mode=visual_mode)
        self.head[-1] = nn.Identity()

    def forward_logits(self, image, state, tokens):
        raw = super().forward(image, state, tokens)
        return raw[..., :3].tanh(), raw[..., 3]

    def forward(self, image, state, tokens):
        xyz, logits = self.forward_logits(image, state, tokens)
        grip = torch.where(logits > 0, 1.0, -1.0)
        return torch.cat([xyz, grip.unsqueeze(-1)], dim=-1)


class ACTPolicy(nn.Module):
    """Compact DETR-style action chunking policy for a 4 GB GPU."""

    def __init__(self, vocab_size: int, state_dim: int = 9, action_dim: int = 4,
                 chunk_size: int = 16, hidden_dim: int = 128, visual_mode: str = 'global'):
        super().__init__()
        self.chunk_size = chunk_size
        self.visual = VisualEncoder(hidden_dim, mode=visual_mode)
        self.language = LanguageEncoder(vocab_size, hidden_dim)
        self.state = nn.Linear(state_dim, hidden_dim)
        self.modality_pos = nn.Parameter(torch.randn(3, 1, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, 4, 256, 0.1, batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, 3)
        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, 4, 256, 0.1, batch_first=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, 3)
        self.queries = nn.Parameter(torch.randn(chunk_size, 1, hidden_dim) * 0.02)
        self.action_head = nn.Sequential(nn.Linear(hidden_dim, action_dim), nn.Tanh())

    def forward(self, image: torch.Tensor, state: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        batch = image.shape[0]
        source = torch.stack([self.visual(image), self.state(state), self.language(tokens)], dim=0)
        memory = self.encoder(source + self.modality_pos)
        queries = self.queries.expand(-1, batch, -1)
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded).transpose(0, 1)


def make_policy(name: str, vocab_size: int, chunk_size: int, state_dim: int = 9, visual_mode: str = 'global') -> nn.Module:
    if name == "bc":
        return BCPolicy(vocab_size, chunk_size=chunk_size, state_dim=state_dim, visual_mode=visual_mode)
    if name == "bc_gripper_cls":
        return BCGripperClassifier(vocab_size, chunk_size=chunk_size, state_dim=state_dim, visual_mode=visual_mode)
    if name == "act":
        return ACTPolicy(vocab_size, chunk_size=chunk_size, state_dim=state_dim, visual_mode=visual_mode)
    raise ValueError(f"Unknown model: {name}")
