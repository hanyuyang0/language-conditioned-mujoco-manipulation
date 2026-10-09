import torch

from vla_sim.models import make_policy


def test_policy_shapes():
    for name in ["bc", "act"]:
        model = make_policy(name, vocab_size=30, chunk_size=8)
        output = model(torch.rand(2, 3, 96, 96), torch.rand(2, 9), torch.ones(2, 16, dtype=torch.long))
        assert output.shape == (2, 8, 4)
        assert torch.isfinite(output).all()

