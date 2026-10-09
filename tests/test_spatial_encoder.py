import torch

from vla_sim.models import make_policy
from vla_sim.checkpoint import save_checkpoint, load_checkpoint


def test_spatial_checkpoint_backward_and_grid(tmp_path):
    model = make_policy('act', 2, 8, state_dim=4, visual_mode='spatial')
    assert model.visual.net[-1].in_features == 128 * 4 * 4
    inputs = (torch.rand(2,3,96,96), torch.rand(2,4), torch.ones(2,16,dtype=torch.long))
    result = model(*inputs)
    assert result.shape == (2,8,4)
    result.square().mean().backward()
    assert model.visual.net[-1].weight.grad.abs().sum() > 0
    model.eval()
    path = tmp_path / 'spatial.pt'
    save_checkpoint(path, model, model_name='act', vocab={'<pad>':0,'<unk>':1},
                    chunk_size=8, state_mode='robot', visual_mode='spatial')
    loaded, payload = load_checkpoint(path, torch.device('cpu'))
    assert payload['visual_mode'] == 'spatial'
    torch.testing.assert_close(loaded(*inputs), model(*inputs))
