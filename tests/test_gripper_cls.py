import torch

from vla_sim.models import make_policy
from vla_sim.losses import policy_loss
from vla_sim.losses import aggregate_metrics
from vla_sim.checkpoint import save_checkpoint, load_checkpoint


def test_classifier_gradient_and_checkpoint(tmp_path):
    model = make_policy('bc_gripper_cls', 4, 1)
    inputs = (torch.rand(2, 3, 96, 96), torch.rand(2, 9), torch.ones(2, 16, dtype=torch.long))
    target = torch.zeros(2, 1, 4)
    target[0, 0, 3] = 1
    target[1, 0, 3] = -1
    loss = policy_loss(model, *inputs, target, torch.zeros(2, 1, dtype=torch.bool), torch.tensor(1.5))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.head[-2].weight.grad[-1].abs().sum() > 0
    model.eval()
    result = model(*inputs)
    assert set(result[..., 3].flatten().tolist()) <= {-1.0, 1.0}
    path = tmp_path / 'model.pt'
    save_checkpoint(path, model, model_name='bc_gripper_cls', vocab={'a': 0, 'b': 1, 'c': 2, 'd': 3}, chunk_size=1)
    restored, _ = load_checkpoint(path, torch.device('cpu'))
    torch.testing.assert_close(restored(*inputs), result)


def test_epoch_metrics_weighted_counts():
    batches = [dict(valid_steps=2, loss=1., xyz_loss=.2, gripper_loss=.8, tp=1, fn=0, fp=1, tn=0),
               dict(valid_steps=1, loss=4., xyz_loss=1., gripper_loss=3., tp=0, fn=1, fp=0, tn=0)]
    result = aggregate_metrics(batches)
    assert result['loss'] == 2.
    assert result['closed_recall'] == .5
    assert result['open_false_close_rate'] == 1.
