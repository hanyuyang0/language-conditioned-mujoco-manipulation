import numpy as np
import torch

from vla_sim.data import select_state
from vla_sim.models import make_policy
from vla_sim.checkpoint import save_checkpoint, load_checkpoint
from vla_sim.inference import predict_action


def test_robot_state_excludes_object_and_target():
    state = np.arange(9, dtype=np.float32)
    changed = state.copy()
    changed[3:8] = 1000
    np.testing.assert_array_equal(select_state(state, 'robot'), [0, 1, 2, 8])
    np.testing.assert_array_equal(select_state(state, 'robot'), select_state(changed, 'robot'))


def test_robot_checkpoint_and_inference(tmp_path):
    vocab = {'<pad>': 0, '<unk>': 1}
    model = make_policy('act', 2, 8, state_dim=4)
    pred = model(torch.rand(2, 3, 96, 96), torch.rand(2, 4), torch.ones(2, 16, dtype=torch.long))
    pred.sum().backward()
    path = tmp_path / 'robot.pt'
    save_checkpoint(path, model, model_name='act', vocab=vocab, chunk_size=8, state_mode='robot')
    loaded, payload = load_checkpoint(path, torch.device('cpu'))
    action, _ = predict_action(loaded, payload, {'image': np.zeros((96,96,3), dtype=np.uint8),
        'state': np.zeros(9, dtype=np.float32)}, 'unknown', torch.device('cpu'))
    assert action.shape == (4,)
    assert np.isfinite(action).all()
