from pathlib import Path

import numpy as np

from scripts.evaluate_split import load_episode_initial_conditions


def test_load_episode_initial_conditions(tmp_path: Path):
    state = np.asarray([0, 0, 0, -0.2, 0.1, 0.035, 0.3, -0.1, 0], dtype=np.float32)
    path = tmp_path / "episode_00000.npz"
    np.savez_compressed(
        path,
        states=state[None],
        task=np.asarray("push"),
        instruction=np.asarray("push the block"),
        success=np.asarray(True),
    )
    initial = load_episode_initial_conditions(path)
    assert initial["task"] == "push"
    assert initial["instruction"] == "push the block"
    np.testing.assert_allclose(initial["object_xy"], [-0.2, 0.1])
    np.testing.assert_allclose(initial["target_xy"], [0.3, -0.1])
    assert initial["expert_success"] is True

