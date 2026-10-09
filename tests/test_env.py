import numpy as np

from vla_sim.env import LanguageManipulationEnv
from vla_sim.expert import expert_action


def test_shapes_and_determinism():
    env = LanguageManipulationEnv(image_size=64, max_steps=10, seed=1)
    try:
        obs1, _ = env.reset(seed=123, options={"task": "push"})
        obs2, _ = env.reset(seed=123, options={"task": "push"})
        assert obs1["image"].shape == (64, 64, 3)
        assert obs1["state"].shape == (9,)
        np.testing.assert_allclose(obs1["state"], obs2["state"])
        action = expert_action(env)
        assert action.shape == (4,)
        assert np.all(np.abs(action) <= 1)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)
        assert "success" in info
    finally:
        env.close()


def test_expert_reaches_tasks():
    env = LanguageManipulationEnv(image_size=48, max_steps=120, seed=2)
    try:
        for task in ["push", "pick_place"]:
            _, _ = env.reset(seed=10, options={"task": task})
            success = False
            for _ in range(120):
                _, _, success, truncated, _ = env.step(expert_action(env))
                if success or truncated:
                    break
            assert success, f"expert failed {task}"
    finally:
        env.close()


def test_pick_place_requires_release():
    env = LanguageManipulationEnv(image_size=48, max_steps=20, seed=3)
    try:
        env.reset(
            options={
                "task": "pick_place",
                "object_xy": np.array([-0.1, 0.0]),
                "target_xy": np.array([0.2, 0.1]),
            }
        )
        env._set_object(np.r_[env.target_xy, 0.035])
        env.attached = True
        assert env._success() is False
        env.attached = False
        assert env._success() is True
    finally:
        env.close()
