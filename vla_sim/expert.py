from __future__ import annotations

import numpy as np

from vla_sim.env import LanguageManipulationEnv


def _move(current: np.ndarray, target: np.ndarray, grip: float) -> np.ndarray:
    scale = np.array([0.035, 0.035, 0.025])
    delta = np.clip((target - current) / scale, -1.0, 1.0)
    return np.r_[delta, grip].astype(np.float32)


def expert_action(env: LanguageManipulationEnv) -> np.ndarray:
    """Finite-state scripted expert for dataset generation."""
    g, obj = env.gripper, env.object_pos
    goal = np.r_[env.target_xy, 0.035]
    if env.task == "pick_place":
        if env.attached:
            if g[2] < 0.19 and np.linalg.norm(g[:2] - goal[:2]) > 0.07:
                return _move(g, np.r_[g[:2], 0.23], 1.0)
            if np.linalg.norm(g[:2] - goal[:2]) > 0.035:
                return _move(g, np.r_[goal[:2], 0.23], 1.0)
            if g[2] > 0.085:
                return _move(g, np.r_[goal[:2], 0.075], 1.0)
            return _move(g, g, -1.0)
        if np.linalg.norm(g[:2] - obj[:2]) > 0.025:
            return _move(g, np.r_[obj[:2], 0.16], -1.0)
        if g[2] > obj[2] + 0.045:
            return _move(g, np.r_[obj[:2], obj[2] + 0.035], -1.0)
        return _move(g, g, 1.0)

    direction = goal[:2] - obj[:2]
    direction /= np.linalg.norm(direction) + 1e-8
    behind = obj[:2] - direction * 0.070
    if g[2] > 0.082:
        return _move(g, np.r_[g[:2], 0.075], -1.0)
    lateral_error = np.linalg.norm(g[:2] - behind)
    if lateral_error > 0.025:
        return _move(g, np.r_[behind, 0.075], -1.0)
    return _move(g, np.r_[goal[:2], 0.075], -1.0)
