from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


TASK_INSTRUCTIONS = {
    "push": [
        "push the red block to the green target",
        "move the red cube onto the green circle",
        "slide the red object to the goal",
    ],
    "pick_place": [
        "pick up the red block and place it on the green target",
        "grasp the red cube and put it at the goal",
        "lift the red object and move it to the green circle",
    ],
}


def _xml() -> str:
    return r"""
<mujoco model="language_manipulation">
  <compiler angle="radian"/>
  <option timestep="0.01" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual><global offwidth="256" offheight="256"/><quality shadowsize="1024"/></visual>
  <asset>
    <material name="table" rgba="0.72 0.74 0.78 1"/>
    <material name="red" rgba="0.85 0.08 0.06 1"/>
    <material name="green" rgba="0.05 0.75 0.25 0.55"/>
    <material name="robot" rgba="0.16 0.22 0.32 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 2.5" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="table" type="plane" size="0.75 0.60 0.04" material="table"/>
    <camera name="top" pos="0 -0.02 1.65" xyaxes="1 0 0 0 1 0"/>
    <body name="target" pos="0.25 0.18 0.006">
      <geom name="target_geom" type="cylinder" size="0.075 0.006" material="green" contype="0" conaffinity="0"/>
    </body>
    <body name="object" pos="-0.18 0 0.035">
      <joint name="object_x" type="slide" axis="1 0 0" damping="3"/>
      <joint name="object_y" type="slide" axis="0 1 0" damping="3"/>
      <joint name="object_z" type="slide" axis="0 0 1" damping="3" limited="true" range="0 0.28"/>
      <geom name="object_geom" type="box" size="0.038 0.038 0.035" material="red" mass="0.08" friction="1.2 0.05 0.01"/>
    </body>
    <body name="carriage" pos="0 0 0.16">
      <joint name="gripper_x" type="slide" axis="1 0 0" limited="true" range="-0.50 0.50" damping="12"/>
      <joint name="gripper_y" type="slide" axis="0 1 0" limited="true" range="-0.42 0.42" damping="12"/>
      <joint name="gripper_z" type="slide" axis="0 0 1" limited="true" range="-0.10 0.25" damping="12"/>
      <geom name="palm" type="box" size="0.055 0.026 0.018" material="robot" contype="0"/>
      <geom name="finger_l" pos="0 0.050 -0.025" type="box" size="0.018 0.010 0.038" material="robot" contype="0"/>
      <geom name="finger_r" pos="0 -0.050 -0.025" type="box" size="0.018 0.010 0.038" material="robot" contype="0"/>
    </body>
  </worldbody>
  <actuator>
    <position name="gx" joint="gripper_x" kp="250" ctrlrange="-0.50 0.50"/>
    <position name="gy" joint="gripper_y" kp="250" ctrlrange="-0.42 0.42"/>
    <position name="gz" joint="gripper_z" kp="250" ctrlrange="-0.10 0.25"/>
  </actuator>
</mujoco>
"""


class LanguageManipulationEnv(gym.Env):
    """Small MuJoCo benchmark for low-VRAM imitation-learning experiments.

    The robot is a Cartesian XYZ gripper. Grasp attachment is modeled as a
    deterministic constraint, while pushing is simulated as planar contact-like
    displacement. This keeps data generation reliable and the benchmark fast.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, image_size: int = 96, max_steps: int = 100, seed: int | None = None):
        super().__init__()
        self.image_size = image_size
        self.max_steps = max_steps
        self.model = mujoco.MjModel.from_xml_string(_xml())
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=image_size, width=image_size)
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, shape=(image_size, image_size, 3), dtype=np.uint8),
            "state": spaces.Box(-2, 2, shape=(9,), dtype=np.float32),
        })
        self.task = "push"
        self.instruction = TASK_INSTRUCTIONS["push"][0]
        self.attached = False
        self.grip_closed = False
        self.step_count = 0

    @property
    def gripper(self) -> np.ndarray:
        return self.data.qpos[3:6] + np.array([0.0, 0.0, 0.16])

    @property
    def object_pos(self) -> np.ndarray:
        return self.data.qpos[:3] + np.array([0.0, 0.0, 0.035])

    @property
    def target_xy(self) -> np.ndarray:
        return self.model.body_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")][:2].copy()

    def _set_object(self, xyz: np.ndarray) -> None:
        self.data.qpos[:3] = xyz - np.array([0.0, 0.0, 0.035])
        self.data.qvel[:3] = 0

    def _set_gripper(self, xyz: np.ndarray) -> None:
        self.data.qpos[3:6] = xyz - np.array([0.0, 0.0, 0.16])
        self.data.ctrl[:] = self.data.qpos[3:6]
        self.data.qvel[3:6] = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        self.task = options.get("task", self.rng.choice(list(TASK_INSTRUCTIONS)))
        self.instruction = options.get("instruction", self.rng.choice(TASK_INSTRUCTIONS[self.task]))
        object_xy = np.asarray(options.get("object_xy", self.rng.uniform([-0.30, -0.22], [0.0, 0.22])), dtype=float)
        target_xy = np.asarray(options.get("target_xy", self.rng.uniform([0.14, -0.25], [0.36, 0.25])), dtype=float)
        mujoco.mj_resetData(self.model, self.data)
        self.model.body_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")][:2] = target_xy
        self._set_object(np.r_[object_xy, 0.035])
        gripper_xyz = np.asarray(options.get('gripper_xyz', [-0.40, 0.0, 0.18]), dtype=float)
        if gripper_xyz.shape != (3,) or not np.isfinite(gripper_xyz).all():
            raise ValueError('gripper_xyz must contain three finite coordinates')
        if np.any(gripper_xyz < [-0.50, -0.42, 0.055]) or np.any(gripper_xyz > [0.50, 0.42, 0.38]):
            raise ValueError('gripper_xyz is outside the workspace')
        self._set_gripper(gripper_xyz)
        self.attached = False
        self.grip_closed = False
        self.step_count = 0
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), self._info()

    def _success(self) -> bool:
        placed = bool(
            np.linalg.norm(self.object_pos[:2] - self.target_xy) < 0.075
            and self.object_pos[2] < 0.075
        )
        if self.task == "pick_place":
            return placed and not self.attached
        return placed

    def _info(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "instruction": self.instruction,
            "success": self._success(),
            "distance": float(np.linalg.norm(self.object_pos[:2] - self.target_xy)),
            "attached": self.attached,
        }

    def _observation(self) -> dict[str, np.ndarray]:
        self.renderer.update_scene(self.data, camera="top")
        image = self.renderer.render().copy()
        state = np.r_[self.gripper, self.object_pos, self.target_xy, float(self.attached)].astype(np.float32)
        return {"image": image, "state": state}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        old_gripper = self.gripper.copy()
        target = old_gripper + action[:3] * np.array([0.035, 0.035, 0.025])
        target = np.clip(target, [-0.50, -0.42, 0.055], [0.50, 0.42, 0.38])
        self.grip_closed = bool(action[3] > 0)
        if self.grip_closed and not self.attached:
            close_xy = np.linalg.norm(old_gripper[:2] - self.object_pos[:2]) < 0.065
            close_z = abs(old_gripper[2] - self.object_pos[2]) < 0.07
            self.attached = bool(close_xy and close_z)
        if not self.grip_closed:
            self.attached = False

        # The benchmark uses a kinematic Cartesian controller. MuJoCo still
        # owns scene state and rendering, but the commanded end-effector delta
        # is exact and therefore independent of actuator tuning.
        self._set_gripper(target)
        mujoco.mj_forward(self.model, self.data)

        if self.attached:
            self._set_object(self.gripper + np.array([0.0, 0.0, -0.045]))
        elif self.task == "push" and self.gripper[2] < 0.10:
            offset = self.object_pos[:2] - self.gripper[:2]
            if np.linalg.norm(offset) < 0.075:
                moved = self.object_pos.copy()
                travel = float(np.linalg.norm((self.gripper - old_gripper)[:2]))
                to_goal = self.target_xy - moved[:2]
                moved[:2] += to_goal / (np.linalg.norm(to_goal) + 1e-8) * travel * 0.95
                moved[:2] = np.clip(moved[:2], [-0.48, -0.40], [0.48, 0.40])
                moved[2] = 0.035
                self._set_object(moved)
        elif not self.attached and self.object_pos[2] > 0.035:
            obj = self.object_pos.copy()
            obj[2] = max(0.035, obj[2] - 0.025)
            self._set_object(obj)

        self.step_count += 1
        success = self._success()
        truncated = self.step_count >= self.max_steps
        reward = float(success) - 0.02 * float(np.linalg.norm(self.object_pos[:2] - self.target_xy))
        return self._observation(), reward, success, truncated, self._info()

    def close(self):
        self.renderer.close()
