from __future__ import annotations

import argparse
import time

import mujoco.viewer
import torch

from vla_sim.checkpoint import load_checkpoint
from vla_sim.inference import predict_action
from vla_sim.env import LanguageManipulationEnv, TASK_INSTRUCTIONS
from vla_sim.expert import expert_action


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive MuJoCo expert/policy viewer")
    parser.add_argument("--mode", choices=["expert", "policy"], default="expert")
    parser.add_argument("--task", choices=list(TASK_INSTRUCTIONS), default="push")
    parser.add_argument("--checkpoint", default="runs/act_robot_spatial_seed42/best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--auto-reset-seconds", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = payload = None
    if args.mode == "policy":
        model, payload = load_checkpoint(args.checkpoint, device)

    env = LanguageManipulationEnv(image_size=96, max_steps=args.max_steps, seed=args.seed)
    state = {"paused": False, "reset": False, "switch": False}
    tasks = list(TASK_INSTRUCTIONS)
    task = args.task
    episode = 0

    def on_key(key: int) -> None:
        if key == 32:  # Space
            state["paused"] = not state["paused"]
        elif key in (ord("R"), ord("r")):
            state["reset"] = True
        elif key in (ord("T"), ord("t")):
            state["switch"] = True

    def reset(current_task: str):
        instruction = TASK_INSTRUCTIONS[current_task][episode % len(TASK_INSTRUCTIONS[current_task])]
        observation, _ = env.reset(
            seed=args.seed + episode,
            options={"task": current_task, "instruction": instruction},
        )
        print(f"\nEpisode {episode} | mode={args.mode} | task={current_task}")
        print(f"Instruction: {instruction}")
        return observation, instruction

    obs, instruction = reset(task)
    frame_period = 1.0 / args.fps
    print("Controls: Space=pause/resume, R=reset, T=switch task, close window=exit")

    try:
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
            viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
            viewer.cam.distance = 1.25
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -55
            viewer.sync()
            while viewer.is_running():
                start = time.perf_counter()
                if state["switch"]:
                    task = tasks[(tasks.index(task) + 1) % len(tasks)]
                    state["switch"] = False
                    state["reset"] = True
                if state["reset"]:
                    episode += 1
                    obs, instruction = reset(task)
                    state["reset"] = False
                if not state["paused"]:
                    action = (
                        expert_action(env)
                        if args.mode == "expert"
                        else predict_action(model, payload, obs, instruction, device)[0]
                    )
                    obs, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        print(
                            f"Finished: success={info['success']}, steps={env.step_count}, "
                            f"distance={info['distance']:.4f}"
                        )
                        viewer.sync()
                        time.sleep(max(0.0, args.auto_reset_seconds))
                        episode += 1
                        obs, instruction = reset(task)
                viewer.sync()
                remaining = frame_period - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        env.close()


if __name__ == "__main__":
    main()
