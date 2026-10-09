from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from vla_sim.data import build_vocab, encode_instruction
from vla_sim.env import LanguageManipulationEnv, TASK_INSTRUCTIONS
from vla_sim.expert import expert_action


def main():
    parser = argparse.ArgumentParser(description="Generate scripted MuJoCo demonstrations")
    parser.add_argument("--output", default="datasets/mujoco_language_new")
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    all_instructions = sum(TASK_INSTRUCTIONS.values(), [])
    vocab = build_vocab(all_instructions)
    env = LanguageManipulationEnv(args.image_size, args.max_steps, args.seed)
    rng = np.random.default_rng(args.seed)
    records, episode_id = [], 0
    try:
        for task in TASK_INSTRUCTIONS:
            successes = 0
            for _ in tqdm(range(args.episodes_per_task), desc=task):
                instruction = str(rng.choice(TASK_INSTRUCTIONS[task]))
                obs, _ = env.reset(options={"task": task, "instruction": instruction})
                images, states, actions = [], [], []
                success = False
                for _step in range(args.max_steps):
                    action = expert_action(env)
                    images.append(obs["image"])
                    states.append(obs["state"])
                    actions.append(action)
                    obs, _, success, truncated, _ = env.step(action)
                    if success or truncated:
                        break
                if success:
                    successes += 1
                np.savez_compressed(
                    output / f"episode_{episode_id:05d}.npz",
                    images=np.asarray(images, dtype=np.uint8),
                    states=np.asarray(states, dtype=np.float32),
                    actions=np.asarray(actions, dtype=np.float32),
                    tokens=encode_instruction(instruction, vocab),
                    task=np.asarray(task), instruction=np.asarray(instruction), success=np.asarray(success),
                )
                records.append({"episode": episode_id, "task": task, "instruction": instruction,
                                "steps": len(actions), "success": bool(success)})
                episode_id += 1
            print(f"expert success {task}: {successes}/{args.episodes_per_task}")
    finally:
        env.close()
    metadata = {"format_version": 1, "image_size": args.image_size, "state_dim": 9,
                "action_dim": 4, "vocab": vocab, "episodes": records}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {episode_id} episodes to {output.resolve()}")


if __name__ == "__main__":
    main()
