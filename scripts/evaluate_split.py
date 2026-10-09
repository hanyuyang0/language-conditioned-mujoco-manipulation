from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from vla_sim.checkpoint import load_checkpoint
from vla_sim.data import validate_episode_split
from vla_sim.inference import predict_action
from vla_sim.env import LanguageManipulationEnv
from vla_sim.diagnostics import diagnose, save_trace


def load_episode_initial_conditions(path: Path) -> dict:
    """Recover the exact initial scene recorded in one demonstration."""
    episode = np.load(path, allow_pickle=False)
    state = episode["states"][0]
    return {
        "task": str(episode["task"].item()),
        "instruction": str(episode["instruction"].item()),
        "object_xy": state[3:5].astype(float),
        "target_xy": state[6:8].astype(float),
        "gripper_xyz": state[:3].astype(float),
        "expert_success": bool(episode["success"].item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop evaluation on an episode-level dataset split")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="datasets/mujoco_language_v2")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", default="outputs/split_evaluation")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--videos-per-task", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--image-mode', choices=['normal', 'black', 'shuffle'], default='normal')
    parser.add_argument('--shuffle-images', default=None,
                        help='NPZ bank with images [N,H,W,3]; required for shuffle')
    parser.add_argument('--image-seed', type=int, default=17001)
    args = parser.parse_args()
    image_bank = None
    image_rng = np.random.default_rng(args.image_seed)
    if args.image_mode == 'shuffle':
        if not args.shuffle_images:
            parser.error('--shuffle-images is required for shuffle mode')
        with np.load(args.shuffle_images, allow_pickle=False) as bank:
            image_bank = bank['images'].copy()
        if image_bank.ndim != 4 or image_bank.shape[1:] != (args.image_size, args.image_size, 3) or not len(image_bank):
            parser.error('Shuffle bank has invalid image dimensions')

    data_root = Path(args.data)
    split_path = Path(args.split_file) if args.split_file else data_root / "split.json"
    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    all_files = sorted(data_root.glob("episode_*.npz"))
    all_ids = [int(path.stem.split("_")[-1]) for path in all_files]
    validate_episode_split(split_data, all_ids)
    episode_ids = list(map(int, split_data[args.split]))
    if not episode_ids:
        raise ValueError(f"Split {args.split!r} is empty")

    device = torch.device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = LanguageManipulationEnv(args.image_size, args.max_steps, seed=0)
    results: list[dict] = []
    latencies: list[float] = []
    video_counts: dict[str, int] = {}

    try:
        for episode_id in episode_ids:
            source = data_root / f"episode_{episode_id:05d}.npz"
            initial = load_episode_initial_conditions(source)
            task = initial["task"]
            instruction = initial["instruction"]
            observation, _ = env.reset(
                seed=episode_id,
                options={
                    "task": task,
                    "instruction": instruction,
                    "object_xy": initial["object_xy"],
                    "target_xy": initial["target_xy"],
                    "gripper_xyz": initial["gripper_xyz"],
                },
            )
            frames: list[np.ndarray] = []
            trace = []
            # Fixed unrelated image for the whole episode; never use held-out labels.
            shuffle_index = int(image_rng.integers(len(image_bank))) if image_bank is not None else None
            success = False
            info = {}
            for step in range(args.max_steps):
                frames.append(observation["image"])
                policy_observation = observation
                if args.image_mode != 'normal':
                    policy_observation = dict(observation)
                    policy_observation['image'] = (np.zeros_like(observation['image'])
                        if args.image_mode == 'black' else image_bank[shuffle_index])
                action, latency = predict_action(model, payload, policy_observation, instruction, device)
                latencies.append(latency)
                before = observation['state'].copy()
                observation, _, success, truncated, info = env.step(action)
                after = observation['state']
                row = {'step': step + 1,
                       'attached_before': bool(before[8]), 'attached_after': bool(after[8]),
                       'grasp_xy_distance': float(np.linalg.norm(before[:2] - before[3:5])),
                       'grasp_z_distance': float(abs(before[2] - before[5])),
                       'distance_after': float(info['distance'])}
                for prefix, values in [('before', before), ('after', after)]:
                    for name, value in zip(['gripper_x','gripper_y','gripper_z','object_x','object_y','object_z','target_x','target_y','attached'], values):
                        row[f'{prefix}_{name}'] = float(value)
                for name, value in zip(['action_x','action_y','action_z','action_gripper'], action):
                    row[name] = float(value)
                trace.append(row)
                if success or truncated:
                    break

            result = {
                "episode_id": episode_id,
                "shuffle_image_index": shuffle_index,
                "task": task,
                "instruction": instruction,
                "success": bool(success),
                "steps": step + 1,
                "final_distance": float(info["distance"]),
                "final_attached": bool(info["attached"]),
                "expert_success": initial["expert_success"],
                "object_xy": initial["object_xy"].tolist(),
                "target_xy": initial["target_xy"].tolist(),
            }
            results.append(result)
            diagnosis = diagnose(task, trace, success)
            result['diagnosis'] = diagnosis
            result['trace'] = save_trace(output, f'{task}_episode_{episode_id:05d}', trace, diagnosis)
            frames.append(observation['image'])
            status = "success" if success else "failure"
            print(
                f"episode={episode_id:05d} task={task:10s} success={success} "
                f"steps={step + 1:3d} distance={info['distance']:.4f}"
            )

            count = video_counts.get(task, 0)
            if count < args.videos_per_task or not success:
                video_path = output / f"{task}_{status}_episode_{episode_id:05d}.mp4"
                imageio.mimsave(video_path, frames, fps=20)
                result["video"] = str(video_path.resolve())
                if success:
                    video_counts[task] = count + 1
    finally:
        env.close()

    tasks = sorted({row["task"] for row in results})
    per_task = {}
    for task in tasks:
        rows = [row for row in results if row["task"] == task]
        per_task[task] = {
            "episodes": len(rows),
            "successes": sum(row["success"] for row in rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "mean_steps": float(np.mean([row["steps"] for row in rows])),
            "mean_final_distance": float(np.mean([row["final_distance"] for row in rows])),
        }
    summary = {
        "split": args.split,
        "episodes": len(results),
        "successes": sum(row["success"] for row in results),
        "success_rate": float(np.mean([row["success"] for row in results])),
        "mean_inference_ms": float(np.mean(latencies) * 1000),
        "per_task": per_task,
    }
    report = {
        'image_mode': args.image_mode,
        'state_mode': payload.get('state_mode', 'full'),
        'visual_mode': payload.get('visual_mode', 'global'),
        'image_seed': args.image_seed,
        'shuffle_images': args.shuffle_images,
        "summary": summary,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data": str(data_root.resolve()),
        "split_file": str(split_path.resolve()),
        "episodes": results,
    }
    report_path = output / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved: {report_path.resolve()}")


if __name__ == "__main__":
    main()
