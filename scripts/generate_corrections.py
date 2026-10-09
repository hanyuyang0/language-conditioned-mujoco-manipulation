"""Copy the baseline dataset and append fresh expert recoveries to Train only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from vla_sim.env import LanguageManipulationEnv, TASK_INSTRUCTIONS
from vla_sim.expert import expert_action
from vla_sim.data import encode_instruction, validate_episode_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='datasets/mujoco_language_v2')
    parser.add_argument('--output', default='datasets/mujoco_language_v3_corrections')
    parser.add_argument('--episodes-per-task', type=int, default=100)
    parser.add_argument('--seed', type=int, default=314159)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    if args.episodes_per_task <= 0:
        parser.error('episodes-per-task must be positive')
    if output.exists():
        parser.error('Output already exists; choose a fresh directory to avoid overwriting')
    meta = json.loads((source / 'metadata.json').read_text(encoding='utf-8'))
    split = json.loads((source / 'split.json').read_text(encoding='utf-8'))
    ids = [r['episode'] for r in meta['episodes']]
    validate_episode_split(split, ids)
    original_split = copy.deepcopy(split)
    output.mkdir(parents=True)
    for episode_id in ids:
        name = f'episode_{episode_id:05d}.npz'
        shutil.copy2(source / name, output / name)
    rng = np.random.default_rng(args.seed)
    env = LanguageManipulationEnv(meta['image_size'], 120, args.seed)
    next_id = max(ids) + 1
    try:
        for task in TASK_INSTRUCTIONS:
            for index in range(args.episodes_per_task):
                # Fresh scenes independent of held-out validation/test trajectories.
                obj = rng.uniform([-0.30, -0.22], [0., 0.22])
                target = rng.uniform([0.14, -0.25], [0.36, 0.25])
                near = index % 2 == 0
                offset = rng.uniform(-0.025 if near else -0.12, 0.025 if near else 0.12, size=2)
                gripper = np.r_[np.clip(obj + offset, [-0.48, -0.40], [0.48, 0.40]),
                                 rng.uniform(0.055, 0.08) if near else rng.uniform(0.08, 0.24)]
                instruction = str(rng.choice(TASK_INSTRUCTIONS[task]))
                obs, _ = env.reset(options=dict(task=task, instruction=instruction,
                    object_xy=obj, target_xy=target, gripper_xyz=gripper))
                images, states, actions = [], [], []
                success = False
                for _ in range(120):
                    action = expert_action(env)
                    images.append(obs['image']); states.append(obs['state']); actions.append(action)
                    obs, _, success, truncated, _ = env.step(action)
                    if success or truncated:
                        break
                if not success:
                    raise RuntimeError(f'Expert failed {task} correction {index}; inspect before training')
                np.savez_compressed(output / f'episode_{next_id:05d}.npz',
                    images=np.asarray(images, dtype=np.uint8), states=np.asarray(states, dtype=np.float32),
                    actions=np.asarray(actions, dtype=np.float32), tokens=encode_instruction(instruction, meta['vocab']),
                    task=np.asarray(task), instruction=np.asarray(instruction), success=np.asarray(True))
                meta['episodes'].append(dict(episode=next_id, task=task, instruction=instruction,
                    steps=len(actions), success=True, source='expert_correction', generation_seed=args.seed))
                split['train'].append(next_id)
                next_id += 1
            print(f'{task}: {args.episodes_per_task}/{args.episodes_per_task} corrections successful')
    finally:
        env.close()
    for name in ('val', 'test'):
        assert split[name] == original_split[name]
        for episode_id in split[name]:
            filename = f'episode_{episode_id:05d}.npz'
            assert hashlib.sha256((source / filename).read_bytes()).digest() == hashlib.sha256((output / filename).read_bytes()).digest()
    split['task_counts'] = {task: {name: sum(r['task'] == task and r['episode'] in split[name]
        for r in meta['episodes']) for name in ('train', 'val', 'test')} for task in TASK_INSTRUCTIONS}
    split['augmentation'] = dict(seed=args.seed, source=str(source.resolve()), method='randomized_gripper_start')
    validate_episode_split(split, [r['episode'] for r in meta['episodes']])
    (output / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    (output / 'split.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    print(f'Train={len(split["train"])}, Val={len(split["val"])}, Test={len(split["test"])}; held-out files unchanged')


if __name__ == '__main__':
    main()
