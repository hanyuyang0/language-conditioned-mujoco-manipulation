"""Evaluate frozen policies on shared, reproducible position suites."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def sample_scene(rng, suite):
    if suite == 'id':
        obj = rng.uniform([-0.30, -0.22], [0.0, 0.22])
        target = rng.uniform([0.14, -0.25], [0.36, 0.25])
    elif suite == 'boundary':
        obj = rng.uniform([-0.30, -0.22], [0.0, 0.22])
        target = rng.uniform([0.14, -0.25], [0.36, 0.25])
        # Stay inside training support, close to one selected boundary.
        axis = int(rng.integers(2))
        obj[axis] = ([-0.30, -0.22][axis] + rng.uniform(0, .01)
                     if rng.random() < .5 else [0., .22][axis] - rng.uniform(0, .01))
        axis = int(rng.integers(2))
        target[axis] = ([.14, -.25][axis] + rng.uniform(0, .01)
                        if rng.random() < .5 else [.36, .25][axis] - rng.uniform(0, .01))
    else:
        # Both object and target lie outside training Y support, within workspace.
        obj = np.array([rng.uniform(-.30, 0), rng.choice([-1, 1]) * rng.uniform(.27, .34)])
        target = np.array([rng.uniform(.14, .36), rng.choice([-1, 1]) * rng.uniform(.29, .35)])
    return obj, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--act-checkpoint', default='runs/act_v2_seed42/best.pt')
    parser.add_argument('--bc-checkpoint', default='runs/bc_gripper_cls_corrections_seed42/best.pt')
    parser.add_argument('--output', default='outputs/position_generalization_seed90001')
    parser.add_argument('--episodes-per-task', type=int, default=50)
    parser.add_argument('--seed', type=int, default=90001)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--videos-per-task', type=int, default=2)
    args = parser.parse_args()
    if args.episodes_per_task <= 0:
        parser.error('episodes-per-task must be positive')
    for checkpoint in [args.act_checkpoint, args.bc_checkpoint]:
        if not Path(checkpoint).is_file():
            parser.error(f'Checkpoint missing: {checkpoint}')
    output = Path(args.output)
    if output.exists():
        parser.error('Choose a fresh output directory to preserve previous results')
    output.mkdir(parents=True)
    # The evaluator accepts NPZ initial-state fixtures, not expert demonstrations.
    from vla_sim.env import TASK_INSTRUCTIONS
    results = {}
    manifest = {'seed': args.seed, 'episodes_per_task': args.episodes_per_task,
                'scene_type': 'synthetic evaluation fixtures, not training data',
                'suites': {}}
    for suite_index, suite in enumerate(['id', 'boundary', 'ood']):
        rng = np.random.default_rng(args.seed + suite_index)
        directory = output / 'scenarios' / suite
        directory.mkdir(parents=True)
        records = []
        for task in TASK_INSTRUCTIONS:
            for i in range(args.episodes_per_task):
                obj, target = sample_scene(rng, suite)
                instruction = TASK_INSTRUCTIONS[task][i % len(TASK_INSTRUCTIONS[task])]
                episode_id = len(records)
                state = np.r_[[-.4, 0, .18], obj, .035, target, 0].astype(np.float32)
                np.savez_compressed(directory / f'episode_{episode_id:05d}.npz', states=state[None],
                                    task=np.asarray(task), instruction=np.asarray(instruction),
                                    success=np.asarray(False))
                records.append(dict(episode_id=episode_id, task=task, instruction=instruction,
                                    object_xy=obj.tolist(), target_xy=target.tolist()))
        (directory / 'split.json').write_text(json.dumps(
            {'train': [], 'val': [], 'test': list(range(len(records)))}, indent=2), encoding='utf-8')
        manifest['suites'][suite] = records
        (output / 'scenarios.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        results[suite] = {}
        for label, checkpoint in [('act', args.act_checkpoint), ('bc', args.bc_checkpoint)]:
            destination = output / suite / label
            subprocess.run([sys.executable, str(Path(__file__).with_name('evaluate_split.py')),
                '--checkpoint', checkpoint, '--data', str(directory), '--split', 'test',
                '--output', str(destination), '--device', args.device,
                '--videos-per-task', str(args.videos_per_task)], check=True)
            report = json.loads((destination / 'evaluation.json').read_text(encoding='utf-8'))
            # There are no expert rollouts for these generated scenes.
            for episode in report['episodes']:
                episode['expert_success'] = None
            report['suite'] = suite
            (destination / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            results[suite][label] = report['summary']
        (output / 'comparison.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('\nSuite       Model    Success   Push     Pick-place')
    for suite, models in results.items():
        for label, summary in models.items():
            print(f"{suite:11s} {label:7s} {summary['success_rate']:8.1%} "
                  f"{summary['per_task']['push']['success_rate']:8.1%} "
                  f"{summary['per_task']['pick_place']['success_rate']:8.1%}")
    print(f'Results: {output.resolve()}')


if __name__ == '__main__':
    main()
